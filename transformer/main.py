import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import random
import pandas as pd 
import wandb

# Import your dataset and other utilities
from dataset import MURADataset, custom_collate
from plot import plot_learning_curves, visualize_results, find_optimal_threshold

# Import the attention model components
from train import (TransformerMURAClassifier, train_transformer_model, 
                            evaluate_transformer_model, visualize_transformer_attention,
                            MarginWeightedBCELoss)

def make_training_deterministic(seed=42):
    # Set environment variable for CuBLAS
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Python RNG
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Numpy RNG
    np.random.seed(seed)

    # PyTorch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDA operations
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Enable deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        print("Warning: torch.use_deterministic_algorithms not supported in this PyTorch version")

def main(backbone='densenet169', threshold=0.5, lr=0.0001, batch_size=8, 
         num_epochs=20, patience=3, margin=0.15, margin_weight=0.3, warmup_epochs=2, seed=42):

    wandb.init(
        project="mura-fracture-detection",
        config={
            "architecture": backbone,
            "model_type": "transformer",
            "threshold": threshold,
            "lr": lr,
            "batch_size": batch_size,
            "loss_fn": "MarginWeightedBCE",
            "scheduler": "Cosine",
            "num_epochs": num_epochs,
            "seed": seed,
            "patience": patience,
            "margin": margin,
            "margin_weight": margin_weight
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CSV file paths
    train_csv = "../csv_files/processed_all_train.csv"
    valid_csv = "../csv_files/processed_all_valid.csv"
    test_csv = "../csv_files/processed_all_test.csv"

    # Set random seeds for reproducibility
    make_training_deterministic(seed)

    # Define data transformations
    train_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    # Add these new augmentations:
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Helps with dark images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    val_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = MURADataset(csv_file=train_csv, transform=train_transform, base_path="../")
    val_dataset = MURADataset(csv_file=valid_csv, transform=val_transform, base_path="../")
    test_dataset = MURADataset(csv_file=test_csv, transform=val_transform, base_path="../")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=8,   
        collate_fn=custom_collate,
        pin_memory=True,
        worker_init_fn=None,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,    
        collate_fn=custom_collate,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8, 
        collate_fn=custom_collate,
        pin_memory=True
    )

    # Create the attention-based model
    model = TransformerMURAClassifier(
        backbone=backbone, 
        pretrained=True, 
        threshold=threshold,
        embed_dim=512,       # Feature dimension in transformer
        num_heads=8,         # Number of attention heads
        transformer_layers=2  # Number of transformer layers
    )
    model = model.to(device)

    # Compute class weights for balanced loss
    df_train = pd.read_csv(train_csv)
    num_abnormal = df_train['label'].sum()
    num_normal = len(df_train) - num_abnormal
    print(f"Training data: {num_normal} normal samples, {num_abnormal} abnormal samples")

    # Define custom Margin Weighted BCE Loss
    criterion = MarginWeightedBCELoss(
        abnormal_count=num_abnormal, 
        normal_count=num_normal,
        margin=margin
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,  # Slightly higher learning rate works well with AdamW
        betas=(0.9, 0.999),
        weight_decay=0.01  
    )

    # Different parameter groups with different learning rates
    # encoder_params = list(model.feature_extractor.parameters())
    # transformer_params = list(model.transformer_encoder.parameters()) + \
    #                     list(model.feature_embedding.parameters()) + \
    #                     list([model.cls_token])
    # classifier_params = list(model.classifier.parameters()) + list(model.image_classifier.parameters())

    # optimizer = torch.optim.AdamW([
    #     {'params': encoder_params, 'lr': 1e-5, 'weight_decay': 0.01},  # Slower for pretrained backbone
    #     {'params': transformer_params, 'lr': 5e-5, 'weight_decay': 0.1},  # Higher regularization for transformer
    #     {'params': classifier_params, 'lr': 1e-4, 'weight_decay': 0.01}  # Faster for classifiers
    # ])

    # Learning rate scheduler
    # scheduler = ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.1, patience=patience, verbose=True
    # )

   # Calculate the total number of training steps
    num_batches_per_epoch = len(train_loader)
    num_training_steps = num_batches_per_epoch * num_epochs 
    num_warmup_steps = num_batches_per_epoch * warmup_epochs 

    # Initialize the cosine schedule with warmup
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,  # Default value - half cosine cycle
        last_epoch=-1  # Start from the beginning
    )

    # Train model
    print(f"Starting training for {num_epochs} epochs...")
    # Instead of train_attention_model
    model, best_val_metrics, history = train_transformer_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        threshold=threshold
    )
    

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = evaluate_transformer_model(model, test_loader, criterion, device, threshold=threshold)

    # Print test results
    print("\nTest Results:")
    print(f"Loss: {test_results['loss']:.4f}")
    print(f"Accuracy: {test_results['acc']:.4f}")
    print(f"AUC: {test_results['auc']:.4f}")
    print(f"F1 Score: {test_results['f1']:.4f}")

    # Visualize results
    visualize_results(test_results)

    # Visualize attention weights for a few samples
    print("\nVisualizing attention weights...")
    visualize_transformer_attention(model, test_loader, num_samples=5, device=device)

    # Find optimal threshold
    best_threshold = find_optimal_threshold(test_results)
    print(f"Optimal threshold: {best_threshold:.4f}")

    # Log final results to wandb
    wandb.log({
        "Test Loss": test_results['loss'],
        "Test Accuracy": test_results['acc'],
        "Test AUC": test_results['auc'],
        "Test F1": test_results['f1'],
        "Best Threshold": best_threshold
    })

    # Save the model
    model_filename = f'mura_attention_model_{backbone}_seed{seed}.pth'
    state_dict_filename = f'mura_attention_model_state_dict_{backbone}_seed{seed}.pth'
    
    print(f"Saving model to {model_filename} and {state_dict_filename}")
    torch.save(model, model_filename)
    torch.save(model.state_dict(), state_dict_filename)

    return model, test_results, best_threshold


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    model, test_results, best_threshold = main(
        backbone='densenet169',
        threshold=0.35,
        lr=0.0005,
        batch_size=16,
        num_epochs=30,
        patience=3,
        margin=0.15,
        margin_weight=0.3,
        seed=42
    )
    