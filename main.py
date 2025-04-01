import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import MURADataset, custom_collate
from train import MURAClassifier, FocalLoss, train_model, evaluate_model, visualize_results, find_optimal_threshold
import random
import os
import pandas as pd 
from plot import plot_learning_curves, show_augmented_image
import wandb

# from torchtune.modules import get_cosine_schedule_with_warmup 

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def compute_class_weights(csv_file):
    """Compute class weights for weighted BCE loss."""
    df = pd.read_csv(csv_file)
    num_abnormal = df['label'].sum()  # Count of abnormal (1s)
    num_normal = len(df) - num_abnormal  # Count of normal (0s)
    total = num_abnormal + num_normal

    weight_abnormal = num_normal / total  # w_T,1
    weight_normal = num_abnormal / total  # w_T,0

    return torch.tensor([weight_normal, weight_abnormal], dtype=torch.float32)

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

def main(backbone='densenet169', threshold=0.5, lr=0.0001, batch_size=8, agg_type='prob_mean', alpha=0.75, gamma=2.0, num_epochs=50, seed=42):

    wandb.init(
        project="mura-fracture-detection",
        config={
            "architecture": backbone,
            "aggregation": agg_type,
            "threshold": threshold,
            "lr": lr,
            "batch_size": batch_size,
            "loss_fn": "Weighted BCE",
            "scheduler": "ReduceLROnPlateau",
            "num_epochs": num_epochs,
            "seed": seed,
            "alpha": alpha,
            "gamma": gamma
        }
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_csv = "csv_files/processed_all_train.csv"
    valid_csv = "csv_files/processed_all_valid.csv"
    test_csv = "csv_files/processed_all_test.csv"

    make_training_deterministic(seed)

    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),  # Resize to 320×320
        transforms.RandomHorizontalFlip(p=0.5),  # Random lateral inversion
        transforms.RandomRotation(degrees=30),  # Rotate up to 30 degrees
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    val_transform = transforms.Compose([
        transforms.Resize((320, 320)),  # Resize to 320×320
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])


    train_dataset = MURADataset(csv_file=train_csv, transform=train_transform)
    val_dataset = MURADataset(csv_file=valid_csv, transform=val_transform)
    test_dataset = MURADataset(csv_file=test_csv, transform=val_transform)

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


    # When creating the model:
    model = MURAClassifier(backbone=backbone, pretrained=True, agg_strategy=agg_type, threshold=threshold)
    model = model.to(device)

    # Compute class weights from training data
    class_weights = compute_class_weights(train_csv).to(device)
    
    # Define Weighted BCE Loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])  # pos_weight applies weight to class 1

    # Use lower learning rate with weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    # # Learning rate scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=0.1, patience=3, verbose=True
    # )

    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=256, num_training_steps=num_epochs*len(train_loader))

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Train model
    model, best_val_metrics, history = train_model(
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
    test_results = evaluate_model(model, test_loader, criterion, device, threshold=threshold)

    # Print test results
    print("\nTest Results:")
    print(f"Loss: {test_results['loss']:.4f}")
    print(f"Accuracy: {test_results['acc']:.4f}")
    print(f"AUC: {test_results['auc']:.4f}")
    print(f"F1 Score: {test_results['f1']:.4f}")


    # Visualize results
    visualize_results(test_results)

    # Find optimal threshold
    best_threshold = find_optimal_threshold(test_results)

    wandb.log({
        "Train Loss": test_results['loss'],
        "Test Accuracy": test_results['acc'],
        "Test AUC": test_results['auc'],
        "Test F1": test_results['f1'],
        "Best Threshold": best_threshold
    })

    # Save the model
    torch.save(model, f'mura_model_{agg_type}_seed{seed}.pth')  # Saves full model
    torch.save(model.state_dict(), f'mura_model_state_dict_{agg_type}_seed{seed}.pth')  # Saves only the state_dict

    return model, test_results, best_threshold


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    model, test_results, best_threshold = main(threshold=0.4, agg_type='prob_mean', alpha=0.75, gamma=2.0, num_epochs=30, seed=42)
