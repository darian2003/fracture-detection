import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import argparse

# Assuming these are in your project directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from dataset import MURADataset, custom_collate
from train import AttentionMURAClassifier, WeightedBCELoss

# Function to visualize studies with high attention variance
def visualize_high_attention_variance_studies(model, data_loader, num_samples=5, variance_threshold=0.03, 
                                              device='cuda', save_dir='attention_visualizations'):
    """
    Visualize studies where there's a high variance in attention weights between images.
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    samples_seen = 0
    studies_processed = 0
    
    print(f"Looking for studies with high attention weight variance (threshold: {variance_threshold})...")
    
    with torch.no_grad():
        for batch in data_loader:
            if samples_seen >= num_samples:
                break
                
            images_list = [imgs.to(device) for imgs in batch['images']]
            labels = batch['label'].to(device)
            
            # Get logits and attention weights
            study_logits, image_logits, attention_weights, masks = model(images_list)
            study_probs = torch.sigmoid(study_logits).cpu().numpy()
            
            # Calculate variance of attention weights for each study
            for i in range(len(images_list)):
                studies_processed += 1
                # Get attention weights for this study
                study_mask = masks[i].squeeze(-1)
                study_attention = attention_weights[i].squeeze(-1)
                real_mask = study_mask.bool()
                
                if real_mask.sum() <= 1:
                    continue  # Skip studies with only one image
                
                real_attention = study_attention[real_mask]
                
                # Normalize to sum to 1
                if real_attention.size(0) > 0:
                    real_attention = real_attention / (real_attention.sum() + 1e-8)
                    weights = real_attention.cpu().numpy()
                    num_images = len(weights)
                    
                    # Calculate variance of attention weights
                    weight_variance = np.var(weights)
                    
                    # Only visualize if variance exceeds threshold
                    if weight_variance < variance_threshold:
                        continue
                        
                    # Create figure for this study
                    fig, axes = plt.subplots(1, num_images + 1, figsize=(num_images * 4 + 4, 4))
                    
                    if num_images == 1:
                        axes = [axes[0], axes[1]]  # Handle the case where axes isn't a list
                    
                    # Plot the images with their attention weights
                    for j in range(num_images):
                        # Convert image tensor to numpy array for display
                        img = images_list[i][j].cpu().permute(1, 2, 0).numpy()
                        
                        # Denormalize if needed
                        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                        img = np.clip(img, 0, 1)
                        
                        # Display image
                        axes[j].imshow(img)
                        axes[j].set_title(f"Weight: {weights[j]:.3f}")
                        axes[j].axis('off')
                    
                    # Plot the attention distribution
                    axes[-1].bar(range(num_images), weights)
                    axes[-1].set_xlabel('Image Index')
                    axes[-1].set_ylabel('Attention Weight')
                    axes[-1].set_title(f"Variance: {weight_variance:.4f}\nLabel: {labels[i].item()}, Pred: {study_probs[i][0]:.3f}")
                    
                    plt.tight_layout()
                    plt.savefig(f"{save_dir}/high_variance_sample_{samples_seen}.png")
                    plt.close()
                    
                    samples_seen += 1
                    print(f"Saved high variance study {samples_seen}: variance={weight_variance:.4f}, label={labels[i].item()}, pred={study_probs[i][0]:.3f}")
                    
                    if samples_seen >= num_samples:
                        break
    
    print(f"Found {samples_seen} studies with high attention variance from {studies_processed} total studies.")
    if samples_seen < num_samples:
        print(f"Note: Couldn't find {num_samples} studies exceeding the variance threshold of {variance_threshold}.")
        print(f"Consider lowering the threshold or processing more data.")


def visualize_misclassified_studies(model, data_loader, num_samples=5, 
                                   device='cuda', threshold=0.5, save_dir='attention_visualizations'):
    """
    Visualize studies that are misclassified by the model and their attention weight distributions.
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    samples_seen = 0
    studies_processed = 0
    false_positives = 0
    false_negatives = 0
    
    print("Looking for misclassified studies...")
    
    with torch.no_grad():
        for batch in data_loader:
            if samples_seen >= num_samples:
                break
                
            images_list = [imgs.to(device) for imgs in batch['images']]
            labels = batch['label'].to(device)
            
            # Forward pass
            study_logits, image_logits, attention_weights, masks = model(images_list)
            study_probs = torch.sigmoid(study_logits).cpu().numpy()
            predictions = (study_probs > threshold).astype(int)
            true_labels = labels.cpu().numpy()
            
            # Find misclassified studies
            for i in range(len(images_list)):
                studies_processed += 1
                if predictions[i][0] != true_labels[i]:
                    # This study was misclassified
                    
                    # Get normalized attention weights for real (non-padded) images
                    study_mask = masks[i].squeeze(-1)
                    study_attention = attention_weights[i].squeeze(-1)
                    real_mask = study_mask.bool()
                    real_image_attention = study_attention[real_mask]
                    
                    # Normalize to sum to 1
                    if real_image_attention.size(0) > 0:
                        real_image_attention = real_image_attention / (real_image_attention.sum() + 1e-8)
                    
                    weights = real_image_attention.cpu().numpy()
                    num_images = len(weights)
                    
                    if num_images == 0:
                        continue  # Skip empty studies
                    
                    # Track false positive vs false negative
                    if predictions[i][0] == 1 and true_labels[i] == 0:
                        error_type = "False Positive"
                        false_positives += 1
                    else:
                        error_type = "False Negative"
                        false_negatives += 1
                    
                    # Create figure for this study
                    fig, axes = plt.subplots(1, num_images + 1, figsize=(num_images * 4 + 4, 4))
                    
                    if num_images == 1:
                        axes = [axes[0], axes[1]]  # Handle the case where axes isn't a list
                    
                    # Plot the images with their attention weights
                    for j in range(num_images):
                        # Convert image tensor to numpy array for display
                        img = images_list[i][j].cpu().permute(1, 2, 0).numpy()
                        
                        # Denormalize if needed
                        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                        img = np.clip(img, 0, 1)
                        
                        # Display image
                        axes[j].imshow(img)
                        axes[j].set_title(f"Weight: {weights[j]:.3f}")
                        axes[j].axis('off')
                    
                    # Plot the attention distribution
                    axes[-1].bar(range(num_images), weights)
                    axes[-1].set_xlabel('Image Index')
                    axes[-1].set_ylabel('Attention Weight')
                    axes[-1].set_title(f"{error_type}\nTrue Label: {true_labels[i]}, Pred: {study_probs[i][0]:.3f}")
                    
                    plt.tight_layout()
                    plt.savefig(f"{save_dir}/misclassified_{error_type.lower().replace(' ', '_')}_{samples_seen}.png")
                    plt.close()
                    
                    samples_seen += 1
                    print(f"Saved misclassified study {samples_seen}: {error_type}, true={true_labels[i]}, pred={study_probs[i][0]:.3f}")
                    
                    if samples_seen >= num_samples:
                        break
    
    print(f"Found {samples_seen} misclassified studies from {studies_processed} total studies.")
    print(f"False Positives: {false_positives}, False Negatives: {false_negatives}")


def analyze_attention_patterns(model, data_loader, device='cuda', threshold=0.5, save_dir='attention_analysis'):
    """
    Perform a comprehensive analysis of attention patterns across the dataset.
    Correctly handles studies with different numbers of images.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # Initialize counters and data containers
    total_studies = 0
    correct_studies = 0
    wrong_studies = 0
    
    # Statistics containers
    attention_variance_correct = []
    attention_variance_wrong = []
    max_weight_correct = []
    max_weight_wrong = []
    
    # Study type counters
    normal_studies = 0
    abnormal_studies = 0
    normal_correct = 0
    abnormal_correct = 0
    
    # Store normalized attention by number of images and class
    attention_by_length = {
        'normal': {},    # Will be indexed by study length
        'abnormal': {}   # Will be indexed by study length
    }
    
    # Count studies by length
    study_lengths = {
        'normal': {},
        'abnormal': {}
    }
    
    # Store relative position data (first, middle, last)
    relative_positions = {
        'normal': {'first': [], 'middle': [], 'last': []},
        'abnormal': {'first': [], 'middle': [], 'last': []}
    }
    
    print("Analyzing attention patterns across the dataset...")
    
    with torch.no_grad():
        for batch in data_loader:
            images_list = [imgs.to(device) for imgs in batch['images']]
            labels = batch['label'].to(device)
            
            # Forward pass
            study_logits, image_logits, attention_weights, masks = model(images_list)
            study_probs = torch.sigmoid(study_logits).cpu().numpy()
            predictions = (study_probs > threshold).astype(int)
            true_labels = labels.cpu().numpy()
            
            # Analyze each study
            for i in range(len(images_list)):
                total_studies += 1
                
                # Get normalized attention weights for real (non-padded) images
                study_mask = masks[i].squeeze(-1)
                study_attention = attention_weights[i].squeeze(-1)
                real_mask = study_mask.bool()
                real_image_attention = study_attention[real_mask]
                
                # Normalize to sum to 1
                if real_image_attention.size(0) > 0:
                    real_image_attention = real_image_attention / (real_image_attention.sum() + 1e-8)
                
                weights = real_image_attention.cpu().numpy()
                num_images = len(weights)
                
                if num_images == 0:
                    continue  # Skip empty studies
                
                # Compute statistics
                attention_variance = np.var(weights) if num_images > 1 else 0
                max_attention = np.max(weights) if weights.size > 0 else 0
                
                # Track by correctness
                is_correct = predictions[i][0] == true_labels[i]
                if is_correct:
                    correct_studies += 1
                    attention_variance_correct.append(attention_variance)
                    max_weight_correct.append(max_attention)
                else:
                    wrong_studies += 1
                    attention_variance_wrong.append(attention_variance)
                    max_weight_wrong.append(max_attention)
                
                # Track by class (normal vs abnormal)
                class_key = 'normal' if true_labels[i] == 0 else 'abnormal'
                
                if class_key == 'normal':
                    normal_studies += 1
                    if is_correct:
                        normal_correct += 1
                else:
                    abnormal_studies += 1
                    if is_correct:
                        abnormal_correct += 1
                
                # Store by length and class
                if num_images not in attention_by_length[class_key]:
                    attention_by_length[class_key][num_images] = []
                    study_lengths[class_key][num_images] = 0
                
                attention_by_length[class_key][num_images].append(weights)
                study_lengths[class_key][num_images] += 1
                
                # Store data for relative position analysis
                if num_images > 0:
                    # First position
                    relative_positions[class_key]['first'].append(weights[0])
                    
                    # Last position
                    relative_positions[class_key]['last'].append(weights[-1])
                    
                    # Middle positions (if there are any)
                    if num_images > 2:
                        relative_positions[class_key]['middle'].extend(weights[1:-1])
    
    # Calculate average statistics
    print("\n--- Attention Analysis Results ---")
    print(f"Total studies: {total_studies}")
    print(f"Overall accuracy: {correct_studies / total_studies:.4f}")
    print(f"Normal studies: {normal_studies}, Accuracy: {normal_correct / normal_studies if normal_studies > 0 else 0:.4f}")
    print(f"Abnormal studies: {abnormal_studies}, Accuracy: {abnormal_correct / abnormal_studies if abnormal_studies > 0 else 0:.4f}")
    
    # Study length distribution
    print("\nStudy Length Distribution:")
    for class_key in ['normal', 'abnormal']:
        print(f"\n{class_key.capitalize()} studies:")
        for length, count in sorted(study_lengths[class_key].items()):
            print(f"  {length} images: {count} studies ({count/total_studies*100:.1f}%)")
    
    # Plot study length distribution
    plt.figure(figsize=(12, 6))
    
    # Get all unique lengths
    all_lengths = set()
    for class_key in ['normal', 'abnormal']:
        all_lengths.update(study_lengths[class_key].keys())
    all_lengths = sorted(all_lengths)
    
    # Prepare data for plotting
    normal_counts = [study_lengths['normal'].get(length, 0) for length in all_lengths]
    abnormal_counts = [study_lengths['abnormal'].get(length, 0) for length in all_lengths]
    
    x = np.arange(len(all_lengths))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, normal_counts, width, label='Normal', color='blue', alpha=0.7)
    rects2 = ax.bar(x + width/2, abnormal_counts, width, label='Abnormal', color='red', alpha=0.7)
    
    ax.set_xlabel('Number of Images in Study')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Study Lengths by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(all_lengths)
    ax.legend()
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/study_length_distribution.png")
    plt.close()
    
    # Attention variance analysis
    avg_variance_correct = np.mean(attention_variance_correct) if attention_variance_correct else 0
    avg_variance_wrong = np.mean(attention_variance_wrong) if attention_variance_wrong else 0
    print(f"\nAttention weight variance:")
    print(f"  Correct studies: {avg_variance_correct:.4f}")
    print(f"  Misclassified studies: {avg_variance_wrong:.4f}")
    
    # Max weight analysis
    avg_max_correct = np.mean(max_weight_correct) if max_weight_correct else 0
    avg_max_wrong = np.mean(max_weight_wrong) if max_weight_wrong else 0
    print(f"\nMax attention weight:")
    print(f"  Correct studies: {avg_max_correct:.4f}")
    print(f"  Misclassified studies: {avg_max_wrong:.4f}")
    
    # Plot attention patterns by study length for each class
    max_plot_lengths = 5  # Plot up to this many different study lengths
    
    # Get the most common study lengths
    common_lengths = []
    for length, count in sorted(
        {length: sum(study_lengths['normal'].get(length, 0) + study_lengths['abnormal'].get(length, 0) 
                    for class_key in ['normal', 'abnormal'])
         for length in all_lengths}.items(),
        key=lambda x: x[1], reverse=True):
        common_lengths.append(length)
        if len(common_lengths) >= max_plot_lengths:
            break
    
    # Plot average attention patterns for common study lengths
    plt.figure(figsize=(15, 10))
    
    subplot_rows = (len(common_lengths) + 1) // 2
    subplot_cols = min(2, len(common_lengths))
    
    for i, length in enumerate(common_lengths):
        plt.subplot(subplot_rows, subplot_cols, i + 1)
        
        for class_key, color, label in [('normal', 'blue', 'Normal'), ('abnormal', 'red', 'Abnormal')]:
            if length in attention_by_length[class_key] and attention_by_length[class_key][length]:
                # Average the attention patterns for this length and class
                avg_pattern = np.mean(attention_by_length[class_key][length], axis=0)
                plt.bar(range(length), avg_pattern, alpha=0.5, color=color, label=label)
        
        plt.title(f'Studies with {length} Images')
        plt.xlabel('Image Position')
        plt.ylabel('Average Attention Weight')
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/avg_attention_by_length_class.png")
    plt.close()
    
    # Plot relative position analysis
    plt.figure(figsize=(12, 6))
    positions = ['first', 'middle', 'last']
    
    normal_rel_means = [np.mean(relative_positions['normal'][pos]) if relative_positions['normal'][pos] else 0 
                       for pos in positions]
    abnormal_rel_means = [np.mean(relative_positions['abnormal'][pos]) if relative_positions['abnormal'][pos] else 0 
                         for pos in positions]
    
    x = np.arange(len(positions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, normal_rel_means, width, label='Normal', color='blue', alpha=0.7)
    rects2 = ax.bar(x + width/2, abnormal_rel_means, width, label='Abnormal', color='red', alpha=0.7)
    
    ax.set_xlabel('Relative Position in Study')
    ax.set_ylabel('Average Attention Weight')
    ax.set_title('Attention by Relative Position')
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.legend()
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/attention_by_relative_position.png")
    plt.close()
    
    # Plot attention variance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(attention_variance_correct, bins=20, alpha=0.5, label='Correct Predictions', color='green')
    plt.hist(attention_variance_wrong, bins=20, alpha=0.5, label='Incorrect Predictions', color='red')
    plt.xlabel('Attention Weight Variance')
    plt.ylabel('Count')
    plt.title('Distribution of Attention Weight Variance')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}/attention_variance_distribution.png")
    plt.close()
    
    # Plot max weight distribution
    plt.figure(figsize=(10, 6))
    plt.hist(max_weight_correct, bins=20, alpha=0.5, label='Correct Predictions', color='green')
    plt.hist(max_weight_wrong, bins=20, alpha=0.5, label='Incorrect Predictions', color='red')
    plt.xlabel('Maximum Attention Weight')
    plt.ylabel('Count')
    plt.title('Distribution of Maximum Attention Weight')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}/max_attention_distribution.png")
    plt.close()
    
    print(f"\nAnalysis results and plots saved to: {save_dir}")
    
    return {
        'total_studies': total_studies,
        'accuracy': correct_studies / total_studies if total_studies > 0 else 0,
        'normal_accuracy': normal_correct / normal_studies if normal_studies > 0 else 0,
        'abnormal_accuracy': abnormal_correct / abnormal_studies if abnormal_studies > 0 else 0,
        'avg_variance_correct': avg_variance_correct,
        'avg_variance_wrong': avg_variance_wrong,
        'avg_max_correct': avg_max_correct,
        'avg_max_wrong': avg_max_wrong,
        'study_lengths': study_lengths
    }


def evaluate_with_visualizations(model, data_loader, criterion=None, 
                                device='cuda', threshold=0.5, save_dir='model_analysis'):
    """
    Complete evaluation function that also generates visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Run regular evaluation metrics
    results = evaluate_model(model, data_loader, criterion, device, threshold)
    
    # Create subdirectories for each visualization type
    high_var_dir = os.path.join(save_dir, 'high_variance')
    misclass_dir = os.path.join(save_dir, 'misclassified')
    analysis_dir = os.path.join(save_dir, 'analysis')
    
    os.makedirs(high_var_dir, exist_ok=True)
    os.makedirs(misclass_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations for studies with high attention variance...")
    visualize_high_attention_variance_studies(
        model, data_loader, num_samples=5, variance_threshold=0.03, 
        device=device, save_dir=high_var_dir
    )
    
    print("\nGenerating visualizations for misclassified studies...")
    visualize_misclassified_studies(
        model, data_loader, num_samples=5, 
        device=device, threshold=threshold, save_dir=misclass_dir
    )
    
    print("\nGenerating comprehensive attention pattern analysis...")
    analysis_results = analyze_attention_patterns(
        model, data_loader, device=device, threshold=threshold, save_dir=analysis_dir
    )
    
    # Combine all results
    results.update(analysis_results)
    
    # Save summary to file
    with open(os.path.join(save_dir, 'results_summary.txt'), 'w') as f:
        f.write("=== MURA Classification Results ===\n\n")
        f.write(f"Accuracy: {results['acc']:.4f}\n")
        f.write(f"AUC: {results['auc']:.4f}\n")
        f.write(f"F1 Score: {results['f1']:.4f}\n\n")
        
        f.write("=== Attention Analysis ===\n\n")
        f.write(f"Total studies analyzed: {results['total_studies']}\n")
        f.write(f"Normal studies accuracy: {results['normal_accuracy']:.4f}\n")
        f.write(f"Abnormal studies accuracy: {results['abnormal_accuracy']:.4f}\n")
        f.write(f"Avg attention variance (correct): {results['avg_variance_correct']:.4f}\n")
        f.write(f"Avg attention variance (wrong): {results['avg_variance_wrong']:.4f}\n")
    
    print(f"Analysis complete. Results saved to {save_dir}")
    return results


def evaluate_model(model, data_loader, criterion=None, device='cuda', threshold=0.5):
    """
    Standard evaluation function for attention-based model
    """
    model.eval()
    val_loss = 0.0
    val_outputs = []
    val_labels = []
    attention_weights_list = []

    with torch.no_grad():
        for batch in data_loader:
            images_list = [imgs.to(device) for imgs in batch['images']]
            labels = batch['label'].to(device)

            # Forward pass with attention
            study_logits, image_logits, attention_weights, masks = model(images_list)
            
            # Calculate loss if criterion provided
            if criterion is not None:
                loss = criterion(study_logits, labels)
                val_loss += loss.item() * len(labels)
            
            # Track metrics
            val_outputs.extend(study_logits.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
            # Store attention weights for analysis
            for i in range(len(images_list)):
                study_mask = masks[i].squeeze(-1)
                study_attention = attention_weights[i].squeeze(-1)
                real_attention = study_attention[study_mask.bool()]
                attention_weights_list.append(real_attention.cpu().numpy())

    # Calculate metrics
    if criterion is not None:
        val_loss /= len(data_loader.dataset)
    else:
        val_loss = 0.0
        
    val_outputs = np.array(val_outputs).flatten()
    val_labels = np.array(val_labels).flatten()

    # Convert logits to probabilities
    val_probs = torch.sigmoid(torch.tensor(val_outputs)).numpy()
    val_preds = (val_probs > threshold).astype(int)

    # Calculate metrics
    val_acc = accuracy_score(val_labels, val_preds)
    val_auc = roc_auc_score(val_labels, val_probs)
    val_f1 = f1_score(val_labels, val_preds)
    
    # Print basic metrics
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {val_acc:.4f}")
    print(f"AUC: {val_auc:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    
    if criterion is not None:
        print(f"Loss: {val_loss:.4f}")

    results = {
        'loss': val_loss,
        'acc': val_acc,
        'auc': val_auc,
        'f1': val_f1,
        'probs': val_probs,
        'preds': val_preds,
        'labels': val_labels,
        'outputs': val_outputs,
        'attention_weights': attention_weights_list
    }

    return results

def analyze_accuracy_by_study_size(model, data_loader, device='cuda', threshold=0.5, save_dir='attention_analysis'):
    """
    Analyze how classification accuracy varies with the number of images in a study.
    
    Args:
        model: The trained model
        data_loader: DataLoader for the dataset
        device: Device to run inference on ('cuda' or 'cpu')
        threshold: Classification threshold
        save_dir: Directory to save results and plots
    
    Returns:
        Dictionary with accuracy metrics by study size
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from collections import defaultdict
    
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # Data containers
    study_sizes = defaultdict(int)  # Count of studies by size
    correct_by_size = defaultdict(int)  # Count of correct predictions by study size
    normal_by_size = defaultdict(int)  # Count of normal studies by size
    abnormal_by_size = defaultdict(int)  # Count of abnormal studies by size
    normal_correct_by_size = defaultdict(int)  # Count of correctly classified normal studies by size
    abnormal_correct_by_size = defaultdict(int)  # Count of correctly classified abnormal studies by size
    
    # Tracking confidence by study size
    confidence_by_size = defaultdict(list)  # Prediction confidence by study size
    
    print("Analyzing accuracy with respect to study size...")
    
    with torch.no_grad():
        for batch in data_loader:
            images_list = [imgs.to(device) for imgs in batch['images']]
            labels = batch['label'].to(device)
            
            # Forward pass
            study_logits, image_logits, attention_weights, masks = model(images_list)
            study_probs = torch.sigmoid(study_logits).cpu().numpy()
            predictions = (study_probs > threshold).astype(int)
            true_labels = labels.cpu().numpy()
            
            # Analyze each study
            for i in range(len(images_list)):
                # Get actual number of images in this study (excluding padding)
                study_mask = masks[i].squeeze(-1)
                num_images = study_mask.sum().int().item()
                
                if num_images == 0:
                    continue  # Skip empty studies
                
                # Update counters
                study_sizes[num_images] += 1
                pred_label = predictions[i][0]
                true_label = true_labels[i]
                
                # Track confidence (absolute distance from threshold)
                confidence = abs(study_probs[i][0] - threshold)
                confidence_by_size[num_images].append(confidence)
                
                # Check if prediction is correct
                if pred_label == true_label:
                    correct_by_size[num_images] += 1
                
                # Track by class
                if true_label == 0:  # Normal
                    normal_by_size[num_images] += 1
                    if pred_label == true_label:
                        normal_correct_by_size[num_images] += 1
                else:  # Abnormal
                    abnormal_by_size[num_images] += 1
                    if pred_label == true_label:
                        abnormal_correct_by_size[num_images] += 1
    
    # Calculate accuracy by study size
    accuracy_by_size = {}
    normal_accuracy_by_size = {}
    abnormal_accuracy_by_size = {}
    avg_confidence_by_size = {}
    
    for size in sorted(study_sizes.keys()):
        if study_sizes[size] > 0:
            accuracy_by_size[size] = correct_by_size[size] / study_sizes[size]
            
            if normal_by_size[size] > 0:
                normal_accuracy_by_size[size] = normal_correct_by_size[size] / normal_by_size[size]
            else:
                normal_accuracy_by_size[size] = 0
                
            if abnormal_by_size[size] > 0:
                abnormal_accuracy_by_size[size] = abnormal_correct_by_size[size] / abnormal_by_size[size]
            else:
                abnormal_accuracy_by_size[size] = 0
            
            if confidence_by_size[size]:
                avg_confidence_by_size[size] = np.mean(confidence_by_size[size])
            else:
                avg_confidence_by_size[size] = 0
    
    # Print results
    print("\n--- Accuracy by Study Size Analysis ---")
    print("Study Size | Total | Normal | Abnormal | Overall Acc | Normal Acc | Abnormal Acc")
    print("--------------------------------------------------------------------------")
    
    for size in sorted(study_sizes.keys()):
        print(f"{size:^10d} | {study_sizes[size]:^5d} | {normal_by_size[size]:^6d} | {abnormal_by_size[size]:^8d} | "
              f"{accuracy_by_size[size]:^11.4f} | {normal_accuracy_by_size[size]:^10.4f} | {abnormal_accuracy_by_size[size]:^12.4f}")
    
    # Plot accuracy by study size
    plt.figure(figsize=(12, 8))
    
    sizes = sorted(study_sizes.keys())
    overall_acc = [accuracy_by_size[size] for size in sizes]
    normal_acc = [normal_accuracy_by_size[size] for size in sizes]
    abnormal_acc = [abnormal_accuracy_by_size[size] for size in sizes]
    
    plt.plot(sizes, overall_acc, 'o-', label='Overall Accuracy', linewidth=2, markersize=8)
    plt.plot(sizes, normal_acc, 's-', label='Normal Cases Accuracy', linewidth=2, markersize=8)
    plt.plot(sizes, abnormal_acc, '^-', label='Abnormal Cases Accuracy', linewidth=2, markersize=8)
    
    plt.xlabel('Number of Images in Study', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Classification Accuracy by Study Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(sizes)
    plt.ylim(0, 1.05)
    
    # Add count labels above each point
    for size in sizes:
        plt.annotate(f"n={study_sizes[size]}", 
                    (size, accuracy_by_size[size] + 0.02),
                    ha='center', va='bottom',
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/accuracy_by_study_size.png")
    plt.close()
    
    # Plot prediction confidence by study size
    plt.figure(figsize=(12, 8))
    
    confidence_values = [avg_confidence_by_size[size] for size in sizes]
    
    plt.plot(sizes, confidence_values, 'o-', linewidth=2, markersize=8, color='purple')
    
    plt.xlabel('Number of Images in Study', fontsize=12)
    plt.ylabel('Average Prediction Confidence', fontsize=12)
    plt.title('Prediction Confidence by Study Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(sizes)
    
    # Add count labels above each point
    for size in sizes:
        plt.annotate(f"n={study_sizes[size]}", 
                    (size, avg_confidence_by_size[size] + 0.01),
                    ha='center', va='bottom',
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confidence_by_study_size.png")
    plt.close()
    
    # Plot distribution of study sizes
    plt.figure(figsize=(12, 8))
    
    # Create data for stacked bar chart
    normal_counts = [normal_by_size[size] for size in sizes]
    abnormal_counts = [abnormal_by_size[size] for size in sizes]
    
    plt.bar(sizes, normal_counts, label='Normal Studies', color='blue', alpha=0.7)
    plt.bar(sizes, abnormal_counts, bottom=normal_counts, label='Abnormal Studies', color='red', alpha=0.7)
    
    plt.xlabel('Number of Images in Study', fontsize=12)
    plt.ylabel('Number of Studies', fontsize=12)
    plt.title('Distribution of Study Sizes in Dataset', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(sizes)
    
    # Add count labels on each bar
    for i, size in enumerate(sizes):
        total = study_sizes[size]
        plt.annotate(f"{total}", 
                     (size, total + 5),
                     ha='center', va='bottom',
                     fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/study_size_distribution.png")
    plt.close()
    
    print(f"\nAnalysis results and plots saved to: {save_dir}")
    
    return {
        'study_sizes': dict(study_sizes),
        'accuracy_by_size': accuracy_by_size,
        'normal_accuracy_by_size': normal_accuracy_by_size,
        'abnormal_accuracy_by_size': abnormal_accuracy_by_size,
        'confidence_by_size': avg_confidence_by_size
    }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize attention weights in MURA classifier')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model weights')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--base_path', type=str, default='../', help='Base path for MURA dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--backbone', type=str, default='densenet169', help='Backbone architecture (resnet50 or densenet169)')
    parser.add_argument('--output_dir', type=str, default='attention_visualizations', help='Directory to save visualizations')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create transform for test data
    test_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    test_dataset = MURADataset(csv_file=args.test_csv, transform=test_transform, base_path=args.base_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        pin_memory=True
    )
    
    # Create model
    model = AttentionMURAClassifier(backbone=args.backbone, pretrained=False, threshold=args.threshold)
    
    # Load model weights
    print(f"Loading model weights from {args.model_path}")
    try:
        # Try loading the entire model first
        model = torch.load(args.model_path, map_location=device)
    except:
        # If that fails, try loading just the state dict
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    model = model.to(device)
    print("Model loaded successfully")
    
    # Create criterion for loss calculation (optional)
    # This is just to compute loss during evaluation, but not required for visualization
    try:
        # Compute abnormal and normal counts from the test dataset
        df_test = pd.read_csv(args.test_csv)
        num_abnormal = df_test['label'].sum()
        num_normal = len(df_test) - num_abnormal
        criterion = WeightedBCELoss(abnormal_count=num_abnormal, normal_count=num_normal).to(device)
    except:
        criterion = None
        print("Warning: Could not create criterion, will not calculate loss during evaluation")
    
    # Create directories for each analysis type
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)
    
    high_var_dir = os.path.join(base_dir, 'high_variance')
    misclass_dir = os.path.join(base_dir, 'misclassified')
    attention_dir = os.path.join(base_dir, 'attention_analysis')
    study_size_dir = os.path.join(base_dir, 'study_size_analysis')
    
    # Run standard evaluation
    print("\n=== Running standard evaluation ===")
    results = evaluate_model(model, test_loader, criterion, device=device, threshold=args.threshold)
    
    # Run attention pattern analysis
    print("\n=== Analyzing attention patterns ===")
    attention_results = analyze_attention_patterns(
        model, test_loader, device=device, threshold=args.threshold, save_dir=attention_dir
    )
    
    # Run study size analysis
    print("\n=== Analyzing accuracy by study size ===")
    study_size_results = analyze_accuracy_by_study_size(
        model, test_loader, device=device, threshold=args.threshold, save_dir=study_size_dir
    )
    
    # Generate high variance visualizations
    print("\n=== Generating visualizations for studies with high attention variance ===")
    visualize_high_attention_variance_studies(
        model, test_loader, num_samples=5, variance_threshold=0.03, 
        device=device, save_dir=high_var_dir
    )
    
    # Generate misclassified studies visualizations
    print("\n=== Generating visualizations for misclassified studies ===")
    visualize_misclassified_studies(
        model, test_loader, num_samples=5, 
        device=device, threshold=args.threshold, save_dir=misclass_dir
    )
    
    # Combine all results
    all_results = {**results, **attention_results, **study_size_results}
    
    # Save summary to file
    with open(os.path.join(base_dir, 'results_summary.txt'), 'w') as f:
        f.write("=== MURA Classification Results ===\n\n")
        f.write(f"Accuracy: {results['acc']:.4f}\n")
        f.write(f"AUC: {results['auc']:.4f}\n")
        f.write(f"F1 Score: {results['f1']:.4f}\n\n")
        
        f.write("=== Attention Analysis ===\n\n")
        f.write(f"Total studies analyzed: {attention_results['total_studies']}\n")
        f.write(f"Normal studies accuracy: {attention_results['normal_accuracy']:.4f}\n")
        f.write(f"Abnormal studies accuracy: {attention_results['abnormal_accuracy']:.4f}\n")
        f.write(f"Avg attention variance (correct): {attention_results['avg_variance_correct']:.4f}\n")
        f.write(f"Avg attention variance (wrong): {attention_results['avg_variance_wrong']:.4f}\n\n")
        
        f.write("=== Study Size Analysis ===\n\n")
        f.write("Study Size | Overall Accuracy | Normal Accuracy | Abnormal Accuracy\n")
        f.write("--------------------------------------------------------------\n")
        for size in sorted(study_size_results['accuracy_by_size'].keys()):
            f.write(f"{size:^10} | {study_size_results['accuracy_by_size'][size]:^16.4f} | ")
            f.write(f"{study_size_results['normal_accuracy_by_size'][size]:^15.4f} | ")
            f.write(f"{study_size_results['abnormal_accuracy_by_size'][size]:^16.4f}\n")
    
    print(f"\nAnalysis complete. Results saved to {base_dir}")
    
    return all_results


if __name__ == "__main__":
    main()