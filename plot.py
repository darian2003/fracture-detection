import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import os
from PIL import Image
import torch

# 7. Visualization Functions
def plot_learning_curves(history, output_dir="results", filename_prefix="learning_curves"):
    """Plot training and validation metrics and save the figures.
    
    Args:
        history: Dictionary containing training and validation metrics.
        output_dir: Directory where plots will be saved.
        filename_prefix: Prefix for the saved plot filenames.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    # AUC
    axes[1, 0].plot(history['train_auc'], label='Train')
    axes[1, 0].plot(history['val_auc'], label='Validation')
    axes[1, 0].set_title('AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()

    # F1 Score
    axes[1, 1].plot(history['train_f1'], label='Train')
    axes[1, 1].plot(history['val_f1'], label='Validation')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{filename_prefix}.png")
    plt.savefig(save_path)
    plt.close(fig)  # Prevent excessive memory use

def visualize_results(results, output_dir="results", filename_prefix="results"):
    """Visualize model results with confusion matrix, ROC curve, etc.
    
    Args:
        results: Dictionary containing labels, predictions, and probabilities.
        output_dir: Directory where plots will be saved.
        filename_prefix: Prefix for the saved plot filenames.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    y_true = np.array(results['labels'])
    y_pred = np.array(results['preds'])
    y_probs = np.array(results['probs'])

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    save_path = os.path.join(output_dir, f"{filename_prefix}_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    save_path = os.path.join(output_dir, f"{filename_prefix}_roc_curve.png")
    plt.savefig(save_path)
    plt.close(fig)

    # Prediction Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(y_probs[y_true == 0], color='green', label='Normal', alpha=0.5, bins=20, kde=True, ax=ax)
    sns.histplot(y_probs[y_true == 1], color='red', label='Abnormal', alpha=0.5, bins=20, kde=True, ax=ax)
    ax.axvline(0.5, color='black', linestyle='--', label='Threshold')
    ax.set_title('Prediction Distribution')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.legend()
    save_path = os.path.join(output_dir, f"{filename_prefix}_prediction_distribution.png")
    plt.savefig(save_path)
    plt.close(fig)

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal'])
    report_path = os.path.join(output_dir, f"{filename_prefix}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Saved all results and plots to {output_dir}.")

def find_optimal_threshold(results, output_dir="results", filename_prefix="threshold_analysis"):
    """Find the optimal threshold for classification and save the plot.
    
    Args:
        results: Dictionary containing labels and probabilities.
        output_dir: Directory where plots will be saved.
        filename_prefix: Prefix for the saved plot filenames.
        
    Returns:
        float: The optimal threshold value.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    y_true = np.array(results['labels'])
    y_probs = np.array(results['probs'])

    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_probs > threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

    # Find best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    # Plot thresholds vs F1
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1_scores, marker='o')
    ax.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold: {best_threshold:.2f} (F1={best_f1:.3f})')
    ax.set_title('Threshold vs F1 Score')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score')
    ax.legend()
    ax.grid(True)
    save_path = os.path.join(output_dir, f"{filename_prefix}_threshold_vs_f1.png")
    plt.savefig(save_path)
    plt.close(fig)

    print(f"Optimal threshold: {best_threshold:.3f} with F1 score: {best_f1:.3f}")

    # Save classification report with new threshold
    optimal_preds = (y_probs > best_threshold).astype(int)
    report = classification_report(y_true, optimal_preds, target_names=['Normal', 'Abnormal'])
    report_path = os.path.join(output_dir, f"{filename_prefix}_optimal_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    return best_threshold


def show_augmented_image(dataset, idx=0, output_dir=None, filename_prefix="augmented_image"):
    """
    Displays the original and augmented version of the image at index `idx` from a MURADataset.
    
    Args:
        dataset: The dataset containing the images.
        idx: Index of the image to display.
        output_dir: Directory where the plot will be saved. If None, only displays the plot.
        filename_prefix: Prefix for the saved plot filename.
    """
    # Get study path
    study_path = dataset.data.iloc[idx]['path']
    image_file = sorted([
        f for f in os.listdir(study_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')
    ])[0]  # pick first valid image

    image_path = os.path.join(study_path, image_file)
    image = Image.open(image_path).convert('RGB')

    # Apply transform
    augmented = dataset.transform(image)

    # Undo normalization to visualize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    unnormalized = augmented * std + mean
    unnormalized = torch.clamp(unnormalized, 0, 1)

    # Plot original and augmented image
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(unnormalized.permute(1, 2, 0).numpy())
    axes[1].set_title("Augmented Image")
    axes[1].axis("off")

    plt.tight_layout()
    
    # Save the figure if output_dir is provided
    if output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{filename_prefix}.png")
        plt.savefig(save_path)
        print(f"Saved augmented image comparison to {save_path}")
    
    plt.show()