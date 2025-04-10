import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from plot import plot_learning_curves, visualize_results, find_optimal_threshold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Model Class with Multiple Aggregation Strategies
class MURAClassifier(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, agg_strategy='prob_mean', threshold=0.5):
        super(MURAClassifier, self).__init__()

        self.threshold = threshold
        self.backbone_name = backbone

        # Load backbone
        if backbone == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.num_features = base_model.fc.in_features
        if backbone == 'densenet169':
            base_model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT if pretrained else None)
            self.feature_extractor = base_model.features  # Use DenseNet feature extractor
            self.num_features = base_model.classifier.in_features  # Get the number of features before FC layer
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Aggregation strategy
        self.agg_strategy = agg_strategy

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_features, 1), 
        )

    def forward_batch(self, x):
        """Forward pass for a batch of images"""
        # x shape: [batch_size, channels, height, width]
        features = self.feature_extractor(x)
        
        # Pooling for DenseNet
        if self.backbone_name == 'densenet169':
            features = F.adaptive_avg_pool2d(features, (1, 1))
            
        # Apply classifier
        logits = self.classifier(features)
        return logits

    def forward(self, x_list):
        """
        Forward pass with optimized GPU usage using padding and masking
        
        Args:
            x_list: List of tensors, each of shape [num_images, channels, height, width]
        """
        batch_size = len(x_list)
        all_study_outputs = []

        # Find max number of images in any study in this batch
        max_images = max([study.size(0) for study in x_list])
        
        # Create padded batch tensor and mask
        batch_tensor_list = []
        mask_list = []
        
        for study_images in x_list:
            num_images = study_images.size(0)
            
            # Create mask (1 for real images, 0 for padding)
            mask = torch.zeros(max_images, 1, device=device)
            mask[:num_images] = 1
            
            # Handle padding if needed
            if num_images < max_images:
                # Create padding
                padding = torch.zeros(
                    max_images - num_images, 
                    study_images.size(1), 
                    study_images.size(2), 
                    study_images.size(3), 
                    device=device
                )
                padded_study = torch.cat([study_images, padding], dim=0)
            else:
                padded_study = study_images
                
            batch_tensor_list.append(padded_study.unsqueeze(0))  # Add batch dimension
            mask_list.append(mask.unsqueeze(0))  # Add batch dimension
            
        # Concatenate all studies into a single batch
        # Shape: [batch_size, max_images, channels, height, width]
        padded_batch = torch.cat(batch_tensor_list, dim=0)
        # Shape: [batch_size, max_images, 1]
        masks = torch.cat(mask_list, dim=0)
        
        # Reshape for efficient processing
        # Shape: [batch_size * max_images, channels, height, width]
        batch_images = padded_batch.view(-1, padded_batch.size(2), padded_batch.size(3), padded_batch.size(4))
        
        # Process all images at once (major GPU optimization)
        batch_logits = self.forward_batch(batch_images)
        
        # Reshape back to [batch_size, max_images, 1]
        batch_logits = batch_logits.view(batch_size, max_images, -1)
        
        # Apply mask to zero out padding predictions
        batch_logits = batch_logits * masks
        
        # Now apply aggregation strategy on the masked predictions
        for i in range(batch_size):
            study_logits = batch_logits[i]
            study_mask = masks[i]
            
            # Count number of real images (non-padded)
            num_real_images = study_mask.sum().item()
            
            # Apply aggregation strategy on logits (before sigmoid)
            if self.agg_strategy == 'prob_mean':
                # Convert to probabilities
                image_probs = torch.sigmoid(study_logits)
                
                # Compute sum and divide by number of real images
                # (exclude padding from the calculation)
                study_prob = image_probs.sum() / num_real_images
                
                # Convert back to logit for BCEWithLogitsLoss
                study_prob = torch.clamp(study_prob, min=1e-6, max=1-1e-6)  # Avoid log(0)
                study_output = torch.log(study_prob / (1 - study_prob))
                study_output = study_output.unsqueeze(0)  # Add dimension back

            elif self.agg_strategy == 'prob_max':
                # For max aggregation, the mask already zeroed out padding
                # so we can safely take max over all values
                image_probs = torch.sigmoid(study_logits)
                study_prob = torch.max(image_probs)
                
                # Convert back to logit
                study_output = torch.log(study_prob / (1 - study_prob + 1e-7))
                study_output = study_output.unsqueeze(0)
                
            else:
                raise ValueError(f"Unsupported aggregation strategy: {self.agg_strategy}")
                
            all_study_outputs.append(study_output)
            
        # Stack all study outputs
        return torch.cat(all_study_outputs, dim=0).unsqueeze(1) 

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Ensure inputs has shape [batch_size, 1]
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(1)
            
        # Ensure targets has shape [batch_size, 1]
        targets_view = targets.view(-1, 1)
        
        # Calculate BCE loss
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets_view)
        
        # Calculate focusing factor
        pt = torch.exp(-bce_loss)
        
        # Apply alpha weighting with properly shaped targets
        alpha_factor = self.alpha * targets_view + (1 - self.alpha) * (1 - targets_view)
        
        # Calculate focal loss
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()

class WeightedBCELoss(nn.Module):
    def __init__(self, abnormal_count, normal_count):
        super(WeightedBCELoss, self).__init__()
        total = abnormal_count + normal_count
        self.w_pos = normal_count / total  # weight for label y = 1 (abnormal)
        self.w_neg = abnormal_count / total  # weight for label y = 0 (normal)

    def forward(self, inputs, targets):
        # Ensure shape compatibility
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        probs = torch.sigmoid(inputs)
        loss = -(
            self.w_pos * targets * torch.log(probs + 1e-8) +
            self.w_neg * (1 - targets) * torch.log(1 - probs + 1e-8)
        )
        return loss.mean()


# 5. Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
               num_epochs=15, device='cuda', threshold=0.5):

    model = model.to(device)
    best_val_loss = float('inf')
    best_model_wts = None
    best_val_metrics = None
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_f1': []
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 40)

        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        train_probs = []

        # Progress bar for training
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            labels = batch['label'].to(device)
            images_list = [imgs.to(device) for imgs in batch['images']]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images_list)
            loss = criterion(outputs, labels.view(-1, 1))

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item() * len(labels)
            batch_probs = torch.sigmoid(outputs).detach().cpu().numpy()
            batch_preds = (batch_probs > threshold).astype(int)
            train_preds.extend(batch_preds)
            train_labels.extend(labels.cpu().numpy())
            train_probs.extend(batch_probs)

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Calculate training metrics
        train_loss /= len(train_loader.dataset)
        train_preds = np.array(train_preds).flatten()
        train_labels = np.array(train_labels).flatten()
        train_probs = np.array(train_probs).flatten()
        train_acc = accuracy_score(train_labels, train_preds)
        train_auc = roc_auc_score(train_labels, train_probs)
        train_f1 = f1_score(train_labels, train_preds)


        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['acc']
        val_auc = val_metrics['auc']
        val_f1 = val_metrics['f1']

        # LR Scheduler step (if not None)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)  # For ReduceLROnPlateau
            else:
                scheduler.step()  # For other schedulers

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)

        wandb.log({
            "Train/Loss": train_loss,
            "Train/Accuracy": train_acc,
            "Train/AUC": train_auc,
            "Train/F1": train_f1,
            "Val/Loss": val_loss,
            "Val/Accuracy": val_acc,
            "Val/AUC": val_auc,
            "Val/F1": val_f1,
            "epoch": epoch
        })

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}")

        # Model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict().copy()
            best_val_metrics = val_metrics.copy()

        print()

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Plot learning curves
    plot_learning_curves(history)

    return model, best_val_metrics, history

# 6. Evaluation Function
def evaluate_model(model, data_loader, criterion, device='cuda', threshold=0.5):
    """
    Evaluate the model
    """
    model.eval()
    val_loss = 0.0
    val_outputs = []
    val_labels = []

    with torch.no_grad():
        for batch in data_loader:
            images_list = [imgs.to(device) for imgs in batch['images']]
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(images_list)
            loss = criterion(outputs, labels.view(-1, 1))

            # Track metrics
            val_loss += loss.item() * len(labels)
            val_outputs.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    val_loss /= len(data_loader.dataset)
    val_outputs = np.array(val_outputs).flatten()
    val_labels = np.array(val_labels).flatten()

    # Convert logits to probabilities
    val_probs = torch.sigmoid(torch.tensor(val_outputs)).numpy()
    val_preds = (val_probs > threshold).astype(int)

    # Calculate metrics
    val_acc = accuracy_score(val_labels, val_preds)
    val_auc = roc_auc_score(val_labels, val_probs)
    val_f1 = f1_score(val_labels, val_preds)

    results = {
        'loss': val_loss,
        'acc': val_acc,
        'auc': val_auc,
        'f1': val_f1,
        'probs': val_probs,
        'preds': val_preds,
        'labels': val_labels,
        'outputs': val_outputs  # Raw outputs (logits)
    }

    return results
