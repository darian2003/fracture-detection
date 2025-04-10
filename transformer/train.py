import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import math
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MarginWeightedBCELoss(nn.Module):
    def __init__(self, abnormal_count, normal_count, margin=0.2):
        super(MarginWeightedBCELoss, self).__init__()
        total = abnormal_count + normal_count
        self.w_pos = normal_count / total  # weight for label y = 1 (abnormal)
        self.w_neg = abnormal_count / total  # weight for label y = 0 (normal)
        self.margin = margin
        
    def forward(self, inputs, targets):
        # Ensure shape compatibility
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
            
        # Standard WBCE calculation
        probs = torch.sigmoid(inputs)
        wbce_loss = -(
            self.w_pos * targets * torch.log(probs + 1e-8) +
            self.w_neg * (1 - targets) * torch.log(1 - probs + 1e-8)
        )
        
        # Margin component to enforce separation
        margin_loss = torch.where(
            targets > 0.5,  # For positive/abnormal samples
            torch.relu(0.5 + self.margin - probs),  # Push abnormal samples above (0.5 + margin)
            torch.relu(probs - (0.5 - self.margin))  # Push normal samples below (0.5 - margin)
        )
        
        # Combine losses (you can adjust the weight of margin_loss as needed)
        total_loss = wbce_loss + 0.3 * margin_loss
        
        return total_loss.mean()


class TransformerMURAClassifier(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, threshold=0.5, 
                 embed_dim=512, num_heads=4, transformer_layers=2, dropout=0.1):
        super(TransformerMURAClassifier, self).__init__()

        self.threshold = threshold
        self.backbone_name = backbone
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Load backbone
        if backbone == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.num_features = base_model.fc.in_features
        elif backbone == 'densenet169':
            base_model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT if pretrained else None)
            self.feature_extractor = base_model.features
            self.num_features = base_model.classifier.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Feature embedding to project CNN features to transformer dimension
        self.feature_embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Special learnable [CLS] token that will be used for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=transformer_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # Image-specific classifier (for per-image loss)
        self.image_classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize transformer weights properly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        # Initialize the CLS token
        nn.init.normal_(self.cls_token, std=0.02)

    def extract_features(self, x):
        """Extract features from a batch of images"""
        features = self.feature_extractor(x)
        
        # Pooling for DenseNet
        if self.backbone_name == 'densenet169':
            features = F.adaptive_avg_pool2d(features, (1, 1))
        
        # Embed features into transformer dimension
        embedded = self.feature_embedding(features)
        return embedded

    def forward(self, x_list):
        """
        Forward pass for transformer-based study-level classifier.

        Args:
            x_list: List of tensors [num_images, C, H, W] for each study

        Returns:
            study_logits: Tensor [B, 1]
            image_logits: Tensor [B, max_images, 1]
            attention_weights: Tensor [B, max_images, 1]
            orig_masks: Tensor [B, max_images, 1] (1 for valid images, 0 for padding)
        """
        batch_size = len(x_list)
        max_images = max([x.size(0) for x in x_list])

        batch_tensor_list = []
        attn_mask_list = []
        orig_mask_list = []

        for study_images in x_list:
            num_images = study_images.size(0)

            # Padding for images
            if num_images < max_images:
                padding = torch.zeros(
                    max_images - num_images,
                    study_images.size(1),
                    study_images.size(2),
                    study_images.size(3),
                    device=study_images.device
                )
                padded_study = torch.cat([study_images, padding], dim=0)
            else:
                padded_study = study_images

            # Build attention mask: True = padding
            attn_mask = torch.ones(max_images + 1, dtype=torch.bool, device=study_images.device)
            attn_mask[0] = False  # CLS token
            attn_mask[1:num_images + 1] = False  # Valid images

            # Build original mask: 1 = valid image
            orig_mask = torch.zeros(max_images, 1, device=study_images.device)
            orig_mask[:num_images] = 1

            batch_tensor_list.append(padded_study.unsqueeze(0))  # [1, max_images, C, H, W]
            attn_mask_list.append(attn_mask.unsqueeze(0))        # [1, max_images + 1]
            orig_mask_list.append(orig_mask.unsqueeze(0))        # [1, max_images, 1]

        # Combine
        padded_batch = torch.cat(batch_tensor_list, dim=0)        # [B, max_images, C, H, W]
        attn_masks = torch.cat(attn_mask_list, dim=0)             # [B, max_images + 1]
        orig_masks = torch.cat(orig_mask_list, dim=0)             # [B, max_images, 1]

        # Flatten for CNN feature extraction
        B, N, C, H, W = padded_batch.shape
        flat_images = padded_batch.view(B * N, C, H, W)
        features = self.extract_features(flat_images)             # [B * N, D]
        features = features.view(B, N, -1)                        # [B, max_images, D]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)             # [B, 1, D]
        x = torch.cat([cls_tokens, features], dim=1)              # [B, max_images + 1, D]

        # Pass through transformer
        x = self.transformer_encoder(x, src_key_padding_mask=attn_masks)  # [B, max_images + 1, D]

        # Study-level prediction from CLS token
        cls_output = x[:, 0, :]                                   # [B, D]
        study_logits = self.classifier(cls_output)                # [B, 1]

        # Image-level predictions
        image_embeddings = x[:, 1:, :]                            # [B, max_images, D]
        image_logits = self.image_classifier(image_embeddings)    # [B, max_images, 1]

        # Attention weights (softmax similarity between CLS and each image embedding)
        attn_scores = torch.bmm(image_embeddings, cls_output.unsqueeze(-1))  # [B, max_images, 1]
        attn_weights = F.softmax(attn_scores, dim=1)              # [B, max_images, 1]

        # Zero-out padding weights
        attn_weights = attn_weights * orig_masks
        attn_sum = attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        attn_weights = attn_weights / attn_sum                    # Normalize non-padded

        # Mask image logits too (if needed)
        image_logits = image_logits * orig_masks

        return study_logits, image_logits, attn_weights, orig_masks

    def visualize_attention(self, x_list):
        """
        Get attention weights for visualization
        
        Returns:
            study_logits: Tensor of shape [batch_size, 1]
            normalized_weights: List of tensors, each containing attention weights for real images
        """
        study_logits, _, attention_weights, masks = self.forward(x_list)
        
        batch_size = len(x_list)
        normalized_weights = []
        
        for i in range(batch_size):
            # Get mask and attention for this study
            study_mask = masks[i].squeeze(-1)
            study_attention = attention_weights[i].squeeze(-1)
            
            # Get weights for real images only
            real_image_attention = study_attention[study_mask.bool()]
            
            # Normalize to sum to 1
            if real_image_attention.size(0) > 0:
                real_image_attention = real_image_attention / real_image_attention.sum()
            
            normalized_weights.append(real_image_attention)
            
        return study_logits, normalized_weights


class MultiHeadAttentionWithWeights(nn.Module):
    """Custom Multi-head Attention that also returns attention weights"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize weights similar to PyTorch's MultiheadAttention
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.qkv_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        
        # Project the queries, keys and values
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multiple heads
            expanded_mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(expanded_mask, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project the output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)
        
        # Return both the output and the attention weights
        return output, attn_weights


# Training function
def train_transformer_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
                         num_epochs=15, device='cuda', threshold=0.5):
    model = model.to(device)
    best_val_loss = float('inf')
    best_model_wts = None
    best_val_metrics = None
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_f1': []
    }
    
    # Learning rate warmup
    warmup_epochs = min(2, num_epochs // 5)
    initial_lr = optimizer.param_groups[0]['lr']

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 40)
        
        # Apply learning rate warmup
        if epoch < warmup_epochs and scheduler is None:
            # Linear warmup
            lr_scale = min(1.0, (epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = initial_lr * lr_scale
            print(f"Warmup learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        train_probs = []

        # Progress bar for training
        for batch_idx, batch in enumerate(train_loader):
            labels = batch['label'].to(device)
            images_list = [imgs.to(device) for imgs in batch['images']]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with transformer
            study_logits, image_logits, attention_weights, masks = model(images_list)
            
            # Study-level loss (primary objective)
            study_loss = criterion(study_logits, labels)
            
            # Optional: Per-image loss as auxiliary task
            # Expand labels to match image count
            labels_expanded = labels.view(-1, 1, 1).expand_as(image_logits)
            
            # Flatten everything and apply mask
            flat_logits = image_logits.reshape(-1, 1)
            flat_labels = labels_expanded.reshape(-1, 1)
            flat_mask = masks.reshape(-1, 1)
            
            # Filter valid (non-padded) image logits
            valid_indices = flat_mask.bool().squeeze()
            if valid_indices.any():
                valid_logits = flat_logits[valid_indices]
                valid_labels = flat_labels[valid_indices]
                
                # Compute per-image loss (with lower weight)
                image_loss = criterion(valid_logits, valid_labels)
                
                # Combined loss (study-level is primary, per-image is auxiliary)
                loss = study_loss + 0.3 * image_loss
            else:
                loss = study_loss

            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Track metrics
            train_loss += loss.item() * len(labels)
            batch_probs = torch.sigmoid(study_logits).detach().cpu().numpy()
            batch_preds = (batch_probs > threshold).astype(int)
            train_preds.extend(batch_preds)
            train_labels.extend(labels.cpu().numpy())
            train_probs.extend(batch_probs)
            
            # Print progress update
            if (batch_idx + 1) % 50 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            

        # Calculate training metrics
        train_loss /= len(train_loader.dataset)
        train_preds = np.array(train_preds).flatten()
        train_labels = np.array(train_labels).flatten()
        train_probs = np.array(train_probs).flatten()
        train_acc = accuracy_score(train_labels, train_preds)
        train_auc = roc_auc_score(train_labels, train_probs)
        train_f1 = f1_score(train_labels, train_preds)

        # Validation phase
        val_metrics = evaluate_transformer_model(model, val_loader, criterion, device, threshold)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['acc']
        val_auc = val_metrics['auc']
        val_f1 = val_metrics['f1']
        val_probs = val_metrics['probs']

        # # LR Scheduler step (if not None and after warmup)
        # if scheduler is not None and epoch >= warmup_epochs:
        #     if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #         scheduler.step(val_loss)  # For ReduceLROnPlateau
        #     else:
        #         scheduler.step()  # For other schedulers

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)

        # Log metrics
        try:
            import wandb
            wandb.log({
                "Train/Loss": train_loss,
                "Train/Accuracy": train_acc,
                "Train/AUC": train_auc,
                "Train/F1": train_f1,
                "Val/Loss": val_loss,
                "Val/Accuracy": val_acc,
                "Val/AUC": val_auc,
                "Val/F1": val_f1,
                "Train/Probabilities": wandb.Histogram(train_probs),
                "Val/Probabilities": wandb.Histogram(val_probs),
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        except ImportError:
            pass

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict().copy()
            best_val_metrics = val_metrics.copy()
            print(f"New best validation loss: {best_val_loss:.4f}")

        print()

    # Load best model weights
    model.load_state_dict(best_model_wts)
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

    # Plot learning curves if available
    try:
        from plot import plot_learning_curves
        plot_learning_curves(history)
    except ImportError:
        print("Warning: plot_learning_curves function not available.")

    return model, best_val_metrics, history


def evaluate_transformer_model(model, data_loader, criterion, device='cuda', threshold=0.5):
    """
    Evaluation function for transformer-based model
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

            # Forward pass with transformer
            study_logits, image_logits, attention_weights, masks = model(images_list)
            
            # Calculate loss
            loss = criterion(study_logits, labels)

            # Track metrics
            val_loss += loss.item() * len(labels)
            val_outputs.extend(study_logits.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
            # Store attention weights for analysis
            for i in range(len(images_list)):
                study_mask = masks[i].squeeze(-1)
                study_attention = attention_weights[i].squeeze(-1)
                real_attention = study_attention[study_mask.bool()]
                attention_weights_list.append(real_attention.cpu().numpy())

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
        'outputs': val_outputs,
        'attention_weights': attention_weights_list
    }

    return results


def visualize_transformer_attention(model, data_loader, num_samples=5, device='cuda'):
    """
    Visualize attention weights from transformer for a few samples
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    model.eval()
    samples_seen = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if samples_seen >= num_samples:
                break
                
            images_list = [imgs.to(device) for imgs in batch['images']]
            labels = batch['label'].to(device)
            
            # Get logits and attention weights
            study_logits, normalized_weights = model.visualize_attention(images_list)
            study_probs = torch.sigmoid(study_logits).cpu().numpy()
            
            # Visualize up to num_samples from this batch
            for i in range(min(len(images_list), num_samples - samples_seen)):
                # Get attention weights for this study
                weights = normalized_weights[i].cpu().numpy()
                num_images = len(weights)
                
                if num_images == 0:
                    continue  # Skip empty studies
                
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
                axes[-1].set_title(f"Label: {labels[i].item()}, Pred: {study_probs[i][0]:.3f}")
                
                plt.tight_layout()
                plt.savefig(f"transformer_attention_sample_{samples_seen}.png")
                plt.close()
                
                samples_seen += 1
                
    print(f"Saved {samples_seen} transformer attention visualization plots.")