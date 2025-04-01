import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MURADataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        study_path = row['path']
        
        # Extract label, body part, and class
        label = torch.tensor(row['label'], dtype=torch.float32)
        body_part = row['body_part']
        body_part_class = torch.tensor(row['class'], dtype=torch.long)

        # Get all images from the study directory
        images = []
        valid_extensions = ['.png', '.jpg', '.jpeg']

        try:
            for file_name in os.listdir(study_path):
                file_path = os.path.join(study_path, file_name)
                if file_name.startswith('.'):
                    continue
                # Check if it's a valid image file
                if os.path.isfile(file_path) and any(file_name.lower().endswith(ext) for ext in valid_extensions):
                    try:
                        image = Image.open(file_path).convert('RGB')
                        if self.transform:
                            image = self.transform(image)
                        images.append(image)
                    except Exception as e:
                        print(f"Error loading image {file_path}: {e}")
            
        except FileNotFoundError:
            print(f"Directory not found: {study_path}")
        except Exception as e:
            print(f"Error accessing directory {study_path}: {e}")

        # Stack all images into a single tensor
        if len(images) > 0:
            images = torch.stack(images)
        else:
            # Handle empty directories by creating a dummy tensor
            print(f"Warning: No valid images found in {study_path}")
            # Creating a dummy tensor with standard image size
            images = torch.zeros((1, 3, 224, 224))

        return {
            'path': study_path,
            'images': images,
            'label': label,
            'body_part': body_part,
            'class': body_part_class,
            'num_images': len(images)
        }

def custom_collate(batch):
    paths = [item['path'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    num_images = [item['num_images'] for item in batch]
    images = [item['images'] for item in batch]
    
    return {
        'path': paths,
        'images': images,
        'label': labels,
        'num_images': num_images
    }
