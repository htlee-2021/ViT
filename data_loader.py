import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
import json
import random

class DFDCFrameDataset(Dataset):
    def __init__(self, frames_root_dir, metadata_path, num_frames=10, 
                 image_size=224, transform=None, mode='train'):
        """
        Args:
            frames_root_dir (str): Directory containing frame folders
            metadata_path (str): Path to metadata JSON file
            num_frames (int): Number of frames per video (10 in your case)
            image_size (int): Size to resize frames to
            transform (callable, optional): Optional transform to be applied on frames
            mode (str): 'train' or 'val'
        """
        self.frames_root_dir = frames_root_dir
        self.num_frames = num_frames
        self.mode = mode
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Get list of frame folders and labels
        self.frame_folders = []
        self.labels = []
        
        # Walk through the frames directory
        for folder in os.listdir(frames_root_dir):
            # Get the corresponding video filename from the folder name
            # Adjust this based on your folder naming convention
            video_filename = folder.replace('_frame', '.mp4')
            
            if video_filename in self.metadata:
                self.frame_folders.append(folder)
                self.labels.append(1 if self.metadata[video_filename]['label'] == 'FAKE' else 0)
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.frame_folders)
    
    def load_frames(self, frame_folder):
        """Load all frames from a folder."""
        frame_paths = sorted([
            os.path.join(self.frames_root_dir, frame_folder, f)
            for f in os.listdir(os.path.join(self.frames_root_dir, frame_folder))
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        # Verify we have the expected number of frames
        assert len(frame_paths) == self.num_frames, \
            f"Expected {self.num_frames} frames, found {len(frame_paths)} in {frame_folder}"
        
        # Load and process frames
        frames = []
        for frame_path in frame_paths:
            try:
                frame = Image.open(frame_path).convert('RGB')
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            except Exception as e:
                print(f"Error loading frame {frame_path}: {str(e)}")
                # Return a blank frame in case of error
                frames.append(torch.zeros((3, 224, 224)))
        
        return torch.stack(frames)

    def __getitem__(self, idx):
        frame_folder = self.frame_folders[idx]
        label = self.labels[idx]
        
        # Load frames
        frames = self.load_frames(frame_folder)
        
        return frames, label

def create_frame_dataloaders(frames_root_dir, metadata_path, batch_size=8, 
                           num_frames=10, image_size=224, num_workers=4, 
                           train_split=0.8):
    """
    Create train and validation dataloaders for frame data.
    """
    # Create dataset with training augmentations
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, 
                                saturation=0.1, hue=0.1)
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset with validation transforms
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create full dataset
    full_dataset = DFDCFrameDataset(
        frames_root_dir=frames_root_dir,
        metadata_path=metadata_path,
        num_frames=num_frames,
        image_size=image_size,
        transform=train_transform,
        mode='train'
    )
    
    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Override transforms for validation dataset
    val_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# Example usage:
"""
# Create dataloaders for your frame dataset
train_loader, val_loader = create_frame_dataloaders(
    frames_root_dir='path/to/dfdc_train_part_0_1_frame',
    metadata_path='path/to/metadata.json',
    batch_size=8,
    num_frames=10,  # Your extracted frame count
    image_size=224  # ViT input size
)

# Use the VideoViT model from the previous artifact
model = VideoViT(
    num_frames=10,  # Match your frame count
    image_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=2  # Binary classification: real/fake
)

# Train the model
model = train_vit(train_loader, val_loader, num_classes=2)
"""