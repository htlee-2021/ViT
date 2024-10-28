# main.py - Your training script
from vit_model import VideoViT, train_vit
from data_loader import create_frame_dataloaders

# Create dataloaders
train_loader, val_loader = create_frame_dataloaders(
    frames_root_dir='path/to/dfdc_train_part_0_1_frame',
    metadata_path='path/to/metadata.json',
    batch_size=8,
    num_frames=10,
    image_size=224
)

# Initialize model
model = VideoViT(
    num_frames=10,  # Match your frame count
    image_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=2
)

# Train model
model = train_vit(train_loader, val_loader, num_classes=2)