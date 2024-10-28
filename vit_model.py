import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from einops import rearrange
from tqdm import tqdm

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, 
                                  kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.projection(x)  # (B, E, H', W')
        x = rearrange(x, 'b e h w -> b (h w) e')  # (B, N, E)
        return x

class VideoViT(nn.Module):
    def __init__(self, num_frames=10, image_size=224, patch_size=16, in_channels=3, 
                 num_classes=2, embed_dim=768, depth=12, num_heads=12, 
                 mlp_ratio=4., dropout=0.1):
        super().__init__()
        
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.total_patches = self.num_patches_per_frame * num_frames
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, 
                                        in_channels, embed_dim)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embeddings - now accounting for temporal dimension
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.total_patches + 1, embed_dim)
        )
        
        # Temporal embedding
        self.temporal_embed = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim)
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Initialize weights
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        B = x.shape[0]
        
        # Reshape for patch embedding
        # Combine batch and time dimensions
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        
        # Patch embedding
        x = self.patch_embed(x)  # Shape: (B*T, num_patches_per_frame, embed_dim)
        
        # Reshape to separate batch and temporal dimensions
        x = rearrange(x, '(b t) n e -> b (t n) e', b=B, t=self.num_frames)
        
        # Add positional embedding
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        
        # Add temporal information
        # Reshape to expose temporal dimension
        x_nocls = x[:, 1:, :]
        x_nocls = rearrange(x_nocls, 'b (t n) e -> b t n e', t=self.num_frames)
        x_nocls = x_nocls + self.temporal_embed.unsqueeze(2)
        x_nocls = rearrange(x_nocls, 'b t n e -> b (t n) e')
        
        # Recombine with CLS token
        x = torch.cat((x[:, :1, :], x_nocls), dim=1)
        
        # Transformer encoder
        x = self.transformer(x)
        
        # MLP head (use [CLS] token)
        x = x[:, 0]
        x = self.mlp_head(x)
        
        return x

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(val_loader), 100. * correct / total

def train_vit(train_loader, val_loader, num_classes, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = ViT(
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        dropout=0.1
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                          optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_vit_model.pth')
    
    return model

# Example usage:
"""
# Define your transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Create your dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Train the model
model = train_vit(train_loader, val_loader, num_classes=your_num_classes)
"""