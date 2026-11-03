import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PIL import Image

from src.dataset import BrainMRIDataset
from src.train import get_device

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class PureResidualBrainCNN(nn.Module):
    """
    Model 1: Custom 2D-CNN with Multi-Stage Residual Connections.
    Designed from Scratch (No ImageNet pretraining).
    """
    def __init__(self, num_classes=4):
        super(PureResidualBrainCNN, self).__init__()
        self.model_name = "Custom 2D-CNN (Pure Residual Architecture)"
        
        # Stem: 3 -> 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 4 Residual Stages:
        self.stage1 = ResidualBlock(32, 64, stride=2)    # 224 -> 112
        self.stage2 = ResidualBlock(64, 128, stride=2)   # 112 -> 56
        self.stage3 = ResidualBlock(128, 256, stride=2)  # 56 -> 28
        self.stage4 = ResidualBlock(256, 384, stride=2)  # 28 -> 14
        
        # 2x2 Spatial pooling retains quadrant layout (384 * 2 * 2 = 1536)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(384 * 2 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_augmented_dataloaders(root_dir="archive", batch_size=64):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_ds = BrainMRIDataset(root_dir, split="Training", transform=train_transform)
    test_ds = BrainMRIDataset(root_dir, split="Testing", transform=test_transform)

    # 85/15 train/val split
    val_size = int(0.15 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def main():
    import sys
    device = get_device()
    print("=" * 65)
    print("TRAINING PURE RESIDUAL CUSTOM 2D-CNN (NO SE-ATTENTION)")
    print("=" * 65)
    
    train_loader, val_loader, test_loader = get_augmented_dataloaders(batch_size=64)
    model = PureResidualBrainCNN(num_classes=4).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Architecture: {model.model_name}")
    print(f"Trainable Parameters: {params:,}")
    print(f"Training Scans: {len(train_loader.dataset)} | Val Scans: {len(val_loader.dataset)} | Test Scans: {len(test_loader.dataset)}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.06)
    optimizer = optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1e-4)
    epochs = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    start_epoch = 1
    save_path = "models/custom_cnn_pure_residual.pth"
    os.makedirs("models", exist_ok=True)

    # Resume capability if requested or checkpoint exists
    resume_flag = "--resume" in sys.argv or os.path.exists(save_path)
    if resume_flag and os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_acc = ckpt.get('val_acc', 0.0)
        # Fast-forward scheduler to the current epoch
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"🔄 RESUMING EXACTLY FROM EPOCH {start_epoch}/15 (Loaded Best Val Acc: {best_val_acc:.2f}%)", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
                
        tr_acc = train_correct / train_total * 100
        vl_acc = val_correct / val_total * 100
        sec = round(time.time() - t0, 1)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({sec}s) - Train Acc: {tr_acc:.2f}% | Val Acc: {vl_acc:.2f}%", flush=True)
        
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': vl_acc
            }, save_path)
            
    print("\nBest Validation Accuracy:", f"{best_val_acc:.2f}%", flush=True)
    
    # Evaluate on Test Set (1,600 unseen scans)
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    test_acc = accuracy_score(all_labels, all_preds) * 100
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    print("=" * 65, flush=True)
    print(f"FINAL HELD-OUT TEST ACCURACY (1,600 Scans): {test_acc:.2f}%", flush=True)
    print(f"Macro Precision: {p*100:.2f}% | Macro Recall: {r*100:.2f}% | Macro F1: {f1*100:.2f}%", flush=True)
    print("=" * 65, flush=True)
    
    with open("models/custom_cnn_pure_residual_results.json", "w") as f:
        json.dump({
            "model_name": model.model_name,
            "test_accuracy": round(test_acc, 2),
            "macro_precision": round(p * 100, 2),
            "macro_recall": round(r * 100, 2),
            "macro_f1": round(f1 * 100, 2)
        }, f, indent=2)

if __name__ == "__main__":
    main()

