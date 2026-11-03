import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.dataset import get_dataloaders
from src.train import get_device


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape
        w = F.adaptive_avg_pool2d(x, 1).view(b, c)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w)).view(b, c, 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)

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
        out = self.se(out)
        out = self.relu(out + residual)
        return out


class EnhancedBrainCNN(nn.Module):
    """
    Enhanced Custom 2D-CNN Architecture Designed from Scratch (No ImageNet Pretraining).
    Combines 4-stage residual skip connections with Squeeze-and-Excitation channel attention.
    """
    def __init__(self, num_classes=4):
        super(EnhancedBrainCNN, self).__init__()
        self.model_name = "Enhanced Custom 2D-CNN (Residual + Attention)"
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage1 = ResidualBlock(32, 64, stride=2)    # 224 -> 112
        self.stage2 = ResidualBlock(64, 128, stride=2)   # 112 -> 56
        self.stage3 = ResidualBlock(128, 256, stride=2)  # 56 -> 28
        self.stage4 = ResidualBlock(256, 384, stride=2)  # 28 -> 14

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
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


def train_enhanced_model(epochs=20, lr=1.5e-3, batch_size=64):
    device = get_device()
    print("=" * 70)
    print("   TRAINING ENHANCED CUSTOM 2D-CNN (FROM SCRATCH, NO PRETRAINING)")
    print("=" * 70)

    # Note: num_workers=0 is optimal on macOS APFS
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        "archive", batch_size=batch_size, num_workers=0
    )

    model = EnhancedBrainCNN(num_classes=4).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Architecture: {model.model_name}")
    print(f"Total Trainable Parameters: {params:,}")
    print(f"Training on: {len(train_loader.dataset)} scans | Validation: {len(val_loader.dataset)} scans")
    print(f"Test Set: {len(test_loader.dataset)} scans")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    save_path = "models/custom_cnn_enhanced_best.pth"
    os.makedirs("models", exist_ok=True)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": []
    }

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Training Phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_train_loss = train_loss / total
        epoch_train_acc = correct / total

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["lr"].append(curr_lr)

        dt = time.time() - t0
        best_tag = ""
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "model_name": model.model_name
            }, save_path)
            best_tag = "🌟 (New Best!)"

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:5.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:5.2f}% | "
              f"LR: {curr_lr:.2e} | {dt:.1f}s {best_tag}")

    print(f"\nTraining completed in {(time.time()-start_time)/60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")

    # Evaluation on 1,600 Held-Out Scans
    print("\n--- Final Evaluation on 1,600 Held-Out Test Scans ---")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            # Test-time augmentation (original + horizontal flip)
            p1 = torch.softmax(model(images), dim=1)
            p2 = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
            probs = (p1 + p2) / 2.0
            preds = probs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    test_acc = accuracy_score(y_true, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    test_results = {
        "model_name": model.model_name,
        "parameters": params,
        "test_accuracy": test_acc,
        "macro_precision": precision * 100,
        "macro_recall": recall * 100,
        "macro_f1": f1 * 100,
        "best_val_acc": best_val_acc * 100
    }

    with open("models/custom_cnn_enhanced_results.json", "w") as f:
        json.dump(test_results, f, indent=4)

    with open("models/custom_cnn_enhanced_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("\n" + "=" * 60)
    print("      FINAL ENHANCED CUSTOM 2D-CNN TEST RESULTS")
    print("=" * 60)
    print(f"Test Accuracy:    {test_acc:.2f}% (Baseline Custom CNN was 76.94%)")
    print(f"Macro Precision:  {precision*100:.2f}%")
    print(f"Macro Recall:     {recall*100:.2f}%")
    print(f"Macro F1-Score:   {f1*100:.2f}%")
    print("=" * 60)
    return test_results


if __name__ == "__main__":
    train_enhanced_model(epochs=20, lr=1.5e-3, batch_size=64)
