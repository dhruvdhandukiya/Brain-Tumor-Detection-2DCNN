"""
Train and evaluate an Enhanced Deep CNN (Pretrained ResNet50)
with Label Smoothing, Cosine Annealing, and Advanced Augmentation
to boost Brain Tumor Classification accuracy past 96%!
"""

import os
import ssl
import time
import json
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

ssl._create_default_https_context = ssl._create_unverified_context

from src.dataset import get_dataloaders
from src.train import get_device

def build_enhanced_resnet50(num_classes=4):
    weights = models.ResNet50_Weights.DEFAULT
    backbone = models.resnet50(weights=weights)
    
    # Freeze initial low-level layers (stem & layer1) to retain ImageNet edge filters
    for name, param in backbone.named_parameters():
        if "layer3" not in name and "layer4" not in name and "fc" not in name:
            param.requires_grad = False
            
    num_ftrs = backbone.fc.in_features  # 2048
    backbone.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.35),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.25),
        nn.Linear(128, num_classes)
    )
    return backbone

def main():
    device = get_device()
    print("=" * 80)
    print("  TRAINING ENHANCED RESNET50 (HIGH-ACCURACY RADIOLOGY SUITE)")
    print("=" * 80)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir="./archive",
        batch_size=64,
        num_workers=2
    )

    model = build_enhanced_resnet50(num_classes=len(class_names)).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Label smoothing cross entropy prevents over-confident misclassifications
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

    epochs = 4
    best_val_acc = 0.0
    save_path = "models/resnet50_enhanced_best.pth"

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch [{epoch}/{epochs}] | Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ⭐ Saved Best Checkpoint -> Val Acc: {val_acc*100:.2f}%")

    # Final Evaluation on the 1,600 Test Set
    print("\n--- Evaluating Enhanced ResNet50 on 1,600 Held-Out Test Scans ---")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            # Apply dual-view Test-Time Augmentation (TTA)
            p1 = torch.softmax(model(images), dim=1)
            p2 = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
            p_avg = (p1 + p2) / 2.0
            all_preds.extend(p_avg.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    print(f"\n🏆 ENHANCED RESNET50 TEST ACCURACY (with TTA): {test_acc*100:.2f}%")
    print(f"🏆 Macro Precision: {prec*100:.2f}% | Macro Recall: {rec*100:.2f}% | Macro F1: {f1*100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    main()
