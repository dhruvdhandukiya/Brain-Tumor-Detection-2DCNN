import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def get_device():
    """
    Selects the fastest available device: Apple Silicon (MPS), CUDA (NVIDIA GPU), or CPU.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: NVIDIA CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    return device


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.2f}%"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, num_epochs=15, lr=0.0003, weight_decay=1e-4, 
                save_path="models/best_model.pth", device=None):
    """
    Complete training procedure with validation, checkpointing, and scheduler.
    """
    if device is None:
        device = get_device()
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "time_per_epoch": []
    }
    
    best_val_acc = 0.0
    print(f"\n🚀 Starting training for {model.model_name} ({num_epochs} epochs)...")
    start_total_time = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - t0

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["time_per_epoch"].append(epoch_time)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'model_name': model.model_name
            }, save_path)
            best_marker = "🌟 (Best Model Saved!)"
        else:
            best_marker = ""

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:6.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}% | "
              f"Time: {epoch_time:.1f}s {best_marker}")

    total_training_time = time.time() - start_total_time
    print(f"\n✅ Training completed in {total_training_time/60:.2f} mins. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    
    # Save training history
    history_path = save_path.replace(".pth", "_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
        
    return history, best_val_acc
