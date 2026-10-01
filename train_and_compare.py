import os
import json
import time
import torch

from src.dataset import get_dataloaders
from src.models import CustomBrainCNN, PretrainedResNet18
from src.train import train_model, get_device
from src.evaluate import (
    evaluate_model_on_test,
    plot_training_curves,
    plot_confusion_matrices,
    plot_metric_bars
)

def run_experiment(data_dir=None, epochs=10, batch_size=32, lr_custom=0.0005, lr_resnet=0.0003):
    if data_dir is None:
        data_dir = "./archive" if os.path.exists("./archive") else "./data/brain_tumor_mri"
    print("=" * 70)
    print("  BRAIN TUMOR DETECTION: 2D-CNN & TRANSFER LEARNING BENCHMARK")
    print("=" * 70)

    device = get_device()
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print(f"\n📂 Loading dataset from: {data_dir}")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=2,
        mode="4class"
    )
    print(f"✅ Classes ({len(class_names)}): {class_names}")
    print(f"📊 Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)} | Test Batches: {len(test_loader)}")

    # -------------------------------------------------------------
    # 1. TRAIN MODEL 1: Custom Designed 2D-CNN
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print(" [1/2] Training Model 1: Custom Designed 2D-CNN")
    print("="*50)
    custom_model = CustomBrainCNN(num_classes=len(class_names))
    custom_save_path = "models/custom_cnn_best.pth"
    
    t0 = time.time()
    history_custom, best_acc_custom = train_model(
        model=custom_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        lr=lr_custom,
        weight_decay=1e-4,
        save_path=custom_save_path,
        device=device
    )
    custom_train_time = time.time() - t0

    # -------------------------------------------------------------
    # 2. TRAIN MODEL 2: Pretrained ResNet18 Transfer Learning
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print(" [2/2] Training Model 2: Pretrained ResNet18 (Transfer Learning)")
    print("="*50)
    resnet_model = PretrainedResNet18(num_classes=len(class_names), pretrained=True)
    resnet_save_path = "models/resnet18_best.pth"

    t0 = time.time()
    history_resnet, best_acc_resnet = train_model(
        model=resnet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        lr=lr_resnet,
        weight_decay=1e-4,
        save_path=resnet_save_path,
        device=device
    )
    resnet_train_time = time.time() - t0

    # -------------------------------------------------------------
    # 3. EVALUATION ON TEST DATASET
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print(" Evaluating Models on Independent Test Set")
    print("="*50)

    # Load best checkpoints
    custom_ckpt = torch.load(custom_save_path, map_location=device)
    custom_model.load_state_dict(custom_ckpt['model_state_dict'])
    
    resnet_ckpt = torch.load(resnet_save_path, map_location=device)
    resnet_model.load_state_dict(resnet_ckpt['model_state_dict'])

    metrics_custom, cm_custom, _, _ = evaluate_model_on_test(custom_model, test_loader, device, class_names)
    metrics_resnet, cm_resnet, _, _ = evaluate_model_on_test(resnet_model, test_loader, device, class_names)

    # -------------------------------------------------------------
    # 4. GENERATE COMPARATIVE PLOTS & JSON REPORT
    # -------------------------------------------------------------
    print("\n📊 Generating comparative plots and charts in './results/'...")
    plot_training_curves(history_custom, history_resnet, save_dir="results")
    plot_confusion_matrices(cm_custom, cm_resnet, class_names, save_dir="results")
    plot_metric_bars(metrics_custom, metrics_resnet, save_dir="results")

    summary_comparison = {
        "custom_2d_cnn": {
            **metrics_custom,
            "training_time_seconds": round(custom_train_time, 2),
            "best_val_accuracy": float(best_acc_custom)
        },
        "pretrained_resnet18": {
            **metrics_resnet,
            "training_time_seconds": round(resnet_train_time, 2),
            "best_val_accuracy": float(best_acc_resnet)
        },
        "class_names": class_names,
        "device_used": str(device)
    }

    with open("results/metrics_comparison.json", "w") as f:
        json.dump(summary_comparison, f, indent=4)

    # -------------------------------------------------------------
    # 5. PRINT BENCHMARK COMPARISON TABLE
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("               FINAL MODEL COMPARISON SUMMARY")
    print("=" * 75)
    print(f"{'Metric':<25} | {'Custom 2D-CNN':<20} | {'Pretrained ResNet18':<20}")
    print("-" * 75)
    print(f"{'Test Accuracy':<25} | {metrics_custom['accuracy']*100:6.2f}%              | {metrics_resnet['accuracy']*100:6.2f}%")
    print(f"{'Macro Precision':<25} | {metrics_custom['precision_macro']*100:6.2f}%              | {metrics_resnet['precision_macro']*100:6.2f}%")
    print(f"{'Macro Recall':<25} | {metrics_custom['recall_macro']*100:6.2f}%              | {metrics_resnet['recall_macro']*100:6.2f}%")
    print(f"{'Macro F1-Score':<25} | {metrics_custom['f1_macro']*100:6.2f}%              | {metrics_resnet['f1_macro']*100:6.2f}%")
    print(f"{'Training Time':<25} | {custom_train_time/60:6.2f} mins            | {resnet_train_time/60:6.2f} mins")
    print("=" * 75)
    print("✅ Experiment completed successfully! Artifacts stored in 'models/' and 'results/'\n")

if __name__ == "__main__":
    run_experiment()
