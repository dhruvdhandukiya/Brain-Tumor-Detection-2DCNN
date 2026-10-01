import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

def evaluate_model_on_test(model, test_loader, device, class_names):
    """
    Evaluates a trained model on the test dataset and computes performance metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    return metrics, cm, all_labels, all_preds


def plot_training_curves(history_custom, history_resnet, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Loss Comparison
    plt.figure(figsize=(10, 5), dpi=300)
    epochs_c = range(1, len(history_custom['train_loss']) + 1)
    epochs_r = range(1, len(history_resnet['train_loss']) + 1)

    plt.plot(epochs_c, history_custom['train_loss'], 'b--', label='Custom CNN (Train Loss)', linewidth=2)
    plt.plot(epochs_c, history_custom['val_loss'], 'b-', label='Custom CNN (Val Loss)', linewidth=2.5)
    plt.plot(epochs_r, history_resnet['train_loss'], 'r--', label='ResNet18 (Train Loss)', linewidth=2)
    plt.plot(epochs_r, history_resnet['val_loss'], 'r-', label='ResNet18 (Val Loss)', linewidth=2.5)

    plt.title('Training & Validation Loss: Custom 2D-CNN vs Pretrained ResNet18', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_comparison.png"))
    plt.close()

    # 2. Accuracy Comparison
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(epochs_c, [a*100 for a in history_custom['train_acc']], 'b--', label='Custom CNN (Train Acc)', linewidth=2)
    plt.plot(epochs_c, [a*100 for a in history_custom['val_acc']], 'b-', label='Custom CNN (Val Acc)', linewidth=2.5)
    plt.plot(epochs_r, [a*100 for a in history_resnet['train_acc']], 'r--', label='ResNet18 (Train Acc)', linewidth=2)
    plt.plot(epochs_r, [a*100 for a in history_resnet['val_acc']], 'r-', label='ResNet18 (Val Acc)', linewidth=2.5)

    plt.title('Training & Validation Accuracy: Custom 2D-CNN vs Pretrained ResNet18', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_comparison.png"))
    plt.close()


def plot_confusion_matrices(cm_custom, cm_resnet, class_names, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Custom CNN Confusion Matrix
    sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=axes[0], annot_kws={"size": 12})
    axes[0].set_title('Custom 2D-CNN Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    
    # ResNet18 Confusion Matrix
    sns.heatmap(cm_resnet, annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=axes[1], annot_kws={"size": 12})
    axes[1].set_title('Pretrained ResNet18 Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrices.png"))
    plt.close()


def plot_metric_bars(metrics_custom, metrics_resnet, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    custom_scores = [
        metrics_custom['accuracy'] * 100,
        metrics_custom['precision_macro'] * 100,
        metrics_custom['recall_macro'] * 100,
        metrics_custom['f1_macro'] * 100
    ]
    resnet_scores = [
        metrics_resnet['accuracy'] * 100,
        metrics_resnet['precision_macro'] * 100,
        metrics_resnet['recall_macro'] * 100,
        metrics_resnet['f1_macro'] * 100
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    rects1 = ax.bar(x - width/2, custom_scores, width, label='Custom 2D-CNN', color='#2563eb', edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + width/2, resnet_scores, width, label='Pretrained ResNet18', color='#ea580c', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison: Custom 2D-CNN vs Pretrained ResNet18', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison_bar.png"))
    plt.close()
