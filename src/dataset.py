import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASS_NAMES_4 = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_NAMES_BINARY = ['no_tumor', 'tumor']

def get_transforms(img_size=224):
    """
    Returns training and validation/testing data transforms with normalization
    and data augmentation as used in academic CNN/ResNet MRI pipelines.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


class BrainMRIDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor MRI Scans.
    Supports both 4-class multi-classification and binary tumor detection.
    """
    def __init__(self, root_dir, split="Training", transform=None, mode="4class"):
        """
        root_dir: Base directory of dataset containing Training/ and Testing/
        split: 'Training' or 'Testing'
        mode: '4class' (glioma, meningioma, notumor, pituitary) or 'binary' (no_tumor, tumor)
        """
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.mode = mode
        self.samples = []
        self.class_to_idx = {}

        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        raw_classes = sorted([d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d)) and not d.startswith('.')])
        
        if self.mode == "4class":
            self.classes = CLASS_NAMES_4
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            for cls_name in raw_classes:
                normalized_name = cls_name.lower().replace(" ", "").replace("_", "")
                target_idx = None
                for std_cls in self.classes:
                    if std_cls.replace("_", "") in normalized_name or normalized_name in std_cls.replace("_", ""):
                        target_idx = self.class_to_idx[std_cls]
                        break
                if target_idx is None:
                    continue

                cls_folder = os.path.join(self.split_dir, cls_name)
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        self.samples.append((os.path.join(cls_folder, fname), target_idx))
        else:
            # Binary mode: 0 -> no_tumor, 1 -> tumor
            self.classes = CLASS_NAMES_BINARY
            self.class_to_idx = {'no_tumor': 0, 'tumor': 1}
            for cls_name in raw_classes:
                normalized_name = cls_name.lower()
                is_no_tumor = "notumor" in normalized_name or "no_tumor" in normalized_name or "negative" in normalized_name or normalized_name == "no"
                target_idx = 0 if is_no_tumor else 1

                cls_folder = os.path.join(self.split_dir, cls_name)
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        self.samples.append((os.path.join(cls_folder, fname), target_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, path


def get_dataloaders(data_dir, batch_size=32, num_workers=2, mode="4class", val_ratio=0.15):
    """
    Creates train, validation, and test dataloaders.
    """
    train_transform, val_transform = get_transforms(img_size=224)

    full_train_dataset = BrainMRIDataset(data_dir, split="Training", transform=train_transform, mode=mode)
    test_dataset = BrainMRIDataset(data_dir, split="Testing", transform=val_transform, mode=mode)

    # Train/Val split from Training set
    total_train = len(full_train_dataset)
    val_size = int(total_train * val_ratio)
    train_size = total_train - val_size

    train_subset, val_subset = torch.utils.data.random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, full_train_dataset.classes
