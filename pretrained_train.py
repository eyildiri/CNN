import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import os
from utils import plot_learning_curves
import time
from torchvision.datasets.folder import default_loader
from pretrained_model import get_pretrained_resnet18
from sklearn.model_selection import train_test_split

class TransformedDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.subset.dataset.samples[self.subset.indices[index]]
        img = default_loader(path)
        return self.transform(img), label

    def __len__(self):
        return len(self.subset)

class AugmentedEpiduralDataset(Dataset):
    def __init__(self, subset, transform_general, transforms_epidural):
        self.transform_general = transform_general
        self.transforms_epidural = transforms_epidural
        self.samples = []

        for idx in subset.indices:
            path, label = subset.dataset.samples[idx]
            self.samples.append((path, label, transform_general))

            if subset.dataset.classes[label] == 'epidural':
                for transform in transforms_epidural:
                    self.samples.append((path, label, transform))

    def __getitem__(self, index):
        path, label, transform = self.samples[index]
        img = default_loader(path)
        return transform(img), label

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':
    data_dir = "multi_class_new_data/train"
    batch_size = 32 
    num_epochs = 15
    lr = 1e-4
    model_save_path = "resnet18_best"
    curve_save_path = "learning_curves_resnet"
    early_stopping_patience = 5

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_epidural_1 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_epidural_2 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(data_dir)
    targets = [sample[1] for sample in full_dataset.samples]
    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
        stratify=targets,
        random_state=42
    )

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_dataset = AugmentedEpiduralDataset(
        train_subset,
        transform_general=transform_train,
        transforms_epidural=[transform_epidural_1, transform_epidural_2]
    )
    val_dataset = TransformedDataset(val_subset, transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_pretrained_resnet18(num_classes=6, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        acc_gap = train_acc - val_acc
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Acc Gap: {acc_gap:.4f}")

        epoch_end = time.time()
        elapsed = epoch_end - epoch_start
        total_elapsed = epoch_end - start_time
        remaining = (num_epochs - (epoch + 1)) * elapsed

        print(f"Epoch {epoch+1}/{num_epochs} terminé en {elapsed:.2f}s "
              f"| Temps écoulé : {total_elapsed:.2f}s | Temps restant estimé : {remaining:.2f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), f"{model_save_path}.pth")

        if epoch - best_epoch >= early_stopping_patience:
            print(f"Early stopping: aucune amélioration de la val_acc depuis {early_stopping_patience} epochs.")
            break

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n Temps total d'entraînement : {elapsed:.2f} secondes")
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_path=f"{curve_save_path}.png")
    print(f"Meilleur modèle sauvegardé dans {model_save_path}.pth")
