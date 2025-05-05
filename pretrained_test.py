import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from pretrained_model import get_pretrained_resnet18

if __name__ == '__main__':
    print("Chargement du dataset de test...")

    data_dir = "multi_class_new_data/test"
    batch_size = 16

    # Transformations compatibles avec ResNet préentraîné (ImageNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    class_names = test_dataset.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "resnet18_best.pth"
    print(f"\nÉvaluation du modèle ({model_path})...")

    model = get_pretrained_resnet18(num_classes=6, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Accuracy sur le jeu de test : {acc*100:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    print("Matrice de confusion :")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Resnet18 Test Accuracy: {acc*100:.2f}%")
    plt.savefig("confusion_matrix_resnet18.png")
    plt.close()
