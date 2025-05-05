import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from cnn_model import get_model

if __name__ == '__main__':
    print("Chargement du dataset de test...")

    data_dir = "multi_class_new_data/test"
    batch_size = 16

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_dataset = datasets.ImageFolder(data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    class_names = test_dataset.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accuracies = []
    n_folds = 5
    for fold in range(1, n_folds + 1):
        model_path = f"ct_cnn_best_fold{fold}.pth"
        print(f"\nÉvaluation du modèle du fold {fold} ({model_path})...")
        model = get_model(device)
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
        print(f"Accuracy sur le jeu de test (fold {fold}) : {acc*100:.2f}%")
        accuracies.append(acc)

        cm = confusion_matrix(all_labels, all_preds)
        print("Matrice de confusion :")
        print(cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold} - Accuracy: {acc*100:.2f}%")
        plt.savefig(f"confusion_matrix_test_fold{fold}.png")
        plt.close()

    print(f"\nMoyenne accuracy sur le test set: {np.mean(accuracies)*100:.2f}% (+/- {np.std(accuracies)*100:.2f}%)")
