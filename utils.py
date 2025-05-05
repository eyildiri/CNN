import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_class_distribution():
    # Chargement des infos pour récap
    print("\n--- Statistiques globales ---")

    splits = ["train", "test"]
    base_path = "multi_class_new_data"

    for split in splits:
        print(f"\n Split : {split}")
        split_path = os.path.join(base_path, split)
        for class_name in sorted(os.listdir(split_path)):
            class_path = os.path.join(split_path, class_name)
            n_images = len(os.listdir(class_path))
            print(f"  Classe '{class_name}' : {n_images} images")

    # Afficher le nombre d'hémorragies épidurales uniques (sans doublon)
    labels_df = pd.read_feather('data/meta/meta/labels.fth')
    epidural_count = labels_df[labels_df["epidural"] == 1]["ID"].nunique()
    print(f"\nNombre d'hémorragies épidurales uniques dans labels.fth : {epidural_count}")

    # Optionnel : afficher un exemple par classe (via le fichier labels.fth)
    print("\n--- Exemples d'images par classe (uniquement avec 1 seul label) ---")
    labels_df["filename"] = labels_df["ID"] + ".jpg"
    label_cols = ["epidural", "subdural", "subarachnoid", "intraparenchymal", "intraventricular"]

    # ajout colonne label
    def get_class(row):
        for col in label_cols:
            if row[col] == 1:
                return col
        return "none"

    labels_df["nb_labels"] = labels_df[label_cols].sum(axis=1)
    single_label_df = labels_df[labels_df["nb_labels"] == 1].copy()
    single_label_df["label"] = single_label_df.apply(get_class, axis=1)

    # un exemple par classe
    for label in sorted(single_label_df["label"].unique()):
        ex = single_label_df[single_label_df["label"] == label].iloc[0]
        print(f"\n🔹 Exemple pour '{label}':")
        print(ex[["ID", "any", *label_cols]])

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def printHeadLabels():
    labels_df = pd.read_feather('data/meta/meta/labels.fth')
    print(labels_df.head())