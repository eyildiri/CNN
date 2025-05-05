import os
import shutil
import pandas as pd
from tqdm import tqdm

def prepare_dataset(src_dir='data/train_jpg/train_jpg', meta_path='data/meta/meta/labels.fth', target_dir='multi_class_new_data', max_per_class=5000, max_test_per_class=200):

    print("Chargement des labels...")
    labels_df = pd.read_feather(meta_path)

    label_cols = ["epidural", "subdural", "subarachnoid", "intraparenchymal", "intraventricular"]
    available_images = set(os.listdir(src_dir))

    labels_df["filename"] = labels_df["ID"] + ".jpg"
    labels_df = labels_df[labels_df["filename"].isin(available_images)].reset_index(drop=True)
    labels_df["nb_labels"] = labels_df[label_cols].sum(axis=1)

    # Garder uniquement les images avec 1 seul type d'hémorragie (ou none)
    single_label_df = labels_df[labels_df["nb_labels"] <= 1].copy()

    # Ajouter la colonne de classe
    def get_class(row):
        for col in label_cols:
            if row[col] == 1:
                return col
        return "none"

    single_label_df["label"] = single_label_df.apply(get_class, axis=1)

    print("Échantillonnage équilibré...")
    sampled_dfs = []
    test_dfs = []
    for cls in single_label_df["label"].unique():
        df_cls = single_label_df[single_label_df["label"] == cls]
        n_total = len(df_cls)

        # max 200 pour test, ou 50% du total si classe petite
        n_test = min(max_test_per_class, n_total // 2)
        n_train = min(max_per_class, n_total - n_test)

        df_cls_shuffled = df_cls.sample(frac=1, random_state=42).reset_index(drop=True)
        train_sampled = df_cls_shuffled.iloc[:n_train]
        test_sampled = df_cls_shuffled.iloc[n_train:n_train + n_test]

        sampled_dfs.append(train_sampled)
        test_dfs.append(test_sampled)

    train_df = pd.concat(sampled_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)

    print("Création des dossiers et copie des fichiers...")

    class_names = train_df["label"].unique().tolist()
    if "none" not in class_names:
        class_names.append("none")

    for split_name, df in [("train", train_df), ("test", test_df)]:
        for class_name in class_names:
            class_path = os.path.join(target_dir, split_name, class_name)
            os.makedirs(class_path, exist_ok=True)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copie {split_name}"):
            label = row["label"]
            fname = row["filename"]
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(target_dir, split_name, label, fname)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)

    print("Préparation terminée.")