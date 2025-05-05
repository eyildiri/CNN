import os
from prepare_data import prepare_dataset
from utils import plot_class_distribution
from utils import printHeadLabels

if not os.path.exists('multi_class_new_data'):
    print("Dossier multi_class_new_data/ non trouvé. Préparation des données...")
    prepare_dataset()
else:
    print("Dossier multi_class_new_data/ déjà présent. Rien à faire.")

# Appel après préparation
plot_class_distribution()
#printHeadLabels()
