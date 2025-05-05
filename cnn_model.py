import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CNN, self).__init__()

        # 3 canaux d'entrée (images gris), 16 filtres, kernel 3x3 
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 3x pooling (MaxPool2d) avec un kernel de 2x2 et un stride de 2 (réduit la taille de moitié après chaque couche)
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout pour éviter le surapprentissage (0.3 est une valeur courante)
        self.dropout = nn.Dropout(0.3)

        # Après 3x MaxPool2d sur 256x256 :
        # 256 → 128 → 64 → 32
        # Donc : 64 canaux * 32 * 32 = 65536
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [batch, 3, 256, 256]
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch, 32, 128, 128]
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch, 64, 64, 64]
        x = self.pool(F.relu(self.conv3(x)))  # -> [batch, 128, 32, 32]

        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_model(device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    return model
