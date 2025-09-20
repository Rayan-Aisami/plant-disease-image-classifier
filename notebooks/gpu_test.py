import sys
import os

# Add project root to sys.path for imports from src/
project_root = os.path.dirname(os.path.dirname(__file__))  # Assumes notebooks/ is under root
sys.path.append(project_root)

import torch
from src.model import PlantClassifier
from src.data_loader import get_loaders

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PlantClassifier(num_classes=38).to(device)
    train_loader, _, _, _ = get_loaders(batch_size=4)

    # Test forward pass on GPU
    model.eval()
    with torch.no_grad():
        imgs, _ = next(iter(train_loader))
        imgs = imgs.to(device)
        outputs = model(imgs)
        print(f"Output shape: {outputs.shape} on {outputs.device}")