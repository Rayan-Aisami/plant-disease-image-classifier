import sys
import os

# Add project root to sys.path for imports from src/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Go up to root from src/
sys.path.append(project_root)

import torch
from tqdm import tqdm  # For progress bar
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from src.data_loader import get_loaders
from src.model import PlantClassifier

if __name__ == '__main__':
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data (only need test_loader and classes)
    _, _, test_loader, classes = get_loaders(batch_size=32)

    # Load best model
    model = PlantClassifier(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.eval()

    # Evaluation loop with tqdm
    preds = []
    trues = []
    test_bar = tqdm(test_loader, desc="Evaluating on test set")
    with torch.no_grad():
        for imgs, labels in test_bar:
            imgs = imgs.to(device)
            outputs = model(imgs)
            pred = outputs.argmax(dim=1).cpu().numpy()
            true = labels.cpu().numpy()
            preds.extend(pred)
            trues.extend(true)

    # Metrics
    acc = accuracy_score(trues, preds)
    report = classification_report(trues, preds, target_names=classes, output_dict=True)
    cm = confusion_matrix(trues, preds)

    # Print summary
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(pd.DataFrame(report).transpose())

    # Confusion Matrix visualization
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved as confusion_matrix.png")

    # Optional: Per-class F1 to check imbalances
    f1_df = pd.DataFrame({'Class': classes, 'F1-Score': [report[c]['f1-score'] for c in classes]})
    print("Per-Class F1-Scores (sorted descending):")
    print(f1_df.sort_values('F1-Score', ascending=False))