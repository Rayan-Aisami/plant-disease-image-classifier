import sys
import os

# Add project root to sys.path for imports from src/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Go up to root from src/
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import wandb  # For logging; run wandb.init()
from tqdm import tqdm  # For progress bars
from src.data_loader import get_loaders
from src.model import PlantClassifier

if __name__ == '__main__':
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters (tune as needed)
    num_epochs = 10
    batch_size = 32  # Adjustable; 32-64 good for RTX 3080
    learning_rate = 0.001
    patience_early_stop = 5  # Stop if no val improvement

    # Load data
    train_loader, val_loader, test_loader, classes = get_loaders(batch_size=batch_size)
    num_classes = len(classes)

    # Compute class weights for imbalances
    train_labels = [label for _, label in train_loader.dataset]  # Extract all train labels
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Model, loss, optimizer, scheduler
    model = PlantClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)  # No verbose

    # WandB setup (optional; comment out if not using)
    wandb.init(project="plant-disease-classifier", config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "architecture": "ResNet50"
    })

    best_val_acc = 0.0
    early_stop_count = 0

    for epoch in range(num_epochs):
        # Training loop with tqdm progress bar
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}", leave=False)
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            # Update bar with running averages
            train_bar.set_postfix(loss=train_loss / (train_bar.n + 1), acc=train_correct / ((train_bar.n + 1) * batch_size))

        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_loader.dataset)

        # Validation loop with tqdm progress bar
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_bar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{num_epochs}", leave=False)
        with torch.no_grad():
            for imgs, labels in val_bar:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                # Update bar with running averages
                val_bar.set_postfix(loss=val_loss / (val_bar.n + 1), acc=val_correct / ((val_bar.n + 1) * batch_size))

        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Scheduler and early stopping
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        if optimizer.param_groups[0]['lr'] < prev_lr:
            print(f"Learning rate reduced to {optimizer.param_groups[0]['lr']}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= patience_early_stop:
                print("Early stopping triggered.")
                break

        # WandB log
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]['lr']})

    wandb.finish()
    print("Training complete. Best model saved in models/best_model.pth")