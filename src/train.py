"""Training module"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm

from .utils import Config, set_seed, save_metrics
from .data import load_cifar10
from models import CNNBaseline, CNNExtension

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
            return False
        
        if val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

def train_model(color_space="rgb", max_epochs=None, use_extension=False):
    """Train CNN model"""
    
    if max_epochs is None:
        max_epochs = Config.MAX_EPOCHS
    
    print(f"\n{'='*60}")
    print(f"Training with {color_space.upper()} color space")
    if use_extension:
        print("Using Extension: Learnable Color Transform Layer")
    print(f"{'='*60}")
    
    # Load data
    train_loader, val_loader, test_loader = load_cifar10(color_space)
    
    # Initialize model
    if use_extension:
        model = CNNExtension(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    else:
        model = CNNBaseline(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, 
                          weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE)
    
    # Tracking
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\nTraining for {max_epochs} epochs...")
    print("-" * 80)
    
    start_time = datetime.now()
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            model_name = f"best_{color_space}"
            if use_extension:
                model_name += "_extension"
            torch.save(model.state_dict(), 
                      os.path.join(Config.CHECKPOINTS_DIR, f"{model_name}.pt"))
        
        # Print progress
        train_val_gap = train_acc - val_acc
        print(f"Epoch {epoch+1:2d}/{max_epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_acc:5.2f}% | Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:5.2f}% | Gap: {train_val_gap:4.2f}%")
        
        # Early stopping
        if early_stopping(val_acc):
            print(f"\n🛑 Early stopping at epoch {epoch+1} (Best val acc: {best_val_acc:.2f}%)")
            break
    
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"\n✓ Training completed in {training_time:.2f} seconds")
    print(f"✓ Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    # Load best model for testing
    model_name = f"best_{color_space}"
    if use_extension:
        model_name += "_extension"
    model.load_state_dict(torch.load(os.path.join(Config.CHECKPOINTS_DIR, f"{model_name}.pt")))
    
    # Evaluate on test set
    test_acc = evaluate_model(model, test_loader)
    print(f"✓ Final test accuracy: {test_acc:.2f}%")
    
    # Save metrics
    metrics = {
        "color_space": color_space,
        "use_extension": use_extension,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "training_time_seconds": training_time,
        "epochs_completed": len(train_losses),
        "train_accuracies": train_accs,
        "val_accuracies": val_accs,
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    
    model_suffix = f"{color_space}_extension" if use_extension else color_space
    save_metrics(metrics, f"metrics_{model_suffix}.json")
    
    return model, test_loader, metrics

def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description="Train CNN on CIFAR-10")
    parser.add_argument("--colorspace", type=str, default="rgb", 
                        choices=["rgb", "hsv", "lab"],
                        help="Color space to use")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs")
    parser.add_argument("--use-extension", action="store_true",
                        help="Use learnable color transform extension")
    
    args = parser.parse_args()
    
    set_seed(42)
    Config.create_dirs()
    
    train_model(args.colorspace, args.epochs, args.use_extension)

if __name__ == "__main__":
    main()