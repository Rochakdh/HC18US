from scripts.data import CustomUltrasoundDataset
from config import *
from models.model import UNet
from torch.utils.data import DataLoader,Subset
from torch.utils.data import random_split
from sklearn.model_selection import KFold
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.optim as optim
from scripts.loss import dice_loss
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class HC18US:
    def __init__(self, dataset, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, num_folds=FOLD, 
                 device=DEVICE, checkpoint_dir=CHECKPOINT_DIR, log_dir=LOG_DIR):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device
        self.num_folds = num_folds
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def save_checkpoint(self, model, optimizer, epoch, fold, train_loss, val_loss, best=False):
        """Saves model checkpoint including train/val loss."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"fold_{fold+1}_epoch_{epoch+1}.pt")
        if best:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"best_model_fold_{fold+1}.pt")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, checkpoint_path)
        tqdm.write(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, model, optimizer, fold):
        """Loads the latest checkpoint for a fold."""
        checkpoint_files = sorted(
            [f for f in os.listdir(self.checkpoint_dir) if f.startswith(f"fold_{fold+1}_epoch_")],
            key=lambda x: int(x.split("_epoch_")[1].split(".pt")[0])
        )
        if checkpoint_files:
            latest_checkpoint = os.path.join(self.checkpoint_dir, checkpoint_files[-1])
            checkpoint = torch.load(latest_checkpoint, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            tqdm.write(f"Resuming from {latest_checkpoint} (Epoch {start_epoch})")
        else:
            start_epoch = 0
            tqdm.write(f"No checkpoint found at {self.checkpoint_dir}")
        
        return start_epoch

    def train_on_epoch(self, model, train_data_loader, optimizer, epoch, fold):
        """Trains the model for one epoch and logs loss."""
        model.train()
        running_loss = 0.
        progress_bar = tqdm(train_data_loader, desc=f"Training Fold {fold+1} Epoch {epoch+1}", leave=False)
        for image, mask in progress_bar:
            image, mask = image.to(self.device), mask.to(self.device)
            optimizer.zero_grad()
            outputs = model(image)
            loss = dice_loss(outputs, mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_data_loader)
        self.writer.add_scalar(f'Fold_{fold+1}/Train_Loss', epoch_loss, epoch)
        return epoch_loss

    def val_on_epoch(self, model, val_data_loader, epoch, fold):
        """Validates the model for one epoch and logs loss."""
        model.eval()
        running_loss = 0.0
        progress_bar = tqdm(val_data_loader, desc=f"Validating Fold {fold+1} Epoch {epoch+1}", leave=False)
        with torch.no_grad():
            for image, mask in progress_bar:
                image, mask = image.to(self.device), mask.to(self.device)
                outputs = model(image)
                loss = dice_loss(outputs, mask)
                running_loss += loss.item()

        epoch_loss = running_loss / len(val_data_loader)
        self.writer.add_scalar(f'Fold_{fold+1}/Val_Loss', epoch_loss, epoch)
        return epoch_loss

    def cross_validation(self):
        """Performs k-fold cross-validation with training, validation, and checkpointing."""
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
            print(f"\nTraining Fold {fold + 1}/{self.num_folds}")

            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, drop_last=True)

            model = UNet(in_channels=1, out_channels=1).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)

            best_val_loss = float("inf")

            for epoch in range(self.num_epochs):
                train_loss = self.train_on_epoch(model, train_loader, optimizer, epoch, fold)
                val_loss = self.val_on_epoch(model, val_loader, epoch, fold)

                print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                self.save_checkpoint(model, optimizer, epoch, fold, train_loss, val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(model, optimizer, epoch, fold, train_loss, val_loss, best=True)

            fold_results.append(best_val_loss)

        print(f"Cross-Validation Results: {fold_results}")
        print(f"Mean Validation Loss: {np.mean(fold_results):.4f}")

    def plot_losses(self):
        """Plots training and validation losses from saved checkpoints."""
        train_losses, val_losses = [], []
        epochs = []

        for file in sorted(os.listdir(self.checkpoint_dir)):
            if file.endswith(".pt"):
                checkpoint = torch.load(os.path.join(self.checkpoint_dir, file), map_location=self.device)
                epochs.append(checkpoint['epoch'])
                train_losses.append(checkpoint.get('train_loss', None))
                val_losses.append(checkpoint.get('val_loss', None))

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label="Training Loss", marker="o")
        plt.plot(epochs, val_losses, label="Validation Loss", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid()
        plt.show()


# Load Dataset
dataset = CustomUltrasoundDataset(
    annnootation_file="./src/training_set_pixel_size_and_HC.csv",
    image_dir="./src/training_set"
)

# Train Model with 5-Fold Cross Validation
trainer = HC18US(dataset, batch_size=2, num_epochs=10, lr=0.001, num_folds=5)
trainer.cross_validation()

# Plot Losses after training
trainer.plot_losses()