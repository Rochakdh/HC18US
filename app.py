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
import time
import numpy as np 

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
        """Loads the latest checkpoint for a fold and resumes training from the last saved epoch."""
        checkpoint_files = sorted(
            [f for f in os.listdir(self.checkpoint_dir) if f.startswith(f"fold_{fold+1}_epoch_")],
            key=lambda x: int(x.split("_epoch_")[1].split(".pt")[0])  # Sort by epoch number
        )
        
        if checkpoint_files:
            latest_checkpoint = os.path.join(self.checkpoint_dir, checkpoint_files[-1])  # Get last checkpoint
            checkpoint = torch.load(latest_checkpoint, map_location=self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']  # Resume from next epoch
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])

            tqdm.write(f"Resuming from {latest_checkpoint} (Epoch {start_epoch})")
        else:
            start_epoch = 0
            train_losses, val_losses = [], []
            tqdm.write(f"No checkpoint found at {self.checkpoint_dir}. Starting from scratch.")

        return start_epoch, train_losses, val_losses


    def train_on_epoch(self, model, train_data_loader, optimizer, epoch, fold):
        """Trains for one epoch with tqdm tracking loss and batch time."""
        model.train()
        running_loss = 0.
        start_time = time.time()  # Track epoch start time

        progress_bar = tqdm(train_data_loader, desc=f"Training Fold {fold+1} Epoch {epoch+1}", leave=False)
        
        for batch_idx, (image, mask) in enumerate(progress_bar):
            batch_start = time.time()  # Track batch time

            image, mask = image.to(self.device), mask.to(self.device)
            optimizer.zero_grad()
            outputs = model(image)
            loss = dice_loss(outputs, mask)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss
            batch_time = time.time() - batch_start  # Compute batch processing time
            
            # Update tqdm bar with loss and batch time
            progress_bar.set_postfix({
                "Batch Loss": f"{batch_loss:.4f}",
                "Avg Loss": f"{running_loss / (batch_idx + 1):.4f}",
                "Batch Time": f"{batch_time:.3f}s"
            })

        epoch_loss = running_loss / len(train_data_loader)
        epoch_time = time.time() - start_time  # Compute epoch processing time

        tqdm.write(f"Epoch {epoch+1} Completed: Train Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")
        
        self.writer.add_scalar(f'Fold_{fold+1}/Train_Loss', epoch_loss, epoch)
        return epoch_loss

    def val_on_epoch(self, model, val_data_loader, epoch, fold):
        """Validates for one epoch with tqdm tracking loss and batch time."""
        model.eval()
        running_loss = 0.0
        start_time = time.time()

        progress_bar = tqdm(val_data_loader, desc=f"Validating Fold {fold+1} Epoch {epoch+1}", leave=False)

        if epoch % 2 == 0:  # Visualize every 2 epochs
            self.visualize_predictions(model, val_data_loader, epoch)

        with torch.no_grad():
            for batch_idx, (image, mask) in enumerate(progress_bar):
                batch_start = time.time()

                image, mask = image.to(self.device), mask.to(self.device)
                outputs = model(image)
                loss = dice_loss(outputs, mask)

                batch_loss = loss.item()
                running_loss += batch_loss
                batch_time = time.time() - batch_start

                progress_bar.set_postfix({
                    "Batch Loss": f"{batch_loss:.4f}",
                    "Avg Loss": f"{running_loss / (batch_idx + 1):.4f}",
                    "Batch Time": f"{batch_time:.3f}s"
                })
        epoch_loss = running_loss / len(val_data_loader)
        epoch_time = time.time() - start_time

        tqdm.write(f"Epoch {epoch+1} Completed: Val Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")

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

            start_epoch, train_loss, val_loss = self.load_checkpoint(model, optimizer, fold)

            best_val_loss = float("inf")

            for epoch in range(start_epoch,self.num_epochs):
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
    
    def visualize_predictions(self, model, dataloader, epoch):
        """ Visualizes model predictions on a few samples from the validation set. """
        model.eval()  # Set model to evaluation mode
        images, masks = next(iter(dataloader))  # Get a batch of validation images
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        with torch.no_grad():
            preds = model(images)  # Get model predictions
            preds = torch.sigmoid(preds)  # Apply sigmoid if using BCE loss
            preds = (preds > 0.5).float()  # Convert to binary mask

        # Convert to numpy for visualization
        images = images.cpu().numpy().squeeze()
        masks = masks.cpu().numpy().squeeze()
        preds = preds.cpu().numpy().squeeze()

        # Plot images, ground truth, and predictions
        fig, axes = plt.subplots(len(images), 3, figsize=(10, 5 * len(images)))
        if len(images) == 1:
            axes = [axes]  # Ensure proper indexing if batch size is 1

        for i in range(len(images)):
            axes[i][0].imshow(images[i], cmap='gray')
            axes[i][0].set_title("Input Image")

            axes[i][1].imshow(masks[i], cmap='gray')
            axes[i][1].set_title("Ground Truth Mask")

            axes[i][2].imshow(preds[i], cmap='gray')
            axes[i][2].set_title(f"Predicted Mask (Epoch {epoch})")

            for ax in axes[i]:
                ax.axis("off")
        plt.savefig(f'./predicted_mask/predictions_epoch_{epoch}.png')
        # plt.tight_layout()
        # plt.show()


# Load Dataset
dataset = CustomUltrasoundDataset(
    annnootation_file="./src/training_set_pixel_size_and_HC.csv",
    image_dir="./src/training_set"
)

# Train Model with 5-Fold Cross Validation
trainer = HC18US(dataset, batch_size=2, num_epochs=100, lr=0.001, num_folds=5)
trainer.cross_validation()

# Plot Losses after training
trainer.plot_losses()