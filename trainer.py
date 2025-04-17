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
from scripts.loss import bce_dice_focal_loss
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
import numpy as np 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from settings import set_deterministics
from scripts.early_stop import EarlyStopping

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

from scripts.grad_cam_tgt import SemanticSegmentationTarget



class HC18US:
    def __init__(self, dataset, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, num_folds=FOLD, 
                 device=DEVICE, checkpoint_dir=CHECKPOINT_DIR, log_dir=LOG_DIR):
        set_deterministics()
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
            loss = bce_dice_focal_loss(outputs, mask)

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
                loss = bce_dice_focal_loss(outputs, mask)

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

            model = UNet().to(self.device)

            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

            start_epoch, train_loss, val_loss = self.load_checkpoint(model, optimizer, fold)

            best_val_loss = float("inf")

            early_stopper = EarlyStopping()

            for epoch in range(start_epoch,self.num_epochs):
                train_loss = self.train_on_epoch(model, train_loader, optimizer, epoch, fold)
                val_loss = self.val_on_epoch(model, val_loader, epoch, fold)
                # scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']

                print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR ={current_lr}")

                self.save_checkpoint(model, optimizer, epoch, fold, train_loss, val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(model, optimizer, epoch, fold, train_loss, val_loss, best=True)
                
                early_stopper(val_loss)

                if early_stopper.early_stop:
                    print("Early stopping triggered.")
                    break

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
        print(f"Visualizing predictions for epoch {epoch}...")
        model.eval()

        images, masks = next(iter(dataloader))
        images, masks = images.to(self.device), masks.to(self.device)

        with torch.no_grad():
            outputs = model(images)
            preds = torch.sigmoid(outputs)

        images_np = images.cpu().numpy()
        masks_np = masks.cpu().numpy()
        preds_np = preds.cpu().numpy()

        output_dir = TRAIN_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        batch_size = images.shape[0]
        fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))

        if batch_size == 1:
            axes = [axes]  # Ensure it's iterable

        target_layers = [model.down4.mpconv[1]]  # The second layer in down4 (i.e., double_conv)

        for i in range(batch_size):
            img_tensor = images[i].unsqueeze(0)  # (1, C, H, W)
            img = images_np[i].transpose(1, 2, 0)  # (H, W, C)
            img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)

            mask = masks_np[i][0] if masks_np[i].shape[0] == 1 else masks_np[i]
            pred = preds_np[i][0] if preds_np[i].shape[0] == 1 else preds_np[i]

            # Grad-CAM target
            target = SemanticSegmentationTarget(mask)

            # Run Grad-CAM
            with GradCAM(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=img_tensor, targets=[target])[0]

            cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)

            # Plotting
            axes[i][0].imshow(img.squeeze(), cmap='gray')
            axes[i][0].set_title("Input Image")

            axes[i][1].imshow(mask, cmap='gray')
            axes[i][1].set_title("Ground Truth Mask")

            axes[i][2].imshow(pred, cmap='gray')
            axes[i][2].set_title(f"Predicted Mask (Epoch {epoch})")

            axes[i][3].imshow(cam_image)
            axes[i][3].set_title("Grad-CAM")

            for ax in axes[i]:
                ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'predictions_with_cam_epoch_{epoch}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization to {save_path}")


transform = A.Compose([
    A.RandomCrop(width=512, height=512), 
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.2),
    ToTensorV2()
])


dataset = CustomUltrasoundDataset(
    annotation_file=ANNOTATION_FILE,
    preprocessed_dir=PREPROCESSED_DIR,
    transform=transform,
    augment=True
)


# Train Model with 5-Fold Cross Validation
trainer = HC18US(dataset, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, num_folds=NUM_FOLD)
trainer.cross_validation()

# Plot Losses after training
trainer.plot_losses()