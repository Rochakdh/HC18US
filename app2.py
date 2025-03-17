import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from models.model import UNet
from scripts.data import CustomUltrasoundDataset
from scripts.loss import dice_loss
from config import *
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Config: 0 for DDP (multi-GPU), 1 for single GPU
RUN_MODE = 0  

def setup(rank, world_size):
    """Initializes distributed training for multi-GPU."""
    os.environ["MASTER_ADDR"] = "localhost"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Destroys the distributed process group after training."""
    dist.destroy_process_group()

def setup_device():
    """Determines device setup based on RUN_MODE."""
    if RUN_MODE == 0 and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
        print(f"Running on {world_size} GPUs with Distributed Data Parallel (DDP).")
        return "ddp", world_size
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on a single GPU: {device}" if torch.cuda.is_available() else "Using CPU.")
        return device, 1

class HC18US:
    def __init__(self, dataset, batch_size, num_epochs, lr, num_folds, checkpoint_dir, log_dir, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.num_folds = num_folds
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.device = device
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    @staticmethod
    def train_fold(rank, world_size, dataset, batch_size, num_epochs, lr, fold, train_idx, val_idx, checkpoint_dir, device):
        """Static method to allow `mp.spawn` to work correctly in DDP."""
        if RUN_MODE == 0:  
            setup(rank, world_size)
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device(device)

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        if RUN_MODE == 0:
            train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            train_sampler, val_sampler = None, None

        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, sampler=val_sampler, num_workers=2, pin_memory=True)

        model = UNet(in_channels=1, out_channels=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scaler = torch.amp.GradScaler()

        if RUN_MODE == 0:
            model = DDP(model, device_ids=[rank], output_device=rank)

        start_epoch, train_losses, val_losses = HC18US.load_checkpoint(model, optimizer, fold, checkpoint_dir, device)

        best_val_loss = float("inf")

        for epoch in range(start_epoch, num_epochs):
            if RUN_MODE == 0:
                train_sampler.set_epoch(epoch)

            train_loss = HC18US.train_on_epoch(model, train_loader, optimizer, scaler, epoch, fold, device, rank)
            val_loss = HC18US.val_on_epoch(model, val_loader, epoch, fold, device, rank)

            if rank == 0 or RUN_MODE == 1:
                print(f"Fold {fold+1}, Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                HC18US.save_checkpoint(model, optimizer, epoch, fold, train_loss, val_loss, checkpoint_dir)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    HC18US.save_checkpoint(model, optimizer, epoch, fold, train_loss, val_loss, checkpoint_dir, best=True)

        if RUN_MODE == 0:
            cleanup()

    @staticmethod
    def train_on_epoch(model, train_loader, optimizer, scaler, epoch, fold, device, rank):
        """Trains for one epoch in DDP or single GPU mode."""
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False, position=rank):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = dice_loss(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    @staticmethod
    def val_on_epoch(model, val_loader, epoch, fold, device, rank):
        """Validates for one epoch in DDP or single GPU mode."""
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Val Epoch {epoch+1}", leave=False, position=rank):
                images, masks = images.to(device), masks.to(device)

                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = dice_loss(outputs, masks)

                running_loss += loss.item()

        return running_loss / len(val_loader)

    @staticmethod
    def save_checkpoint(model, optimizer, epoch, fold, train_loss, val_loss, checkpoint_dir, best=False):
        """Saves model checkpoint."""
        if RUN_MODE == 1 or (RUN_MODE == 0 and dist.get_rank() == 0):
            checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold+1}_epoch_{epoch+1}.pt")
            if best:
                checkpoint_path = os.path.join(checkpoint_dir, f"best_model_fold_{fold+1}.pt")

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if RUN_MODE == 0 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            tqdm.write(f"Checkpoint saved: {checkpoint_path}")

    @staticmethod
    def load_checkpoint(model, optimizer, fold, checkpoint_dir, device):
        """Loads the latest checkpoint for a fold and resumes training from the last saved epoch."""
        checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith(f"fold_{fold+1}_epoch_")], key=lambda x: int(x.split("_epoch_")[1].split(".pt")[0]))

        if checkpoint_files:
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            tqdm.write(f"Resuming from {latest_checkpoint} (Epoch {checkpoint['epoch']})")
            return checkpoint['epoch'], checkpoint.get('train_losses', []), checkpoint.get('val_losses', [])

        return 0, [], []

if __name__ == "__main__":
    device, world_size = setup_device()
    dataset = CustomUltrasoundDataset("./src/training_set_pixel_size_and_HC.csv", "./src/training_set", transform=A.Compose([A.HorizontalFlip(p=0.5), A.Rotate(limit=20, p=0.5), A.Normalize(mean=[0.5], std=[0.5]), ToTensorV2()]))
    trainer = HC18US(dataset, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, 5, CHECKPOINT_DIR, LOG_DIR, device)
    trainer.cross_validation(world_size)
