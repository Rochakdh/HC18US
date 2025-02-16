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

class HC18US:
    def __init__(self, dataset, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, lr = LEARNING_RATE, num_folds=FOLD,device=DEVICE):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device
        self.num_folds = num_folds
    
    def train_on_epoch(self,model,train_data_loader,criterion,optimizer):
        model.train()
        running_loss = 0.0

        for image,mask in train_data_loader:
            image,mask = image.to(self.device), mask.to(self.device)
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs,mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss/len(train_data_loader)
    
    def val_on_epoch(self,model,val_data_loader,criterion):
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for image,mask in val_data_loader:
                image,mask = image.to(self.device), masks.to(self.device)
                outputs = model(image)
                loss = criterion(outputs,mask)
                running_loss += loss.item()
        return running_loss/len(val_data_loader)
    
    def cross_validation(self):
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
            print(f"Training Fold {fold + 1}/{self.num_folds}")

            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size)

            model = UNet(in_channels=1, out_channels=2).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.lr)

            for epoch in range(self.num_epochs):
                train_loss = self.train_on_epoch(model, train_loader, criterion, optimizer)
                val_loss = self.val_on_epoch(model, val_loader, criterion)

                print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            fold_results.append(val_loss)

        print(f"Cross-Validation Results: {fold_results}")
        print(f"Mean Validation Loss: {np.mean(fold_results):.4f}")

# Load Dataset
dataset = CustomUltrasoundDataset(
        annnootation_file="./src/training_set_pixel_size_and_HC.csv",
        image_dir="./src/training_set"
    )
print(dataset[0])
# Train Model with 5-Fold Cross Validation
trainer = HC18US(dataset, batch_size=2, num_epochs=10, lr=0.001, num_folds=5)
trainer.cross_validation()
    
    