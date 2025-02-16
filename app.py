from scripts.data import CustomUltrasoundDataset
from config import *

class HC18US:
    def __init__(self, dataset, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, lr = LEARNING_RATE, num_fold=FOLD,device=DEVICE):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device
        self.num_fold = num_fold
