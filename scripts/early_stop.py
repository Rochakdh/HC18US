class EarlyStopping:
    def __init__(self, patience=10, delta=0.00, verbose=True):
        """
        Early stopping utility that stops training when validation loss doesn't improve after a given patience.
        
        Args:
            patience (int): Number of epochs to wait after last improvement.
            delta (float): Minimum change in loss to qualify as improvement.
            verbose (bool): If True, prints when early stopping counter increases.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
