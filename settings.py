import torch
from config import (DEVICE, SEED)

#set device
def configure_device(device):
    """
    Returns the specified device if provided, otherwise defaults to CUDA if available or CPU.
    
    Parameters:
        device (str or None): The device to use ('cuda', 'cpu').
    
    Returns:
        torch.device: The selected PyTorch device.
    """
    try:
        if device == "gpu" and torch.cuda.is_available():
            return torch.device("cuda")
        if device == "cpu":
            return torch.device("cpu")
    except:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_deterministics(seed=SEED):
    """
    Sets the PyTorch environment for deterministic results.
    
    Parameters:
        seed (int): The seed value for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False