import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
SEED =  42
BATCH_SIZE = 2
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
FOLD = 5
DROPOUT = 0.3
TRAIN_DATA_PATH = "./src/train_set/"
TEST_DATA_PATH = "./src/test_set/"
SAVE_MODEL_PATH = './model/model.pth'
ANNOTATION_FILE = "./src/training_set_pixel_size_and_HC.csv"
IMG_DIR = "./src/training_set"
CHECKPOINT_DIR = './models/checkpoint/'
LOG_DIR = './logs/'
TRAIN_OUTPUT_DIR = './predicted_mask'