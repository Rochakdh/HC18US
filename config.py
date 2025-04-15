import torch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
SEED =  42
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
FOLD = 5
DROPOUT = 0.3
TRAIN_DATA_PATH = "./src/train_set/"
TEST_DATA_PATH = "./src/test_set/"
SAVE_MODEL_PATH = './model/model.pth'
ANNOTATION_FILE = "./src/train_generated.csv"
TEST_ANNONATION_FILE = "./src/test_generated.csv"
PREPROCESSED_DIR = "./src/generated_training_set/"
TEST_PREPROCESSED_DIR = "./src/generated_test_set/"
IMG_DIR = "./src/training_set"
CHECKPOINT_DIR = './models/checkpoint/'
LOG_DIR = './logs/'
TRAIN_OUTPUT_DIR = './predicted_mask'
NUM_FOLD = 5
SET_TEST_FOLDS = [1,2,3,4,5] #test for all the folds or individual folds 