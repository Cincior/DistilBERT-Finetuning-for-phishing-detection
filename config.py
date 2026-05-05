import torch

MODEL_NAME = "distilbert-base-uncased"
SAVE_PATH = "./saved_modelT"
SEED = 42
MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
TFIDF_MAX_FEATURES = 5000
COSINE_SIMILARITY_THRESHOLD = 0.9
COSINE_BATCH_SIZE = 2000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")