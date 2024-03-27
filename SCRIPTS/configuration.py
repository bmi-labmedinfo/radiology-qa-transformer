import torch
from transformers import AutoTokenizer

#Parameters
class Config:
  #GLOBAL
  HOME_DIR = "..."
  INPUT_FILE_PATH = "..."
  OUTPUT_DIR = "..."
  RESULTS_DIR = "..."
  ENCODING ="utf-8"
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  SEEDS = [1, 42, 66, 6, 18]
  USE_PRETRAINED = True
  POST_PROCESSING = False

  #PREPROCESSING
  MAX_INPUT_TOKENS = 512
  MAX_OUTPUT_TOKENS = 100
  SEPARATOR_SPECIAL_TOKEN = " "
  OPTION_SEPARATOR = ")"
  K_FOLD = 5

  #TRAINING
  MODEL_CHECKPOINT = "it5/it5-base-question-answering"
  TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
  BATCH_SIZE = 4
  ACCUMULATION_STEP = 1
  EPOCHS = 20
  PATIENCE = 3
  METRIC_BEST = "loss"
  LEARNING_RATE = 5e-5
  FP16 = False
  WEIGHT_DECAY = 0.005
  SAVE_TOTAL_LIMIT=1

  #GENERATION
  MAX_LEN = 30
  CONSTRAINED = True
  MIN_LEN = 2