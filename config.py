import os
import torch
import datetime

# === Directories and Paths ===
DATE_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_DIR = f'./runs/{DATE_TIME}'
DATA_DIR = './data'

# Log and Model Paths
LOG_PATH = os.path.join(LOG_DIR, 'train.log')
MODEL_PATH = os.path.join(LOG_DIR, 'model.pth')

# Pretraining Data Paths
PRETRAIN_ORI_PATH = os.path.join(DATA_DIR, 'pretrain', 'wiki80/wiki80_train.txt')
PRETRAIN_DATA_PATH = os.path.join(DATA_DIR, 'pretrain', 'pretrain.pkl')

# Finetuning Data Paths
FINETUNE_DIR = os.path.join(DATA_DIR, 'finetune')
TRAIN_ORI_PATH = os.path.join(
    FINETUNE_DIR, 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
)
TEST_ORI_PATH = os.path.join(
    FINETUNE_DIR, 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
)
TRAIN_DATA_PATH = os.path.join(FINETUNE_DIR, 'train.pkl')
TEST_DATA_PATH = os.path.join(FINETUNE_DIR, 'test.pkl')
RELATIONS_PATH = os.path.join(FINETUNE_DIR, 'relations.pkl')

# Evaluation Paths
PREDICT_PATH = os.path.join(LOG_DIR, 'preds.txt')
GOLD_PATH = os.path.join(LOG_DIR, 'golds.txt')
SCORER_PATH = os.path.join(
    FINETUNE_DIR, 'SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2.pl'
)

# Checkpoint Path
CHECKPOINT_PATH = r'/home/yifei/code/NLP/BERT-EM/runs/bert-base-uncased/finetune_best.pth.tar'  # Use None for original pretrained model
# CHECKPOINT_PATH = None

# === Training Parameters ===
BATCH_SIZE = 128
SAMPLE_SIZE = 8
LR = 3e-4
EPOCHS = 25

# === Model Parameters ===
MODEL_TYPE = 'bert-base-uncased'  # Support for bert and albert models: bert-base-uncased, albert-xxlarge-v2, etc.
N_CLASSES = 19  # Number of classes for SemEval2010_task8

# === Device Configuration ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
