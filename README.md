# BERTem (BERT for Relation Extraction)

Personal repository for reproducing and extending the ideas presented in [Matching the Blanks: Distributional Similarity for Relation Learning](https://arxiv.org/abs/1906.03158). This project explores the use of BERT and its variants for relation extraction tasks, including pretraining and fine-tuning on datasets such as SemEval-2010 Task 8.

## Directory Structure
```
.
├── config.py                # Configuration file
├── data/                    # Data directory
│   ├── finetune/            # Finetune data directory
│   │   ├── relations.pkl    # Relations file
│   │   ├── SemEval2010_task8_all_data/  # SemEval2010 Task8 dataset
│   │   ├── test.pkl         # Preprocessed test set
│   │   └── train.pkl        # Preprocessed training set
│   └── pretrain/            # Pretrain data directory
│       ├── pretrain.pkl     # Preprocessed pretrain dataset
│       └── wiki80/          # Wiki80 dataset
├── dataset.py               # Dataset processing script
├── finetune.py              # Finetune task implementation script
├── loss.py                  # Loss function definition file
├── preprocess.py            # Data preprocessing script
├── pretrain.py              # Pretrain task implementation script
├── runs/                    # Training logs and model checkpoints
├── test.py                  # Testing script (model evaluation)
├── transformers/            # BERT, ALBERT model files
└── utils.py                 # Utility functions library
```

## Data Preparation

### Pretrain Data
1.	Dataset: Use the Wiki80 dataset for pretraining the model. Place the raw dataset under `data/pretrain/wiki80/`. Download the dataset from [here](https://github.com/thunlp/OpenNRE/blob/master/benchmark/download_wiki80.sh).
2.	Preprocessing: Use th `preprocess.py` script to preprocess the dataset into pretrain.pkl.

### Finetune Data
1. Dataset: Download the SemEval-2010 Task 8 dataset and place it in `data/finetune/SemEval2010_task8_all_data/`.
2.	Preprocessing: Convert the raw dataset into preprocessed `.pkl` files using `preprocess.py`.

## Configuration
The `config.py` file contains the configuration details for training, evaluation, and data paths. Update this file to customize hyperparameters such as learning rate, batch size, and training epochs, and ensure the data paths are correctly set to the appropriate directories.

## Pretrain
The `pretrain.py` script handles the pretraining process. Before running, ensure that the preprocessed data file `pretrain.pkl` exists in `data/pretrain/`. Run the script as follows:

``` bash
python pretrain.py
``` 

## Finetune
The `finetune.py` script fine-tunes the model on the SemEval-2010 Task 8 dataset. Make sure the preprocessed training and test files (`train.pkl`, `test.pkl`) are available in `data/finetune/`. Run the script as follows:

``` bash
python finetune.py
``` 

## Test
Use the `test.py` script to evaluate the fine-tuned model on the test dataset. Modify  the `checkpoint_path` to the fine-tuned model checkpoint. Run the script as follows:

``` bash
python test.py
```

## Additional Notes
- Model Variants: This repository supports both BERT and ALBERT architectures, with configuration options available in `config.py`.
- Logging and Checkpoints: All training logs, metrics, and checkpoints are saved in the `runs/` directory for reproducibility and analysis.