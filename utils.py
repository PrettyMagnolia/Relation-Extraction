import datetime
import os
import pickle
import config
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.albert.modeling_albert import AlbertModel
from transformers.models.albert.tokenization_albert import AlbertTokenizer


def get_logger(module_name):
    os.makedirs(config.LOG_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,  # Set default log level to INFO
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            # Write logs to file
            logging.FileHandler(config.LOG_PATH, mode="w"),
            logging.StreamHandler(),  # Output logs to console
        ],
    )
    return logging.getLogger(module_name)


def get_writer():
    return SummaryWriter(log_dir=config.LOG_DIR)


def load_model(model_type, n_classes=None, pretrained_path=None):
    if "albert" in model_type:
        model = AlbertModel.from_pretrained(
            model_type, force_download=False, n_classes=n_classes
        )
        tokenizer = AlbertTokenizer.from_pretrained(
            config.MODEL_TYPE, do_lower_case=False
        )
    else:
        model = BertModel.from_pretrained(
            model_type, force_download=False, n_classes=n_classes
        )
        tokenizer = BertTokenizer.from_pretrained(
            config.MODEL_TYPE, do_lower_case=False
        )

    tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]", "[BLANK]"])
    model.resize_token_embeddings(len(tokenizer))

    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        # model_dict = model.state_dict()
        # pretrained_dict = {
        #     k: v for k, v in checkpoint["state_dict"].items() if k in model_dict.keys()
        # }
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(pretrained_dict, strict=False)

    return model.to(config.DEVICE), tokenizer


def load_pickle(file_path):
    with open(file_path, "rb") as pkl_file:
        data = pickle.load(pkl_file)
    return data


def save_pickle(file_path, data):
    with open(file_path, "wb") as pkl_file:
        pickle.dump(data, pkl_file)
