import config
import os
from itertools import permutations
import pickle
import os
import re
import spacy
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
# from .misc import save_as_pickle, load_pickle, get_subject_objects
from tqdm import tqdm
import logging

from torch.utils.tensorboard import SummaryWriter
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer

from dataset import PretainDataset
from loss import Two_Headed_Loss
import torch.optim as optim

from utils import save_pickle, load_pickle, get_logger, get_writer, load_model


logger = get_logger(__file__)
writer = get_writer()


def evaluate(lm_logits,  masked_for_pred):
    '''
    evaluate must be called after loss.backward()
    '''
    # lm_logits
    lm_logits_pred_ids = torch.softmax(lm_logits, dim=-1).max(1)[1]
    lm_accuracy = ((lm_logits_pred_ids == masked_for_pred).sum(
    ).float()/len(masked_for_pred)).item()

    return lm_accuracy


def main():
    model, tokenizer = load_model(
        model_type=config.MODEL_TYPE,
        n_classes=None,  # None for pretraining without classification head
        pretrained_path=None
    )

    # unfrozen_layers = ["classifier", "pooler", "encoder.layer.22", "encoder.layer.23", "blanks_linear", "lm_linear", "cls"]
    # for name, param in model.named_parameters():
    #     if not any([layer in name for layer in unfrozen_layers]):
    #         # logging.info("[FROZE]: %s" % name)
    #         param.requires_grad = False
    #     else:
    #         logging.info("[FREE]: %s" % name)
    #         param.requires_grad = True


    criterion = Two_Headed_Loss(lm_ignore_idx=tokenizer.pad_token_id,
                                use_logits=True, normalize=False)

    optimizer = optim.Adam([{"params": model.parameters(), "lr": config.LR}])

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[2, 4, 6, 8, 12, 15, 18, 20, 22, 24, 26, 30],
        gamma=0.8
    )

    data = load_pickle(r'/home/yifei/code/NLP/MY_EXTRACT/data/all.pkl')
    train_dataset = PretainDataset(data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, collate_fn=train_dataset.collate_fn)

    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    update_size = len(train_loader) // 10
    best_pred = 0.0

    step = 0
    for epoch in range(config.EPOCHS):
        model.train()
        loss_epoch, acc_epoch = [], []

        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            x, masked_for_pred, e1_e2_start, _, blank_labels = data

            masked_for_pred = masked_for_pred[(masked_for_pred != pad_id)]
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros(
                (x.shape[0], x.shape[1])).long().to(config.DEVICE)

            blanks_logits, lm_logits = model(
                x,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                e1_e2_start=e1_e2_start
            )

            lm_logits = lm_logits[(x == mask_id)]
            loss = criterion(lm_logits, blanks_logits,
                             masked_for_pred, blank_labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            acc = evaluate(lm_logits, masked_for_pred)

            step += 1
            if step % update_size == 0:
                logger.info(
                    f'[Epoch: {epoch + 1}, Step: {step}] Train Loss: {loss.item():.5f}, Train Accuracy: {acc:.5f}')
                writer.add_scalar('Loss/train-step', loss.item(), step)
                writer.add_scalar('Accuracy/train-step', acc, step)

                loss_epoch.append(loss.item())
                acc_epoch.append(acc)
            

        avg_epoch_loss = sum(loss_epoch)/len(loss_epoch)
        avg_epoch_accuracy = sum(acc_epoch)/len(acc_epoch)
        logger.info(
            f'Epoch {epoch + 1} - Train Loss: {avg_epoch_loss}, Train Accuracy: {avg_epoch_accuracy}')
        writer.add_scalar('Loss/train-epoch', avg_epoch_loss, epoch)
        writer.add_scalar('Accuracy/train-epoch', avg_epoch_accuracy, epoch)

        if avg_epoch_accuracy > best_pred:
            best_pred = avg_epoch_accuracy
            logging.info("Saving best model...")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': avg_epoch_accuracy,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(config.LOG_DIR, "pretrain_best.pth.tar"))
        scheduler.step()

    writer.close()


if __name__ == '__main__':
    main()
