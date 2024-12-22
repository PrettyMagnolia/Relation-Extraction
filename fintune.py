import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

import config
from utils import load_pickle, get_logger, get_writer, load_model, save_pickle
from preprocess import FinetuneDataProcessor, RelationsMapper
from dataset import FintuneDataset
from test import test

logger = get_logger(__file__)
writer = get_writer()


def load_finetune_data():
    if os.path.isfile(config.TRAIN_DATA_PATH) and os.path.isfile(config.TEST_DATA_PATH) and os.path.isfile(config.RELATIONS_PATH):
        logger.info("Found preprocessed finetune data, loading from files...")
        df_train = load_pickle(config.TRAIN_DATA_PATH)
        df_test = load_pickle(config.TEST_DATA_PATH)
        rm = load_pickle(config.RELATIONS_PATH)
    else:
        train_data, test_data, rm = FinetuneDataProcessor(
            config.TRAIN_ORI_PATH,
            config.TEST_ORI_PATH,
        ).preprocess_finetune_data()
        save_pickle(config.TRAIN_DATA_PATH, train_data)
        save_pickle(config.TEST_DATA_PATH, test_data)
        save_pickle(config.RELATIONS_PATH, rm)
    return df_train, df_test, rm


def evaluate(output, labels, ignore_idx):
    # ignore index 0 (padding) when calculating accuracy
    idxs = (labels != ignore_idx).squeeze()
    o_labels = torch.softmax(output, dim=1).max(1)[1]
    l = labels.squeeze()[idxs]
    o = o_labels[idxs]
    acc = (l == o).sum().item()/len(idxs)
    return acc


def main():
    model, tokenizer = load_model(
        model_type=config.MODEL_TYPE, n_classes=config.N_CLASSES, pretrained_path=config.CHECKPOINT_PATH
    )

    # unfrozen_layers = [ "cls", "classifier", "pooler", "encoder.layer.22", "encoder.layer.23","encoder.layer.21", "encoder.layer.20","encoder.layer.19", "encoder.layer.18",]
    # for name, param in model.named_parameters():
    #     if not any([layer in name for layer in unfrozen_layers]):
    #         logging.info("[FROZE]: %s" % name)
    #         param.requires_grad = False
    #     else:
    #         logging.info("[FREE]: %s" % name)
    #         param.requires_grad = True

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam([{"params": model.parameters(), "lr": config.LR}])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[2, 4, 6, 8, 12, 15, 18, 20, 22, 24, 26, 30],
        gamma=0.8
    )

    best_pred = 0.0

    logger.info("Starting training process...")

    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    pad_id = tokenizer.pad_token_id

    df_train, df_test, rm = load_finetune_data()

    train_set = FintuneDataset(
        df_train, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
    test_set = FintuneDataset(
        df_test, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE,
                              shuffle=True, collate_fn=train_set.collate_fn)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE,
                             shuffle=False, collate_fn=test_set.collate_fn)

    update_size = len(train_loader) // 10

    step = 0  # Initialize step counter

    for epoch in range(config.EPOCHS):
        model.train()
        loss_epoch, acc_epoch = [], []

        for data in tqdm(train_loader):
            x, e1_e2_start, labels = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long().to(config.DEVICE)

            classification_logits = model(
                x,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                e1_e2_start=e1_e2_start
            )

            loss = criterion(classification_logits, labels.squeeze(1))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Calculate accuracy
            acc = evaluate(classification_logits, labels, ignore_idx=-1)
            
            if step % update_size == 0:
                logger.info(f'[Epoch: {epoch + 1}, Step: {step}] Train Loss: {loss.item():.5f}, Train Accuracy: {acc:.5f}')
                writer.add_scalar('Loss/train-step', loss.item(), step)
                writer.add_scalar('Accuracy/train-step', acc, step)

                loss_epoch.append(loss.item())
                acc_epoch.append(acc)
            
            step += 1
            
        # Calculate test performance at the end of each epoch
        avg_epoch_loss = sum(loss_epoch) / len(loss_epoch)
        avg_epoch_accuracy = sum(acc_epoch) / len(acc_epoch)
        logger.info(f'Epoch {epoch + 1} - Train Loss: {avg_epoch_loss}, Train Accuracy: {avg_epoch_accuracy}')
        writer.add_scalar('Loss/train-epoch', avg_epoch_loss, epoch)
        writer.add_scalar('Accuracy/train-epoch', avg_epoch_accuracy, epoch)

        test_f1 = test(model, tokenizer, test_loader, config.PREDICT_PATH, config.GOLD_PATH)
        logger.info(f'Epoch {epoch + 1} - Test F1: {test_f1}')
        writer.add_scalar('F1/test-epoch', test_f1, epoch)

        # Save the best model
        if test_f1 > best_pred:
            best_pred = test_f1
            logger.info("Saving best model...")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': avg_epoch_accuracy,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(config.LOG_DIR, "finetune_best.pth.tar"))

        scheduler.step()

    writer.close()


if __name__ == '__main__':
    main()
