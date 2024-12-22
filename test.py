import os
import subprocess

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import FintuneDataset
from preprocess import RelationsMapper
from utils import load_pickle, load_model


def test(model, tokenizer, test_loader, pred_path, gold_path, output=False):
    """Evaluate the model on the test set and return the F1 score"""
    model.eval()
    pad_id = tokenizer.pad_token_id
    all_preds, all_golds = [], []
    relation_mapper = load_pickle(config.RELATIONS_PATH)

    with torch.no_grad():
        for batch_idx, batch_data in tqdm(
            enumerate(test_loader), total=len(test_loader), desc="Evaluating"
        ):
            inputs, e1_e2_start, labels = batch_data
            attention_mask = (inputs != pad_id).float()
            token_type_ids = torch.zeros((inputs.shape[0], inputs.shape[1])).long()

            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            attention_mask, token_type_ids = attention_mask.to(
                config.DEVICE
            ), token_type_ids.to(config.DEVICE)

            logits = model(
                inputs,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                e1_e2_start=e1_e2_start,
            )

            # Compute inference results
            valid_idxs = (
                labels != -1
            ).squeeze()  # Find valid labels (non-ignore labels)
            predicted_labels = torch.softmax(logits, dim=1).max(1)[1]
            valid_labels = labels.squeeze()[valid_idxs]
            valid_preds = predicted_labels[valid_idxs]

            # Convert results to list form
            valid_labels = (
                valid_labels.cpu().numpy().tolist()
                if valid_labels.is_cuda
                else valid_labels.numpy().tolist()
            )
            valid_preds = (
                valid_preds.cpu().numpy().tolist()
                if valid_preds.is_cuda
                else valid_preds.numpy().tolist()
            )

            all_preds.extend(valid_preds)
            all_golds.extend(valid_labels)

    # Write prediction results to file
    with open(pred_path, "w") as pred_file, open(gold_path, "w") as gold_file:
        for i, (pred, gold) in enumerate(zip(all_preds, all_golds)):
            pred_file.write(str(i) + "\t" + relation_mapper.idx2rel[pred] + "\n")
            gold_file.write(str(i) + "\t" + relation_mapper.idx2rel[gold] + "\n")

    # Calculate F1 score
    f1_score = calculate_f1_score(pred_path, gold_path, output=output)

    return f1_score


def calculate_f1_score(pred_path, gold_path, output=False):
    """Calculate F1 score using a Perl script"""
    process = subprocess.Popen(
        ["perl", config.SCORER_PATH, pred_path, gold_path], stdout=subprocess.PIPE
    )
    str_content = process.communicate()[0]

    if output:
        with open(os.path.dirname(pred_path) + "/results.txt", "w") as f:
            f.write(str_content.decode("utf-8"))
    str_parse = str(str_content).split("\\n")[-2]
    idx = str_parse.find("%")
    f1_score = float(str_parse[idx - 5 : idx])

    return f1_score


if __name__ == "__main__":
    model, tokenizer = load_model(
        model_type=config.MODEL_TYPE,
        n_classes=config.N_CLASSES,
        pretrained_path=config.CHECKPOINT_PATH,
    )
    model.eval()
    e1_id = tokenizer.convert_tokens_to_ids("[E1]")
    e2_id = tokenizer.convert_tokens_to_ids("[E2]")

    df_test = load_pickle(config.TEST_DATA_PATH)
    test_set = FintuneDataset(df_test, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=test_set.collate_fn,
    )

    import os
    pred_path = os.path.dirname(config.CHECKPOINT_PATH) + "/preds.txt"
    gold_path = os.path.dirname(config.CHECKPOINT_PATH) + "/golds.txt"
    f1 = test(model, tokenizer, test_loader, pred_path, gold_path, output=True)
    print(f"F1 Score: {f1}")
