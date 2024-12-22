import re
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

import config
from utils import get_logger, save_pickle, load_model


logger = get_logger(__file__)


class RelationsMapper(object):
    def __init__(self, relations):
        logger.info("Initializing RelationsMapper...")
        self.rel2idx = {}
        self.idx2rel = {}
        self.n_classes = 0
        for relation in relations:
            if relation not in self.rel2idx.keys():
                self.rel2idx[relation] = self.n_classes
                self.n_classes += 1

        for key, value in self.rel2idx.items():
            self.idx2rel[value] = key
        logger.info("RelationsMapper initialized with %d classes.", self.n_classes)


class PretrainDataProcessor:
    def __init__(
        self, pretrain_path, tokenizer, sample_size, mask_probability=0.15, alpha=0.7
    ):
        with open(pretrain_path, "r") as f:
            self.lines = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.mask_probability = mask_probability
        self.alpha = alpha

        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.E1s_token_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        self.E1e_token_id = self.tokenizer.convert_tokens_to_ids("[/E1]")
        self.E2s_token_id = self.tokenizer.convert_tokens_to_ids("[E2]")
        self.E2e_token_id = self.tokenizer.convert_tokens_to_ids("[/E2]")

    def put_blanks(self, D):
        blank_e1 = np.random.uniform()
        blank_e2 = np.random.uniform()
        if blank_e1 >= self.alpha:
            r, e1, e2 = D
            D = (r, "[BLANK]", e2)

        if blank_e2 >= self.alpha:
            r, e1, e2 = D
            D = (r, e1, "[BLANK]")
        return D

    def create_corpus(self):
        D = []
        for line in tqdm(self.lines):
            tokens = line["token"]
            e1_pos = (line["h"]["pos"][0], line["h"]["pos"][1])
            e2_pos = (line["t"]["pos"][0], line["t"]["pos"][1])
            e1_text = line["h"]["name"]
            e2_text = line["t"]["name"]
            D.append(((tokens, e1_pos, e2_pos), e1_text, e2_text))
        return D

    def tokenize(self, D):
        """Tokenize and mask the input data."""
        (x, s1, s2), e1, e2 = D
        if s1[0] > s2[0]:
            s1, s2 = s2, s1
            e1, e2 = e2, e1
        x = [w.lower() if isinstance(w, str) else w for w in x if w != "[BLANK]"]

        forbidden_idxs = [i for i in range(s1[0], s1[1])] + [
            i for i in range(s2[0], s2[1])
        ]
        pool_idxs = [i for i in range(len(x)) if i not in forbidden_idxs]
        masked_idxs = np.random.choice(
            pool_idxs, size=round(self.mask_probability * len(pool_idxs)), replace=False
        )
        masked_for_pred = [x[idx] for idx in masked_idxs]

        x = [
            token if (idx not in masked_idxs) else self.tokenizer.mask_token
            for idx, token in enumerate(x)
        ]
        x = (
            [self.cls_token]
            + x[: s1[0]]
            + ["[E1]"]
            + (x[s1[0] : s1[1]] if e1 != "[BLANK]" else ["[BLANK]"])
            + ["[/E1]"]
            + x[s1[1] : s2[0]]
            + ["[E2]"]
            + (x[s2[0] : s2[1]] if e2 != "[BLANK]" else ["[BLANK]"])
            + ["[/E2]"]
            + x[s2[1] :]
            + [self.sep_token]
        )
        x = self.tokenizer.convert_tokens_to_ids(x)

        assert len([idx for idx in x if idx == self.tokenizer.mask_token_id]) == len(
            masked_for_pred
        )
        e1_e2_start = (x.index(self.E1s_token_id), x.index(self.E2s_token_id))

        return x, self.tokenizer.convert_tokens_to_ids(masked_for_pred), e1_e2_start

    def create_dataset(self, D):
        """Preprocess the data into training samples."""
        logger.info("Creating dataset...")
        df = pd.DataFrame(D, columns=["r", "e1", "e2"])
        all_samples = []
        for idx in tqdm(range(len(df))):
            samples = []
            r, e1, e2 = df.iloc[idx]

            pos_pool = df[(df["e1"] == e1) & (df["e2"] == e2)].index
            neg_pool = df[(df["e1"] != e1) | (df["e2"] != e2)].index

            pos_idxs = [idx] + list(
                np.random.choice(
                    pos_pool,
                    min(self.sample_size // 2 - 1, len(pos_pool)),
                    replace=True,
                )
            )
            neg_idxs = list(
                np.random.choice(
                    neg_pool, self.sample_size - len(pos_idxs), replace=True
                )
            )

            for pos_idx in pos_idxs:
                samples.append(
                    (*self.tokenize(self.put_blanks(D[pos_idx])), [1.0], [1])
                )
            for neg_idx in neg_idxs:
                samples.append(
                    (
                        *self.tokenize(self.put_blanks(D[neg_idx])),
                        [1.0 / len(neg_pool)],
                        [0],
                    )
                )

            all_samples.append(samples)
        logger.info("Dataset created with %d samples.", len(all_samples))
        return all_samples

    def preprocess(self):
        D = self.create_corpus()
        samples = self.create_dataset(D)
        return samples


class FinetuneDataProcessor:
    def __init__(self, train_path, test_path):
        # Initialize member variables with path information
        self.train_path = train_path
        self.test_path = test_path

    def process_text(self, text, mode="train"):
        """Process text data to extract sentences, relations, comments, and blanks"""
        logger.info("Processing text in %s mode...", mode)
        sents, relations, comments, blanks = [], [], [], []
        num_entries = len(text) // 4  # Each entry contains 4 lines

        for i in range(num_entries):
            # Extract information for the current entry
            sent = text[4 * i].strip()
            relation = text[4 * i + 1].strip()
            comment = text[4 * i + 2].strip()
            blank = text[4 * i + 3].strip()

            # Validate entry ID
            entry_id = int(re.match(r"^\d+", sent)[0])
            expected_id = i + 1 if mode == "train" else i + 1 + 8000
            assert (
                entry_id == expected_id
            ), f"ID mismatch: expected {expected_id}, got {entry_id}"

            # Extract content within quotes and replace entity tags with uppercase
            sent = re.findall(r"\"(.+)\"", sent)[0]
            sent = re.sub(r"<e1>", "[E1]", sent)
            sent = re.sub(r"</e1>", "[/E1]", sent)
            sent = re.sub(r"<e2>", "[E2]", sent)
            sent = re.sub(r"</e2>", "[/E2]", sent)

            sents.append(sent)
            relations.append(relation)
            comments.append(comment)
            blanks.append(blank)

        logger.info("Processed %d entries.", num_entries)
        return sents, relations, comments, blanks

    def preprocess_finetune_data(self):
        """Load and process training and test data"""
        # Load and process training data
        logger.info("Reading training file: %s", self.train_path)
        with open(self.train_path, "r", encoding="utf8") as f:
            train_text = f.readlines()
        sents, relations, _, _ = self.process_text(train_text, mode="train")
        df_train = pd.DataFrame({"sents": sents, "relations": relations})

        # Load and process test data
        logger.info("Reading test file: %s", self.test_path)
        with open(self.test_path, "r", encoding="utf8") as f:
            test_text = f.readlines()
        sents, relations, comments, blanks = self.process_text(test_text, mode="test")
        df_test = pd.DataFrame({"sents": sents, "relations": relations})

        # Map relation categories to IDs
        logger.info("Mapping relation categories to IDs...")
        rm = RelationsMapper(df_train["relations"])
        df_train["relations_id"] = df_train["relations"].map(rm.rel2idx)
        df_test["relations_id"] = df_test["relations"].map(rm.rel2idx)

        return df_train, df_test, rm


if __name__ == "__main__":
    ## Preprocess data
    logger.info("Preprocessing Pretrain data...")
    _, tokenizer = load_model(
        model_type=config.MODEL_TYPE, n_classes=config.N_CLASSES, pretrained_path=None
    )
    pretrain_data = PretrainDataProcessor(
        config.PRETRAIN_ORI_PATH, tokenizer, config.SAMPLE_SIZE
    ).preprocess()
    save_pickle(config.PRETRAIN_DATA_PATH, pretrain_data)
    logger.info("Pretrain data saved.")

    ## Finetune data
    logger.info("Preprocessing Finetune data...")
    train_data, test_data, rm = FinetuneDataProcessor(
        config.TRAIN_ORI_PATH,
        config.TEST_ORI_PATH,
    ).preprocess_finetune_data()
    save_pickle(config.TRAIN_DATA_PATH, train_data)
    save_pickle(config.TEST_DATA_PATH, test_data)
    save_pickle(config.RELATIONS_PATH, rm)
    logger.info("Preprocessing completed and data saved!")
