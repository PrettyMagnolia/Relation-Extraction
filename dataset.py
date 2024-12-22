import torch
from torch.utils.data import Dataset
from utils import get_logger
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import config

tqdm.pandas()
# logger = get_logger(__file__)


class PretainDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        samples = self.data[idx]
        return [list(map(lambda x: torch.LongTensor(x).to(config.DEVICE), sample)) for sample in samples]

    def collate_fn(self, batch):
        flatten_batch = [item for sublist in batch for item in sublist]
        batch = list(zip(*flatten_batch))
        return [pad_sequence(b, batch_first=True, padding_value=self.tokenizer.pad_token_id) for b in batch]


class FintuneDataset(Dataset):
    def __init__(self, df, tokenizer, e1_id, e2_id):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.tokenizer = tokenizer
        self.df = df
        # logger.info("Tokenizing and encoding sentences...")
        self.df['input'] = self.df['sents'].progress_apply(tokenizer.encode)
        self.df['input'] = self.df.progress_apply(
            lambda x: tokenizer.encode(x['sents']), axis=1)

        self.df['e1_e2_start'] = self.df['input'].progress_apply(
            lambda x: self.get_e1e2_start(x, self.e1_id, self.e2_id)
        )
        self.df.dropna(axis=0, inplace=True)

    def get_e1e2_start(self, x, e1_id, e2_id):
        try:
            e1_start = next(i for i, token in enumerate(x) if token == e1_id)
            e2_start = next(i for i, token in enumerate(x) if token == e2_id)
            return (e1_start, e2_start)
        except StopIteration:
            return None

    def __len__(self,):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        return torch.LongTensor(data['input']).to(config.DEVICE), \
            torch.LongTensor(data['e1_e2_start']).to(config.DEVICE), \
            torch.LongTensor([data['relations_id']]).to(config.DEVICE)

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        return [pad_sequence(b, batch_first=True, padding_value=self.tokenizer.pad_token_id) for b in batch]
