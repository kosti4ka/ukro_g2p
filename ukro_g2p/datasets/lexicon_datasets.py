from torch.utils.data import Dataset
from pathlib import Path
from ukro_g2p.utils.lexicon_util import read_lexicon_dataset
from ukro_g2p.utils.util import load_list
import torch
import numpy as np

# TODO put this into config
GRAPHEMES_PADDING = 37
PHONEMES_PADDING = 99


class LexiconDataset(Dataset):
    def __init__(self, data_root, split='train'):
        self.data_root = Path(data_root)
        self.lexicon_dataset_path = self.data_root / 'data.txt'
        if split in ['train', 'dev', 'test']:
            self.data_keys_path = self.data_root / f'{split}_set'
        else:
            raise KeyError
        #TODO take it from config file
        self.phonemes_path = self.data_root / 'phones'
        self.letters_path = self.data_root / 'letters'

        self.lexicon = read_lexicon_dataset(self.lexicon_dataset_path)
        self.data_keys = load_list(self.data_keys_path)
        self.phoneme2idx = {p.split()[1]: int(p.split()[0]) for p in load_list(self.phonemes_path)}
        self.phoneme2idx['pad'] = len(self.phoneme2idx)
        self.phoneme2idx['<os>'] = len(self.phoneme2idx)
        self.phoneme2idx['</os>'] = len(self.phoneme2idx)
        self.idx2phoneme = dict((v, k) for k, v in self.phoneme2idx.items())
        self.grapheme2idx = {p.split()[1]: int(p.split()[0]) for p in load_list(self.letters_path)}
        self.grapheme2idx['pad'] = len(self.grapheme2idx)
        self.grapheme2idx['<s>'] = len(self.grapheme2idx)
        self.grapheme2idx['</s>'] = len(self.grapheme2idx)
        self.idx2grapheme = dict((v, k) for k, v in self.grapheme2idx.items())

    def __getitem__(self, idx):
        key = self.data_keys[idx]
        datapoint = self.lexicon[key]

        graphemes = datapoint.split()[0]
        phonemes = datapoint.split()[1:]

        graphemes_idx = self.graphemes2idx(graphemes)
        phonemes_idx = self.phonemes2idx(phonemes)
        return graphemes_idx, phonemes_idx

    def __len__(self):
        return len(self.data_keys)

    def graphemes2idx(self, graphemes):
        # graphemes_idx = [self.grapheme2idx['<s>']]
        graphemes_idx = [self.grapheme2idx[g] for g in graphemes]
        return graphemes_idx

    def phonemes2idx(self, phonemes):
        phonemes_idx = [self.phoneme2idx['<os>']]
        phonemes_idx.extend([self.phoneme2idx[p] for p in phonemes])
        phonemes_idx.append(self.phoneme2idx['</os>'])
        return phonemes_idx

    def idx2graphemes(self, idx):
        return [self.idx2grapheme[g] for g in idx]

    def idx2phonemes(self, idx):
        return [self.idx2phoneme[p] for p in idx]


def _collate_fn(batch):
    """
    Merges list of samples to form a mini-batch.
    Pads input sequences to longest inputs sequence in the batch
    Pads all target sequences to longest sequence in mini-batch with constraint

    Args:
        batch:

    Returns:

    """

    graphemes_lengths = list(map(lambda x: len(x[0]), batch))
    phonemes_lengths = list(map(lambda x: len(x[1]), batch))
    max_word_length = max(graphemes_lengths)
    max_phonemes_length = max(phonemes_lengths)
    batch_size = len(batch)

    graphemes_inputs = torch.LongTensor(batch_size, max_word_length).zero_() + GRAPHEMES_PADDING
    phonemes_targets = torch.LongTensor(batch_size, max_phonemes_length).zero_() + PHONEMES_PADDING
    graphemes_length = torch.LongTensor(batch_size).zero_()
    phonemes_length = torch.LongTensor(batch_size).zero_()

    for x in range(batch_size):
        sample = batch[x]
        graphemes, phonemes = sample

        graphemes = torch.from_numpy(np.asarray(graphemes)).long()

        phonemes = torch.from_numpy(np.asarray(phonemes)).long()

        graphemes_seq_length = graphemes.size(0)
        phonemes_seq_length = phonemes.size(0)

        graphemes_inputs[x].narrow(0, 0, graphemes_seq_length).copy_(graphemes)
        graphemes_length[x] = graphemes_seq_length
        phonemes_length[x] = phonemes_seq_length

        phonemes_targets[x].narrow(0, 0, phonemes_seq_length).copy_(phonemes)

    return graphemes_inputs, graphemes_length, phonemes_targets, phonemes_length
