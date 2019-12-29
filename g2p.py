# -*- coding: utf-8 -*-
#!/usr/bin/env python3.6

import argparse

import torch
from torch.autograd import Variable

from models.g2p_model import G2PModel, PRETRAINED_MODEL_MAP
from utils.tokenization import G2PTokenizer


class G2P(object):
    def __init__(self, model_name):
        super().__init__()

        self.model = G2PModel.from_pretrained(model_name)
        self.tokenizer = G2PTokenizer.from_pretrained(model_name)

    def __call__(self, word, n_best=None):

        # map word's chars into graphemes idx
        graphemes = self.tokenizer.tokenize_graphemes(word)
        graphemes = self.tokenizer.convert_graphemes_to_ids(graphemes)
        g_length = [len(graphemes)]

        graphemes = Variable(torch.LongTensor(graphemes).unsqueeze(0))
        g_length = Variable(torch.LongTensor(g_length))

        phonemes = self.model(graphemes, g_length, n_best=n_best).tolist()[0]

        return [self.tokenizer.convert_ids_to_phonemes(p) for p in phonemes]


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument('word', type=str, help='Word to generate pronunciation for')
    parser.add_argument('-m', '--model_name', required=False, type=str, default='ukr-base-uncased',
                        choices=PRETRAINED_MODEL_MAP.keys(),
                        help='Trained model name')
    parser.add_argument('-nb', '--n_best', required=False, type=int, default=1,
                        help='Number of best pronunciations')

    # parse
    script_args = parser.parse_args()

    g2p = G2P('ukr-base-uncased')
    pron = g2p(script_args.word, n_best=script_args.n_best)

    [print(f"{' '.join(p)}") for p in pron]
