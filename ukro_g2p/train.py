import argparse

from ukro_g2p.datasets.lexicon_datasets import LexiconDataset
from ukro_g2p.trainer import Trainer
from torch.optim import Adam
import torch.nn as nn

from ukro_g2p.models.g2p_model import G2PConfig, G2PModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-d', '--dataset', default=None, type=str,
                        help='path to the dataset dir (default: None)')
    parser.add_argument('-n', '--experiment_name', default=None, type=str,
                        help='path to the output dir (default: None)')
    parser.add_argument('-r', '--restore_epoch', default=-1, type=int,
                        help='restore epoch number (default: -1)')

    args = parser.parse_args()

    config = G2PConfig(args.config)
    model = G2PModel(config)
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss = nn.NLLLoss(ignore_index=config.decoder_padding_idx)
    datasets = {'train': LexiconDataset(args.dataset, split='train'),
                  'dev': LexiconDataset(args.dataset, split='dev')}

    trainer = Trainer(model, datasets, optimizer, loss, epochs=100, batch_size=256,
                      experiment_name=args.experiment_name,
                      logging_freq=10, restore_epoch=args.restore_epoch)
    trainer.train_and_validate()
