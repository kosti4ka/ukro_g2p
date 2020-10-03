import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ukro_g2p.datasets.lexicon_datasets import LexiconDataset, _collate_fn
from ukro_g2p.models.g2p_model import G2PConfig, G2PModel
from ukro_g2p.utils.util import phoneme_error_rate

from pathlib import Path
from ukro_g2p.utils.util import dump


def main(model, dataset, resume, out_dir):

    # init paths
    out_dir = Path(out_dir)
    ref_path = out_dir / 'ref'
    hyp_path = out_dir / 'hyp'
    scores_path = out_dir / 'scores'

    # make dir and files
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        ref_path.unlink()
        hyp_path.unlink()
        scores_path.unlink()
    except:
        pass

    # setup data_loader instances
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_fn, num_workers=1)

    # load model weights
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # init mertic values
    avr_per = 0
    total_phonemes_length = 0
    num_batches = len(data_loader)

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(data_loader), total=num_batches):
            graphemes, graphemes_length, phonemes, _ = batch

            graphemes = graphemes.to(device)
            graphemes_length = graphemes_length.to(device)
            phonemes = phonemes.to(device)

            phonemes_predictions = model(graphemes, graphemes_length).tolist()
            phonemes_targets = phonemes[:, 1:].contiguous().tolist()

            for predictions, targets, input_graphemes in zip(phonemes_predictions, phonemes_targets, graphemes):
                targets_length = len(targets)
                per = phoneme_error_rate(predictions, targets)
                avr_per += per * targets_length
                total_phonemes_length += targets_length

                # saving hyp, ref and scores
                graphemes_str = ''.join([g for g in dataset.idx2graphemes(input_graphemes.tolist()) if g != '<s>'])
                predictions_str = ' '.join([p for p in dataset.idx2phonemes(predictions) if p not in ['<os>', '</os>']])
                targets_str = ' '.join([p for p in dataset.idx2phonemes(targets) if p not in ['<os>', '</os>']])
                dump(f'{graphemes_str}\t{predictions_str}', hyp_path, append=True)
                dump(f'{graphemes_str}\t{targets_str}', ref_path, append=True)
                dump(f'{round(per, 4)}\t{targets_str}\t{predictions_str}', scores_path, append=True)

    avr_per /= total_phonemes_length
    log_str = f'Phoneme Error Rate: {round(avr_per, 4)}'
    dump(log_str, scores_path, append=True)
    print(log_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-d', '--dataset', default=None, type=str,
                        help='path to the dataset dir (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-o', '--out_dir', default=None, type=str,
                        help='path to the output dir (default: None)')

    args = parser.parse_args()

    config = G2PConfig(args.config)
    model = G2PModel(config)
    dataset = LexiconDataset(args.dataset, split='test')

    main(model, dataset, args.resume, args.out_dir)
