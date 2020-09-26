import time
from collections import OrderedDict

from pathlib import Path

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ukro_g2p.predict import _collate_fn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from text import text_to_sequence
from ukro_g2p.predict import infolog
from tensorboardX import SummaryWriter
from tqdm import tqdm

log = infolog.log


class Trainer(object):
    def __init__(self, model, datasets, optimizer, loss, epochs, batch_size, experiment_name, logging_freq,
                 restore_epoch=-1):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.experiment_name = experiment_name
        self.logging_freq = logging_freq
        self.model = model.to(self.device)
        self.restore_epoch = restore_epoch

        self.optimizer = optimizer
        self.loss = loss
        # TODO remove scheduler from trainer object
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        self.global_batch_index = 0
        self.global_batch_index_dev = 0
        self.datasets = datasets
        self.dataloaders = OrderedDict()
        for data_name in datasets:
            self.dataloaders[data_name] = DataLoader(datasets[data_name], batch_size=batch_size,
                                                     shuffle=True if data_name == 'train' else False,
                                                     collate_fn=_collate_fn,
                                                     num_workers=8)


        self.experiment_dir = Path('./exp') / self.experiment_name
        self.log_dir = self.experiment_dir / 'logs'
        self.log_path = self.log_dir / 'log.txt'
        self.checkpoint_dir = self.experiment_dir / 'checkpoints'

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # initialize logger
        infolog.init(self.log_path, self.experiment_name)
        # initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        if self.restore_epoch != -1:
            self.load_from_checkpoint(epoch=self.restore_epoch)

    def train_and_validate(self):
        for e in range(self.restore_epoch + 1, self.epochs):
            for ds, dataloader in self.dataloaders.items():
                if ds != 'train':
                    self.model.eval()  # turn off batchnorm /dropout
                self.run_epoch(dataloader, dataset_name=ds, epoch=e)
                self.model.train()  # turn on batchnorm /dropout
            self.save_to_checkpoint(epoch=e)

    def run_epoch(self, dataloader, dataset_name, epoch):
        begin_time = time.time()
        train = dataset_name == 'train'
        num_batches = len(dataloader)
        avg_loss = 0

        progress_bar = tqdm(total=num_batches)
        for batch_idx, batch in enumerate(dataloader):
            word_inputs, input_length, pron_targets, p_length = batch

            word_inputs = Variable(word_inputs).to(self.device)
            input_length = Variable(input_length).to(self.device)
            pron_targets = Variable(pron_targets).to(self.device)
            p_length = Variable(p_length).to(self.device)

            p_preds, _ = self.model(word_inputs, input_length, pron_targets[:, :-1], p_length - 1)
            targets = pron_targets[:, 1:].contiguous()

            loss = self.loss(p_preds.view(p_preds.size(0) * p_preds.size(1), -1),
                             targets.view(targets.size(0) * targets.size(1)))

            loss_floats = loss.data.cpu().item()
            avg_loss += loss_floats

            progress_bar.set_description(f'{dataset_name}: {epoch}/{self.epochs}, current loss: {round(loss_floats,4)}')
            progress_bar.refresh()
            progress_bar.update()

            if train:
                self.global_batch_index += 1
                self.writer.add_scalar(dataset_name, loss_floats, self.global_batch_index)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # if (batch_idx + 1) % self.logging_freq == 0:
                #     log_str = self.construct_logging_str(loss_floats, epoch, num_batches, batch_idx + 1)
                #     log(log_str)
            else:
                # TODO it is ugly part of code, shuold add global step explicitly
                self.global_batch_index_dev += 1
                self.writer.add_scalar(dataset_name, loss_floats,  self.global_batch_index_dev)

        progress_bar.close()

        # add generated text to tensor board
        if not train:
            graphemes, graphemes_length, phonemes, _ = next(iter(dataloader))

            graphemes = graphemes.to(self.device)
            graphemes_length = graphemes_length.to(self.device)
            phonemes = phonemes.to(self.device)

            # phonemes_predictions = self.model(graphemes, graphemes_length).tolist()
            phonemes_targets = phonemes[:, 1:].contiguous().tolist()

            text = ''
            for idx in range(0, 5):
                phonemes_predictions = self.model(graphemes[idx].unsqueeze(0), graphemes_length[idx].unsqueeze(0)).tolist()[0]

                graphemes_str = ''.join(
                    [g for g in self.datasets[dataset_name].idx2graphemes(graphemes[idx].tolist()) if g not in ['<s>', 'pad']])
                predictions_str = ' '.join(
                    [p for p in self.datasets[dataset_name].idx2phonemes(phonemes_predictions) if p not in ['<os>', '</os>']])
                targets_str = ' '.join(
                    [p for p in self.datasets[dataset_name].idx2phonemes(phonemes_targets[idx]) if p not in ['<os>', '</os>', 'pad']])

                text = text + graphemes_str + '  \n' + targets_str + '  \n' + predictions_str + '  \n'

            self.writer.add_text('Text', text, epoch)

        avg_loss /= num_batches
        end_time = time.time()
        if train:
            self.scheduler.step(avg_loss)
        log_str = 'Epoch: {}, {} loss: {:.5f} time: {:.2f} sec'
        log_str = log_str.format(epoch, dataset_name, avg_loss, end_time - begin_time)
        log(log_str)

    def load_from_checkpoint(self, epoch):
        checkpoint_file = f'{self.experiment_name}-epoch-{epoch}.th'
        checkpoint_path = self.checkpoint_dir / checkpoint_file
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        print('model loaded from checkpoint: {}'.format(checkpoint_file))

    def save_to_checkpoint(self, epoch):
        checkpoint_file = f'{self.experiment_name}-epoch-{epoch}.th'
        checkpoint_path = self.checkpoint_dir / checkpoint_file
        if self.checkpoint_dir.exists():
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f'model saved to checkpoint: {checkpoint_path}')
        else:
            raise FileNotFoundError

    @staticmethod
    def construct_logging_str(loss, epoch, total_batches, idx):
        tmpstr = 'Epoch:{:2} Batch:[{:3}/{:3}] Loss: {:.4f}'
        tmpstr = tmpstr.format(epoch, idx, total_batches, loss)
        return tmpstr
