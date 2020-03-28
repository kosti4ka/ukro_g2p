import torch.nn as nn
from torch.optim import Adagrad, SGD
import torch
# from .decoder import Decoder
# from .encoder import Encoder
# from .converter import Converter
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from models.modules import Encoder, Decoder
from models.modules import Beam
from pathlib import Path
import configparser


PRETRAINED_MODEL_MAP = {
    'ukr-base-uncased': "../trained_models/g2p_ukr"
}
CONFIG_SUFFIX = '.config'
WEIGHTS_SUFFIX = '.th'


class G2PConfig(dict):

    def __init__(self, model_config_file):
        super(G2PConfig, self).__init__()

        self.use_cuda = False
        self.model_path = None

        # reading config file
        config_file = configparser.ConfigParser()
        config_file.read(model_config_file)

        self.padding = config_file['VocabConfig']['padding']
        # TODO simplify this part - use same bos, eos symbol for both phonemes and graphemes
        self.decoder_bos = config_file['VocabConfig']['phoneme_bos']
        self.decoder_eos = config_file['VocabConfig']['phoneme_eos']
        # reading graphemes
        self.graphemes = config_file['VocabConfig']['graphemes'].split()
        self.encoder_vocab_size = len(self.graphemes)
        self.encoder_padding_idx = self.graphemes.index(self.padding)
        # reading phonemes
        self.phonemes = config_file['VocabConfig']['phonemes'].split()
        self.decoder_vocab_size = len(self.phonemes)
        self.decoder_padding_idx = self.phonemes.index(self.padding)
        self.decoder_bos_idx = self.phonemes.index(self.decoder_bos)
        self.decoder_eos_idx = self.phonemes.index(self.decoder_eos)

        # encoder config
        self.encoder_d_embed = int(config_file['EncoderConfig']['encoder_d_embed'])
        self.encoder_d_hidden = int(config_file['EncoderConfig']['encoder_d_hidden'])
        self.encoder_n_layers = int(config_file['EncoderConfig']['encoder_n_layers'])
        self.encoder_bidirectional = True if config_file['EncoderConfig']['encoder_bidirectional'].lower() == 'true' else False

        # decoder config
        self.decoder_d_embed = int(config_file['DecoderConfig']['decoder_d_embed'])
        self.decoder_d_hidden = int(config_file['DecoderConfig']['decoder_d_hidden'])
        self.decoder_n_layers = int(config_file['DecoderConfig']['decoder_n_layers'])
        self.attention = True if config_file['DecoderConfig']['attention'] == 'True' else False

        # generator
        self.beam_size = int(config_file['GeneratorConfig']['beam_size'])
        self.max_generate_len = int(config_file['GeneratorConfig']['max_generate_len'])

        # optimizer
        self.lr = float(config_file['OptimizerConfig']['lr'])
        self.weight_decay = float(config_file['OptimizerConfig']['weight_decay'])


class PreTrainedG2PModel(nn.Module):
    def __init__(self, config):
        super(PreTrainedG2PModel, self).__init__()
        self.config = config

    @classmethod
    def from_pretrained(cls, pretrained_model_name):

        if pretrained_model_name in PRETRAINED_MODEL_MAP:
            base_path = Path(__file__).parent / PRETRAINED_MODEL_MAP[pretrained_model_name]
            config_file = base_path.with_suffix(CONFIG_SUFFIX)
            model_weigths_file = base_path.with_suffix(WEIGHTS_SUFFIX)
        else:
            raise ValueError

        # load config
        config = G2PConfig(config_file)  # TODO add metod from_file

        # instantiate model
        model = cls(config)

        # loading weigths
        model.load_state_dict(torch.load(model_weigths_file, map_location=lambda storage, loc: storage))

        return model


class G2PModel(PreTrainedG2PModel):

    def __init__(self, config):
        super(G2PModel, self).__init__(config)

        # init
        self.config = config
        self.use_cuda = config.use_cuda

        # encoder
        self.encoder = Encoder(config)

        # decoder
        self.decoder = Decoder(config)
        self.attention = config.attention

        # generator
        self.beam_size = config.beam_size
        self.max_generate_len = config.max_generate_len

        # optimizer
        self.lr = config.lr
        self.weight_decay = config.weight_decay

        # init optimizer
        self.add_optimizer_and_losses()

        self.maybe_move_to_cuda()

    def forward(self, x, x_length, y=None, p_length=None, n_best=1):
        # TODO rewrite desscription

        encoder_out, encoder_hc = self.encoder(x, x_length)

        if y is not None:
            out = self.decoder(y, p_length, encoder_hc, context=encoder_out if self.attention else None)
        else:
            out = self._generate(encoder_hc, context=encoder_out if self.attention else None, n_best=n_best)

        return out

    def _generate(self, hc, context=None, n_best=1):
        beam = Beam(self.config)
        h, c = hc
        # Make a beam_size batch.
        h = h.expand(h.size(1), beam.size, h.size(2))
        c = c.expand(c.size(1), beam.size, c.size(2))
        if context is not None:
            context = context.expand(beam.size, context.size(1), context.size(2))
        p_length = Variable(torch.from_numpy(np.array([1])))
        p_length = p_length.expand(beam.size)
        if self.use_cuda:
            p_length = p_length.cuda()

        for i in range(self.max_generate_len):
            x = beam.get_current_state()
            o, hc = self.decoder(Variable(x.unsqueeze(1)), p_length, (h, c), context=context)
            if beam.advance(o.data.squeeze(1)):
                break
            h, c = hc
            h.data.copy_(h.data.index_select(1, beam.get_current_origin()))
            c.data.copy_(c.data.index_select(1, beam.get_current_origin()))
        return torch.LongTensor(beam.get_hyp(0)).unsqueeze(0)

    def add_optimizer_and_losses(self):
        self.optimizer = SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss = nn.NLLLoss(ignore_index=102)

    def maybe_move_to_cuda(self):
        if self.use_cuda:
            self.cuda()
            self.loss = self.loss.cuda()
