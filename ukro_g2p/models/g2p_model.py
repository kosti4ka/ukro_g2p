import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
from ukro_g2p.models.modules import Encoder, Decoder
from ukro_g2p.models.modules import Beam
from pathlib import Path
import configparser
from collections import namedtuple
from torch.utils import model_zoo

model_obj = namedtuple("model_obj", ["url", "config_path"])

pretrained_models = {
    "ukro-base-uncased": model_obj(
        url="https://github.com/kosti4ka/ukro_g2p/releases/download/ukro_base_uncased_v.0.1/ukro_base_uncased-epoch-99-d545c0d.th",
        config_path=Path(__file__).parent / "../configs/ukro_base_uncased.config",
    )
}


class G2PConfig(dict):

    def __init__(self, model_config_file):
        super(G2PConfig, self).__init__()

        self.use_cuda = False
        self.model_path = None

        # reading config file
        config_file = configparser.ConfigParser()
        config_file.read(model_config_file, encoding='utf8')

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
        # reading human phonemes
        self.human_phonemes = config_file['VocabConfig']['human_phonemes'].split()

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
    def from_pretrained(cls, model_name):

        if model_name not in pretrained_models:
            raise ValueError

        # load config
        config = G2PConfig(pretrained_models[model_name].config_path)  # TODO add metod from_file

        # instantiate model
        model = cls(config)

        # loading weights
        state_dict = model_zoo.load_url(pretrained_models[model_name].url,
                                        progress=True, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

        return model


class G2PModel(PreTrainedG2PModel):

    def __init__(self, config):
        super(G2PModel, self).__init__(config)

        # init
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # encoder
        self.encoder = Encoder(config)

        # decoder
        self.decoder = Decoder(config)
        self.attention = config.attention

        # generator
        self.beam_size = config.beam_size
        self.max_generate_len = config.max_generate_len

    def forward(self, x, x_length, y=None, p_length=None, n_best=1):
        # TODO rewrite desscription

        encoder_out, encoder_hc = self.encoder(x, x_length)

        if y is not None:
            out = self.decoder(y, p_length, encoder_hc, context=encoder_out if self.attention else None)
        else:
            out = self._generate(encoder_hc, context=encoder_out if self.attention else None)

        return out

    def _generate(self, hc, context=None):
        beam = Beam(self.config)
        h, c = hc
        # Make a beam_size batch.
        h = h.expand(h.size(1), beam.size, h.size(2))
        c = c.expand(c.size(1), beam.size, c.size(2))
        if context is not None:
            context = context.expand(beam.size, context.size(1), context.size(2))
        p_length = Variable(torch.from_numpy(np.array([1])))
        p_length = p_length.expand(beam.size).to(self.device)

        for i in range(self.max_generate_len):
            x = beam.get_current_state()
            o, hc = self.decoder(Variable(x.unsqueeze(1)).to(self.device), p_length, (h, c), context=context)
            if beam.advance(o.data.squeeze(1)):
                break
            h, c = hc
            h.data.copy_(h.data.index_select(1, beam.get_current_origin()))
            c.data.copy_(c.data.index_select(1, beam.get_current_origin()))
        return torch.LongTensor(beam.get_hyp(0)).unsqueeze(0)

