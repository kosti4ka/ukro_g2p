from pathlib import Path
from ukro_g2p.models.g2p_model import G2PConfig
from collections import namedtuple

tokenizer_obj = namedtuple("tokenizer_obj", ["config_path"])

pretrained_tokenizers = {
    "ukro-base-uncased": tokenizer_obj(
        config_path=Path(__file__).parent / "configs/ukro_base_uncased.config",
    )
}


class G2PTokenizer(object):
    def __init__(self, config):

        self.g2idx = {g: idx for idx, g in enumerate(config.graphemes)}
        self.idx2p = {idx: p for idx, p in enumerate(config.phonemes)}
        self.idx2h = {idx: h for idx, h in enumerate(config.human_phonemes)}

    def tokenize_graphemes(self, word):
        return list(word)

    def convert_graphemes_to_ids(self, graphemes):
        return [self.g2idx[g] for g in graphemes]

    def convert_ids_to_phonemes(self, ids):
        return [self.idx2p[i] for i in ids if self.idx2p[i] not in ['<os>', '</os>']]

    def convert_ids_to_human_phonemes(self, ids):
        return [self.idx2h[i] for i in ids if self.idx2h[i] not in ['<os>', '</os>']]

    @classmethod
    def from_pretrained(cls, tokenizer_name):

        if tokenizer_name not in pretrained_tokenizers:
            raise ValueError

        # load config
        config = G2PConfig(pretrained_tokenizers[tokenizer_name].config_path)  # TODO add metod from_file
        tokenizer = cls(config)

        return tokenizer
