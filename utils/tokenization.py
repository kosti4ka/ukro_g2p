from pathlib import Path
from models.g2p_model import G2PConfig

PRETRAINED_TOKENIZER_MAP = {
    'ukr-base-uncased': "../trained_models/g2p_ukr"
}
CONFIG_SUFFIX = '.config'


class G2PTokenizer(object):
    def __init__(self, config):

        self.g2idx = {g: idx for idx, g in enumerate(config.graphemes)}
        self.idx2p = {idx: p for idx, p in enumerate(config.phonemes)}

    def tokenize_graphemes(self, word):
        return ['<s>'] + list(word)

    def convert_graphemes_to_ids(self, graphemes):
        return [self.g2idx[g] for g in graphemes]

    def convert_ids_to_phonemes(self, ids):
        return [self.idx2p[i] for i in ids if self.idx2p[i] not in ['<os>', '</os>']]

    @classmethod
    def from_pretrained(cls, pretrained_tokenizer_name):

        if pretrained_tokenizer_name in PRETRAINED_TOKENIZER_MAP:
            base_path = Path(__file__).parent / PRETRAINED_TOKENIZER_MAP[pretrained_tokenizer_name]
            config_file = base_path.with_suffix(CONFIG_SUFFIX)
        else:
            raise ValueError

        # load config
        config = G2PConfig(config_file)  # TODO add metod from_file
        tokenizer = cls(config)

        return tokenizer
