import argparse
import torch
from torch.autograd import Variable

from ukro_g2p.models.g2p_model import G2PModel, pretrained_models
from ukro_g2p.tokenization import G2PTokenizer


class G2P(object):
    def __init__(self, model_name):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = G2PModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = G2PTokenizer.from_pretrained(model_name)
        self.model.eval()

    def __call__(self, word, human_readable=False):

        # map word's chars into graphemes idx
        graphemes = self.tokenizer.tokenize_graphemes(word)
        graphemes = self.tokenizer.convert_graphemes_to_ids(graphemes)
        g_length = [len(graphemes)]

        graphemes = Variable(torch.LongTensor(graphemes).unsqueeze(0)).to(self.device)
        g_length = Variable(torch.LongTensor(g_length)).to(self.device)

        phonemes = self.model(graphemes, g_length).tolist()[0]

        return self.tokenizer.convert_ids_to_phonemes(phonemes) if not human_readable else self.tokenizer.convert_ids_to_human_phonemes(phonemes)


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument('word', type=str, help='Word to generate pronunciation for')
    parser.add_argument('-m', '--model_name', required=False, type=str, default='ukro-base-uncased',
                        choices=pretrained_models.keys(),
                        help='Trained model name')
    parser.add_argument('-hr', '--human_readable', required=False, action='store_true',
                        help='Output human readable set of phonemes')

    # parse
    script_args = parser.parse_args()

    g2p = G2P('ukro-base-uncased')
    pron = g2p(script_args.word, human_readable=True if script_args.human_readable else False)

    print(f"{' '.join(pron)}")
