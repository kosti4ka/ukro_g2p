from pathlib import Path
import json
from collections import OrderedDict
import Levenshtein


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def load_list(fname):
    fname = Path(fname)
    return [x for x in open(fname, 'r', encoding='utf-8').read().split('\n') if x]


def phoneme_error_rate(p_seq1, p_seq2):
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return Levenshtein.distance(''.join(c_seq1), ''.join(c_seq2)) / len(c_seq2)


def dump(iterable, file_name, append=False):

    # init paths and make sure the dir to write in exists
    file_name = Path(file_name)
    out_dir = file_name.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(file_name, 'a' if append else 'w', encoding='utf-8') as f:
        f.writelines(('\t'.join(str(ll) for ll in l) if type(l) != str and hasattr(l, '__iter__') else str(l)).rstrip('\r\n') + '\n' for l in ([iterable] if type(iterable) == str else iterable))
