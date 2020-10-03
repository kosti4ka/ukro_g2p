from ukro_g2p.utils.util import load_list


def read_lexicon_dataset(lexicon_dataset_path):
    """
    Read lexicon out of file to the dict in format - "id word pronunciation"
    :param lexicon_dataset_path: path to the lexicon dataset text file
    :return: lexicon dictionary
    """

    lexicon = {x.split()[0]: ' '.join(x.split()[1:]) for x in load_list(lexicon_dataset_path)}

    return lexicon
