import torch.nn as nn
import torch.nn.functional as F
import torch

# TODO remove config, send par explicitly


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()

        self.vocab_size = config.encoder_vocab_size
        self.padding_idx = config.encoder_padding_idx

        self.d_embed = config.encoder_d_embed
        self.d_hidden = config.encoder_d_hidden
        self.num_layers = config.encoder_n_layers
        self.bidirectional = config.encoder_bidirectional

        self.embedding = nn.Embedding(self.vocab_size, self.d_embed, padding_idx=self.padding_idx)
        self.lstm = nn.LSTM(self.d_embed, self.d_hidden // 2 if self.bidirectional else self.d_hidden, self.num_layers,
                            batch_first=True, bidirectional=self.bidirectional)

    def forward(self, x, x_length):

        x = self.embedding(x)  # B x T x D
        x = nn.utils.rnn.pack_padded_sequence(x, x_length, batch_first=True, enforce_sorted=False)

        out, hc = self.lstm(x)

        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        if self.bidirectional:
            # stacking hidden and cell from forward and backward layers
            hc = tuple(torch.cat((hc_[0::2, :, :], hc_[1::2, :, :]), 2) for hc_ in hc)

        return out, hc


class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()

        self.vocab_size = config.decoder_vocab_size
        self.padding_idx = config.decoder_padding_idx

        self.d_embed = config.decoder_d_embed
        self.d_hidden = config.decoder_d_hidden
        self.num_layers = config.decoder_n_layers

        self.embedding = nn.Embedding(self.vocab_size, self.d_embed, padding_idx=self.padding_idx)
        self.lstm = nn.LSTM(self.d_embed, self.d_hidden, self.num_layers, batch_first=True)
        if config.attention:
            self.attn = Attention(self.d_hidden)
        else:
            self.attn = None

        self.linear = nn.Linear(self.d_hidden, self.vocab_size)

    def forward(self, y, y_length, hc, context=None):

        batch_size, seq_len = y.size()

        y = self.embedding(y)  # B x T x D
        y = nn.utils.rnn.pack_padded_sequence(y, y_length, batch_first=True, enforce_sorted=False)

        out, hc = self.lstm(y, hc)

        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        if self.attn:
            out = self.attn(out, context)

        out = out.contiguous()

        out = self.linear(out.view(-1, out.size(2)))

        return F.log_softmax(out, dim=1).view(batch_size, -1, out.size(1)), hc


class Attention(nn.Module):
    """Dot global attention from https://arxiv.org/abs/1508.04025"""

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, x, context):

        batch_size, seq_len, _ = x.size()

        attn = F.softmax(x.bmm(context.transpose(1, 2)), dim=2)
        weighted_context = attn.bmm(context)

        o = self.linear(torch.cat((x, weighted_context), 2).view(batch_size * seq_len, -1))
        return torch.tanh(o).view(batch_size, seq_len, -1)


class Beam(object):
    """Ordered beam of candidate outputs"""

    def __init__(self, config):
        """Initialize params"""
        self.size = config.beam_size
        self.done = False
        self.pad = config.decoder_padding_idx
        self.bos = config.decoder_bos_idx
        self.eos = config.decoder_eos_idx
        self.tt = torch.cuda if config.use_cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(self.size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(self.size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(best_scores_id - prev_k * num_words)
        # End condition is when n-best are EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True
        return self.done

    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]
        return hyp[::-1]
