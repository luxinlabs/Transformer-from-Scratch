import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]["translation"]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        # Transform the text into tokens
        # Truncate to leave room for the special tokens: SOS + EOS on the encoder
        # side, and a single SOS/EOS on the decoder side. About 1% of opus_books
        # pairs are longer than a typical seq_len, and raising here would abort
        # training partway through the epoch.
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids[: self.seq_len - 2]
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids[: self.seq_len - 1]

        enc_num_padded_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padded_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Add SOS and EOS tokens with padding to the source text
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token.item()] * enc_num_padded_tokens, dtype=torch.int64)
        ])
        # Add SOS to the decoder input (no EOS - that's only in the label)
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token.item()] * dec_num_padded_tokens, dtype=torch.int64)
        ])

        # Add EOS to the label (what we expect as output from the decoder)
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token.item()] * dec_num_padded_tokens, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_len, f"encoder_input size {encoder_input.size(0)} != seq_len {self.seq_len}"
        assert decoder_input.size(0) == self.seq_len, f"decoder_input size {decoder_input.size(0)} != seq_len {self.seq_len}, dec_tokens={len(dec_input_tokens)}, padding={dec_num_padded_tokens + 1}"
        assert label.size(0) == self.seq_len, f"label size {label.size(0)} != seq_len {self.seq_len}"

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, 1, seq_len) & (1, seq_len, seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0