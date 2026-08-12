"""
Interactive English -> Italian translation against a trained checkpoint.

    python chat.py                                  # uses weights/tmodel_best.pt
    python chat.py --checkpoint weights/tmodel_49.pt
    echo "Hello, how are you?" | python chat.py     # one-shot, reads stdin

Defaults to the *best* checkpoint rather than the last one: validation loss
turns upward long before training ends, so the final epoch is normally a worse
translator than an earlier one. Type 'quit' or Ctrl-D to exit.
"""

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import get_config, get_weights_file_path
from model import build_transformer
from train import get_device, greedy_decode


def load_tokenizers(config):
    tokenizers = {}
    for role, lang in (('src', config['lang_src']), ('tgt', config['lang_tgt'])):
        path = Path(config['tokenizer_file'].format(lang))
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run train.py once to build the tokenizers.")
        tokenizers[role] = Tokenizer.from_file(str(path))
    return tokenizers['src'], tokenizers['tgt']


def encode_source(text, tokenizer_src, seq_len, device):
    """Match BilingualDataset exactly: SOS + tokens + EOS + padding."""
    sos = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
    eos = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
    pad_id = tokenizer_src.token_to_id('[PAD]')

    tokens = tokenizer_src.encode(text).ids[: seq_len - 2]
    padding = seq_len - len(tokens) - 2

    encoder_input = torch.cat([
        sos,
        torch.tensor(tokens, dtype=torch.int64),
        eos,
        torch.tensor([pad_id] * padding, dtype=torch.int64),
    ]).unsqueeze(0).to(device)                                   # (1, seq_len)

    encoder_mask = (encoder_input != pad_id).unsqueeze(1).unsqueeze(1).int().to(device)
    return encoder_input, encoder_mask


def translate(text, model, tokenizer_src, tokenizer_tgt, seq_len, device):
    encoder_input, encoder_mask = encode_source(text, tokenizer_src, seq_len, device)
    output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, seq_len, device)

    # Drop the special tokens greedy_decode brackets the output with
    special = {tokenizer_tgt.token_to_id('[SOS]'), tokenizer_tgt.token_to_id('[EOS]'), tokenizer_tgt.token_to_id('[PAD]')}
    ids = [int(i) for i in output.detach().cpu().numpy() if int(i) not in special]
    return tokenizer_tgt.decode(ids)


def main():
    config = get_config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=get_weights_file_path(config, 'best'),
                        help="checkpoint to load (default: the best-validation-loss one)")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(f"No checkpoint at {args.checkpoint}. Train first, or pass --checkpoint.")

    device = get_device()
    tokenizer_src, tokenizer_tgt = load_tokenizers(config)

    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(),
                              config['seq_len'], config['seq_len'], config['d_model']).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    trained_for = f"epoch {state['epoch']}"
    if 'val_loss' in state:
        trained_for += f", val_loss {state['val_loss']:.3f}"
    print(f"{config['lang_src']} -> {config['lang_tgt']} | {args.checkpoint} ({trained_for}) | {device}")
    print("Type a sentence to translate, or 'quit' to exit.\n")

    interactive = sys.stdin.isatty()
    while True:
        if interactive:
            print("> ", end="", flush=True)
        line = sys.stdin.readline()
        if not line:                     # EOF
            break
        text = line.strip()
        if not text:
            continue
        if text.lower() in {'quit', 'exit'}:
            break
        if not interactive:
            print(f"> {text}")
        print(f"{translate(text, model, tokenizer_src, tokenizer_tgt, config['seq_len'], device)}\n")


if __name__ == '__main__':
    main()
