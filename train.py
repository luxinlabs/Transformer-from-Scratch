import os

# Let unsupported ops fall back to CPU instead of crashing on Apple Metal (MPS).
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace # Split the word by whitespace
from torch.utils.tensorboard import SummaryWriter

import warnings
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_device():
    """
    Pick the best available accelerator: CUDA, then Apple Metal (MPS), then CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.backends.mps.is_built():
        # Built with MPS support but no usable device (needs macOS 12.3+ on Apple silicon)
        print("MPS is built but not available on this machine, falling back to CPU")
    return torch.device("cpu")

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precoompute the encoder output and reuse it we get from the decoder
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) >= max_len:
            break
        
        # Build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of decoder
        out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)

        # Get the next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        
        if next_word.item() == eos_idx:
            break
    return decoder_input.squeeze(0)
    
    

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    # Size of the control window (Just use a default value)
    control_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device, non_blocking=True)
            encoder_mask = batch["encoder_mask"].to(device, non_blocking=True)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            # Print to the console
            print_msg('-' * control_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")
            
            if count == num_examples:
                break

    if writer:
        # Evaluate the common metrics, TorchMetrics, CharErrorRate, BLEU, WordErrorRate
        pass


def compute_validation_loss(model, dataloader, loss_fn, vocab_size, device):
    """
    Average per-sample cross entropy over the validation set.

    This is the only signal that says when to stop: train loss keeps falling
    straight through the point where the model starts overfitting, so it cannot
    be used to choose a checkpoint.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            proj_output = model.project(model.decode(decoder_input, model.encode(encoder_input, encoder_mask), encoder_mask, decoder_mask))
            loss = loss_fn(proj_output.reshape(-1, vocab_size), label.reshape(-1))

            batch_size = label.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    model.train()
    return total_loss / max(total_samples, 1)


def get_all_sentences(ds, lang):
    """
    Get all sentences from the dataset for a given language.
    """
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
    Get or build a tokenizer for the given language.
    """
    # config['tokenizer_file'] = "../tokenizers/tokenizer_{0}.json"
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tolenizer = Tokenizer(WordLevel(unk_token="[UNK]")) # If the word is not in the vocabulary, use [UNK]
        tolenizer.pre_tokenizer = Whitespace() # splict the words using whitespace
        trainer = WordLevelTrainer(special_tokens=["[PAD]", "[SOS]", "[EOS]", "[UNK]"], min_frequency=2)
        tolenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tolenizer.save(str(tokenizer_path))
    else:
        tolenizer = Tokenizer.from_file(str(tokenizer_path))
    return tolenizer

def get_ds(config, device):
    """
    Get the dataset.
    """
    # The bare "opus_books" id no longer resolves; the hub requires namespace/name
    ds_raw = load_dataset("Helsinki-NLP/opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train")

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% of the data for training and 10% for validation
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_tokens = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_tokens = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_tokens))
        max_len_tgt = max(max_len_tgt, len(tgt_tokens))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    # Tokenization happens per item, so worker processes keep the GPU fed.
    # pin_memory is a CUDA-only optimization and is a no-op cost on MPS.
    num_workers = config['num_workers']
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': device.type == 'cuda',
        'persistent_workers': num_workers > 0,
    }

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, **loader_kwargs)
    # batch_size 1: greedy_decode only handles one sentence at a time
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, **loader_kwargs)
    # Separate batched loader for scoring val loss, which needs no decoding
    val_loss_dataloader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, **loader_kwargs)

    return train_dataloader, val_dataloader, val_loss_dataloader, tokenizer_src, tokenizer_tgt



def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    Get the model.
    """
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(config):
    """
    Train the model.
    """
    # Create the device: CUDA, Apple Metal (MPS), or CPU
    device = get_device()
    print(f"Using device: {device}")
    if device.type == "mps":
        print("Apple Metal (MPS) backend enabled")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Create the dataset
    train_dataloader, val_dataloader, val_loss_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config, device)

    # Create the model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    model.to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        # map_location keeps checkpoints portable between cpu / mps / cuda
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        print(f"Preloaded model from epoch {initial_epoch}")

    pad_token = tokenizer_src.token_to_id('[PAD]')
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token, label_smoothing=0.1).to(device)
    # No label smoothing for validation: smoothing adds a constant floor that
    # makes the number harder to read, and we only need it to be comparable
    # across epochs.
    val_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token).to(device)

    best_val_loss = float('inf')
    best_epoch = None

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iter = tqdm(train_dataloader, desc=f"Processing epoch {epoch}")
        for batch in batch_iter:
            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            
            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Extract Labels
            label = batch['label'].to(device) # (B, seq_len)
            
            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.reshape(-1, tokenizer_tgt.get_vocab_size()), label.reshape(-1))
            batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            # Backpropagate the loss
            loss.backward()
            
            # Update the weights
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()
        
        # Run validation at the end of the epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iter.write(msg), global_step, writer)

        # Score the validation set so we can tell overfitting from progress
        val_loss = compute_validation_loss(model, val_loss_dataloader, val_loss_fn, tokenizer_tgt.get_vocab_size(), device)
        writer.add_scalar('val_loss', val_loss, global_step)
        writer.flush()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
        batch_iter.write(f"Epoch {epoch:02d} val_loss: {val_loss:6.3f}" + (" (best so far)" if is_best else f" (best was {best_val_loss:6.3f} at epoch {best_epoch:02d})"))

        # Save the model at the end of every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'val_loss': val_loss
        }
        torch.save(checkpoint, get_weights_file_path(config, f"{epoch:02d}"))
        # Keep a copy of the best epoch so it does not have to be hunted down
        # afterwards; the final epoch is usually not the best one.
        if is_best:
            torch.save(checkpoint, get_weights_file_path(config, 'best'))

    if best_epoch is not None:
        print(f"\nBest validation loss: {best_val_loss:.3f} at epoch {best_epoch:02d} -> {get_weights_file_path(config, 'best')}")
    writer.close()

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
    