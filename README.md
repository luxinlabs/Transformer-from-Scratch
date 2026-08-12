# Transformer-from-Scratch

A complete implementation of the Transformer architecture from scratch using PyTorch, trained on English-Italian translation.

## Features

- Full transformer architecture with multi-head attention
- Positional encoding and layer normalization
- Bilingual dataset with proper tokenization and masking
- Training pipeline with validation and greedy decoding
- GPU acceleration on NVIDIA (CUDA) and Apple silicon (Metal/MPS)
- TensorBoard logging for monitoring training progress

## Requirements

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── model.py          # Transformer architecture implementation
├── dataset.py        # Bilingual dataset and data loading
├── train.py          # Training loop and validation
├── config.py         # Configuration parameters
└── requirements.txt  # Python dependencies
```

## Configuration

Edit `config.py` to adjust hyperparameters:

- `batch_size`: Number of samples per batch (default: 4)
- `num_epochs`: Number of training epochs (default: 5)
- `seq_len`: Maximum sequence length (default: 128)
- `d_model`: Model dimension (default: 256)
- `lang_src`: Source language (default: 'en')
- `lang_tgt`: Target language (default: 'it')

## How to Run

### Method 1: Direct Python Execution

```bash
python train.py
```

### Method 2: Debug Mode (Two Terminal Setup)

**Terminal 1 - Start TensorBoard:**

```bash
tensorboard --logdir=runs
```

Then open http://localhost:6006 in your browser to monitor training metrics.

**Terminal 2 - Run Training:**

```bash
# Regular mode
python train.py

# Or with Python debugger
python -m pdb train.py

# Or with VS Code/Cursor debugger
# Press F5 or use "Run and Debug" panel
```

### Method 3: Using Conda Environment

```bash
# Activate your conda environment
conda activate your_env_name

# Run training
python train.py
```

## Training Output

The training script will:

1. Download the `opus_books` dataset (en-it language pair)
2. Build tokenizers for source and target languages
3. Train the transformer model for the specified number of epochs
4. Save model checkpoints to `weights/` directory
5. Log training metrics to `runs/` directory for TensorBoard

## Model Checkpoints

Model weights are saved after each epoch:

- Location: `weights/tmodel_00.pt`, `weights/tmodel_01.pt`, etc.
- Contains: model state, optimizer state, epoch number, global step

## Monitoring Training

View training progress in real-time:

```bash
tensorboard --logdir=runs
```

Metrics logged:

- Training loss per batch
- Validation examples with source, target, and predicted translations

## Environment Variables

Create a `.env` file for Hugging Face authentication (optional):

```
HF_TOKEN=your_huggingface_token_here
```

## Hardware Acceleration

The device is selected automatically at startup, in order of preference:

1. **CUDA** — NVIDIA GPUs
2. **MPS** — Apple silicon GPU via Metal (M1/M2/M3/M4, macOS 12.3+)
3. **CPU** — fallback

The selected device is printed when training starts:

```
Using device: mps
Apple Metal (MPS) backend enabled
```

No configuration is needed. If a rarely-used op is missing from the Metal
backend, `PYTORCH_ENABLE_MPS_FALLBACK=1` (set automatically in `train.py`)
runs just that op on the CPU rather than crashing.

### Getting the most out of Apple silicon

The default `batch_size` of 4 leaves most of the GPU idle. Metal throughput
scales well with batch size — measured on this model (`seq_len=128`,
`d_model=256`), per training step:

| Device | Batch | s/step | samples/s |
| ------ | ----- | ------ | --------- |
| CPU    | 4     | 0.141  | 28        |
| MPS    | 4     | 0.052  | 77        |
| MPS    | 8     | 0.072  | 111       |
| MPS    | 16    | 0.116  | 138       |
| MPS    | 32    | 0.186  | 172       |

Raising `batch_size` to 16–32 in `config.py` is the single biggest win. Note
that larger batches mean fewer optimizer steps per epoch, so you may want to
raise `lr` alongside it.

## Performance Notes

- **CPU Training**: Expect ~2-5 seconds per batch on modern CPUs
- **GPU Training**: Significantly faster (recommended for production)
- **Memory**: ~2-4GB RAM for default configuration
- **Disk**: ~500MB for dataset and tokenizers

## Troubleshooting

**Slow training:**

- Reduce `seq_len` (e.g., 64 or 128)
- Reduce `batch_size` (e.g., 2 or 4)
- Reduce `d_model` (e.g., 128 or 256)

**Out of memory:**

- Reduce `batch_size`
- Reduce `seq_len`
- Use gradient accumulation

**Dataloader workers hang or crash on macOS:**

macOS starts worker processes with `spawn`, which re-imports the entry module.
Set `num_workers` to `0` in `config.py` to rule the workers out when debugging.

**Dataset download issues:**

- Check internet connection
- Set `HF_TOKEN` in `.env` file
- Try different dataset: change `lang_tgt` in `config.py`

## License

MIT
