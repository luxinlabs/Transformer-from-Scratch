# Transformer-from-Scratch

A complete implementation of the Transformer architecture from scratch using PyTorch, trained on English-Italian translation.

## Features

- Full transformer architecture with multi-head attention
- Positional encoding and layer normalization
- Bilingual dataset with proper tokenization and masking
- Training pipeline with validation and greedy decoding
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

**Dataset download issues:**

- Check internet connection
- Set `HF_TOKEN` in `.env` file
- Try different dataset: change `lang_tgt` in `config.py`

## License

MIT
