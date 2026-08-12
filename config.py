from pathlib import Path

def get_config():
    return {
        # Tuned to saturate the GPU. Measured end-to-end on an M5 Pro (20 cores):
        # batch 4 -> 60 samples/s, 16 -> 118, 32 -> 131, 64 -> 132, 128 -> 125.
        # 32 buys ~99% of peak throughput while keeping twice as many optimizer
        # steps per epoch as 64, and uses only ~1.3GB of the ~55GB available.
        'batch_size': 32,
        # Not the bottleneck: throughput is identical for 0/2/4/8 workers because
        # the step is GPU-bound. 2 is enough to hide tokenization. 0 to debug.
        'num_workers': 2,
        'num_epochs': 5,
        # Scaled with batch_size (sqrt rule: 1e-4 * sqrt(32/4) ~= 2.8e-4). A bigger
        # batch means 8x fewer updates per epoch, so the old 1e-4 would now
        # underfit. There is no warmup here, so avoid going much above this.
        'lr': 3e-4,
        'seq_len': 128,
        'd_model': 256,
        'lang_src': 'en',
        'lang_tgt': 'it',
        'model_folder': 'weights',
        'model_basename': 'tmodel',
        'preload': None,
        'tokenizer_file': "tokenizer_{0}.json",
        'experiment_name': 'runs/tmodel'
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}_{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)