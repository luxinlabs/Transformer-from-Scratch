"""
Plot training vs validation loss from the TensorBoard logs and mark the epoch
with the best validation loss.

    python plot_losses.py                 # reads runs/tmodel, writes loss_curve*.png
    python plot_losses.py --logdir runs/tmodel --out loss_curve.png

The point of the chart is the divergence: training loss keeps falling long after
validation loss turns around, and everything past that turn is memorization.
"""

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from config import get_config

# Categorical slots 1 and 2, validated for CVD separation on both surfaces.
THEMES = {
    'light': {
        'surface': '#fcfcfb', 'train': '#2a78d6', 'val': '#eb6834',
        'ink': '#0b0b0b', 'muted': '#898781', 'grid': '#e1e0d9', 'axis': '#c3c2b7',
    },
    'dark': {
        'surface': '#1a1a19', 'train': '#3987e5', 'val': '#d95926',
        'ink': '#ffffff', 'muted': '#898781', 'grid': '#2c2c2a', 'axis': '#383835',
    },
}


def load_scalars(logdir):
    """
    Read train_loss and val_loss from the event file holding the longest run.

    A log directory accumulates one event file per run, including aborted ones,
    so the newest file is not necessarily the most complete.
    """
    event_files = sorted(glob.glob(str(Path(logdir) / "events*")))
    if not event_files:
        raise SystemExit(f"No TensorBoard event files in {logdir}")

    best = None
    for path in event_files:
        acc = EventAccumulator(path, size_guidance={'scalars': 0})
        acc.Reload()
        tags = acc.Tags()['scalars']
        if 'val_loss' not in tags:
            continue
        val = acc.Scalars('val_loss')
        if best is None or len(val) > len(best[1]):
            best = (path, val, acc.Scalars('train_loss') if 'train_loss' in tags else [])

    if best is None:
        raise SystemExit(f"No run in {logdir} logged val_loss -- retrain to record it")

    path, val, train = best
    if len(event_files) > 1:
        print(f"{len(event_files)} event files found; using the longest run ({Path(path).name})")
    return train, val


def per_epoch_train(train_scalars, steps_per_epoch):
    """Mean training loss within each epoch, so it plots against the same x as val."""
    buckets = defaultdict(list)
    for s in train_scalars:
        buckets[(s.step - 1) // steps_per_epoch].append(s.value)
    return sorted(buckets), [sum(buckets[e]) / len(buckets[e]) for e in sorted(buckets)]


def render(mode, epochs, train_y, val_epochs, val_y, best_i, out_path):
    c = THEMES[mode]
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    fig.patch.set_facecolor(c['surface'])
    ax.set_facecolor(c['surface'])

    best_epoch, best_val = val_epochs[best_i], val_y[best_i]

    # Everything right of the turn is memorization, not learning
    if best_epoch < val_epochs[-1]:
        ax.axvspan(best_epoch, val_epochs[-1], color=c['val'], alpha=0.05, zorder=0)
        ax.text((best_epoch + val_epochs[-1]) / 2, max(max(train_y), max(val_y)) * 0.97,
                'overfitting — validation loss rising', ha='center', va='top',
                fontsize=9, color=c['muted'], zorder=2)

    ax.grid(True, color=c['grid'], linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)

    ax.plot(epochs, train_y, color=c['train'], linewidth=2, label='Training loss', zorder=3)
    ax.plot(val_epochs, val_y, color=c['val'], linewidth=2, label='Validation loss', zorder=3)

    # The headline: where validation bottomed out
    ax.plot([best_epoch], [best_val], marker='o', markersize=9, color=c['val'],
            markeredgecolor=c['surface'], markeredgewidth=2, zorder=5)
    ax.annotate(f'best — epoch {best_epoch} · {best_val:.3f}',
                xy=(best_epoch, best_val), xytext=(best_epoch + 2.5, best_val - 0.62),
                fontsize=10, color=c['ink'], fontweight='bold', zorder=5,
                arrowprops=dict(arrowstyle='-', color=c['axis'], linewidth=1))
    ax.axvline(best_epoch, color=c['axis'], linewidth=1, linestyle=(0, (4, 4)), zorder=2)

    ax.set_xlabel('Epoch', fontsize=10, color=c['muted'])
    ax.set_ylabel('Cross-entropy loss', fontsize=10, color=c['muted'])
    ax.set_title('Training vs validation loss', fontsize=14, color=c['ink'],
                 fontweight='bold', loc='left', pad=18)
    ax.text(0, 1.02, f'Best model at epoch {best_epoch} of {val_epochs[-1] + 1} — '
                     f'the final epoch is {val_y[-1] - best_val:+.3f} worse',
            transform=ax.transAxes, fontsize=10, color=c['muted'], va='bottom')

    ax.tick_params(colors=c['muted'], labelsize=9)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(c['axis'])

    legend = ax.legend(frameon=False, fontsize=10, loc='lower left')
    for text in legend.get_texts():
        text.set_color(c['ink'])

    fig.text(0.01, 0.015,
             'Training loss uses label smoothing 0.1; validation does not — compare the shapes, not the gap.',
             fontsize=8, color=c['muted'])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, facecolor=c['surface'])
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    config = get_config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default=config['experiment_name'])
    parser.add_argument('--out', default='loss_curve.png')
    args = parser.parse_args()

    train_scalars, val_scalars = load_scalars(args.logdir)
    val_epochs = list(range(len(val_scalars)))
    val_y = [s.value for s in val_scalars]

    steps_per_epoch = max(1, val_scalars[0].step)  # val is logged once per epoch
    epochs, train_y = per_epoch_train(train_scalars, steps_per_epoch)

    best_i = min(val_epochs, key=lambda i: val_y[i])

    out = Path(args.out)
    render('light', epochs, train_y, val_epochs, val_y, best_i, out)
    render('dark', epochs, train_y, val_epochs, val_y, best_i,
           out.with_name(f"{out.stem}_dark{out.suffix}"))

    # Table view, so the numbers are readable without the chart
    print(f"\n{'epoch':>6} {'train':>8} {'val':>8}")
    for i in val_epochs:
        marker = '  <- best' if i == best_i else ''
        train_at = f"{train_y[i]:8.3f}" if i < len(train_y) else " " * 8
        print(f"{i:>6} {train_at} {val_y[i]:8.3f}{marker}")
    print(f"\nBest: epoch {best_i}, val_loss {val_y[best_i]:.3f}. "
          f"Final epoch {val_epochs[-1]}: {val_y[-1]:.3f} ({val_y[-1] - val_y[best_i]:+.3f}).")


if __name__ == '__main__':
    main()
