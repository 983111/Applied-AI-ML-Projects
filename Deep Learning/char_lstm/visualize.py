"""
character_level_lstm/visualize.py
───────────────────────────────────
Plot the training loss curve and a summary of gate activations.
Requires matplotlib (optional).

Usage:
    python visualize.py
"""

import json
import os

def plot_loss(history):
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(history, color="#4f86c6", linewidth=1.5)
        ax.set_xlabel("Epoch",         fontsize=12)
        ax.set_ylabel("Avg. Cross-Entropy Loss", fontsize=12)
        ax.set_title("Character-Level LSTM — Training Loss", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.fill_between(range(len(history)), history, alpha=0.1, color="#4f86c6")

        # Annotate best
        best_epoch = int(min(range(len(history)), key=lambda i: history[i]))
        ax.annotate(f"Best: {history[best_epoch]:.4f}",
                    xy=(best_epoch, history[best_epoch]),
                    xytext=(best_epoch + len(history)*0.05, history[best_epoch] + 0.1),
                    arrowprops=dict(arrowstyle="->", color="red"),
                    color="red", fontsize=10)

        plt.tight_layout()
        path = "saved_model/loss_curve.png"
        plt.savefig(path, dpi=150)
        print(f"Loss curve saved → {path}")
        plt.close()
    except ImportError:
        print("matplotlib not available — skipping plot")
        text_plot(history)


def text_plot(history):
    """ASCII art loss curve fallback."""
    rows   = 12
    cols   = min(len(history), 60)
    step   = max(1, len(history) // cols)
    sample = [history[i] for i in range(0, len(history), step)][:cols]

    lo, hi = min(sample), max(sample)
    span   = max(hi - lo, 1e-8)

    print("\nTraining Loss (ASCII)\n" + "─" * (cols + 4))
    for r in range(rows, -1, -1):
        thresh = lo + (r / rows) * span
        row    = "".join("█" if v <= thresh else " " for v in sample)
        label  = f"{thresh:5.3f} │"
        print(f"{label}{row}│")
    print(" " * 7 + "└" + "─" * cols + "┘")
    print(f"  Loss went {history[0]:.4f} → {history[-1]:.4f}  "
          f"(↓{(history[0]-history[-1])/history[0]*100:.1f}%)")


if __name__ == "__main__":
    path = "saved_model/loss_history.json"
    if not os.path.exists(path):
        print("No training history found. Run train.py first.")
    else:
        with open(path) as f:
            history = json.load(f)
        text_plot(history)       # always show ASCII
        plot_loss(history)       # also try matplotlib
