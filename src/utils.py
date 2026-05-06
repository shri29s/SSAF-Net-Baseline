import torch
import numpy as np
import logging
import os
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("Agg") # No display needed

# Endmember number estimation removed
# Automatic HySime/PCA estimation functions have been removed.
# Use the command-line `--p` option or dataset presets to set the
# number of endmembers (P).

# VCA (endmember initialisation)
def vca(Y, p):
    """
    Vertex Component Analysis.
    Y : (L, N)   → returns M0 : (L, P) as float32 torch tensor
    """
    L, N = Y.shape
    R = np.zeros((L, p), dtype=np.float32)

    for i in range(p):
        f = np.random.randn(L)
        if i > 0:
            U = np.linalg.svd(R[:, :i], full_matrices=False)[0]
            f = f - U @ (U.T @ f)
        idx = np.argmax(np.abs(Y.T @ f))
        R[:, i] = Y[:, idx]
    
    return torch.tensor(R, dtype=torch.float32)

class Logger:
    # Global logging setup
    logger = None
    _step_counter = 0

    def __init__(self, out_dir="results"):
        self.setup_logging(out_dir)

    def setup_logging(self, out_dir):
        """Configure logging to write to both console and file."""
        self._step_counter = 0
        
        self.logger = logging.getLogger("SSAFNet")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        console_formatter = logging.Formatter("%(message)s")
        file_formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        os.makedirs(out_dir, exist_ok=True)
        log_file = os.path.join(out_dir, "training.log")
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        return log_file

    def step_msg(self, msg):
        """Log a major step message."""
        self._step_counter += 1
        self.logger.info(f"\n[Step {self._step_counter}] {msg}")

    def info(self, msg):
        """Log an info message."""
        self.logger.info(f"  -> {msg}")

    def subsection(self, msg):
        """Log a subsection message."""
        self.logger.info(f"  [{msg}]")

    def result(self, msg):
        """Log a success/result message."""
        self.logger.info(f"  [OK] {msg}")


# Plotting
class Plotter:
    @staticmethod
    def save_abundance_maps(A_hat, P, out_dir, name="abundance_maps.png", logger: Logger = None):
        # Scale each panel from the map aspect ratio and clamp extremes.
        if A_hat.ndim == 3:
            _, H, W = A_hat.shape
        elif A_hat.ndim == 2:
            H = A_hat.shape[-1]
            W = A_hat.shape[-1]
        else:
            H = 1
            W = 1

        aspect = W / max(H, 1)
        panel_h = 4.5
        panel_w = float(np.clip(panel_h * aspect, 1.8, 6.0))
        fig_w = max(P * panel_w, 3.0)
        fig_h = panel_h + 0.7

        fig, axes = plt.subplots(1, P, figsize=(fig_w, fig_h))
        if P == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            im = ax.imshow(A_hat[i].reshape(-1, A_hat.shape[-1]) if A_hat.ndim == 2 else A_hat[i], cmap="jet", vmin=0, vmax=1)
            ax.set_title(f"Endmember {i+1}", fontsize=12)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.suptitle("Estimated Abundance Maps", fontsize=14, fontweight="bold")
        plt.tight_layout()

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        plt.savefig(path, dpi=300)
        plt.close()
        if logger:
            logger.result(f"Saved {name}")
        else:
            print(f"Saved {name}")

    @staticmethod
    def save_endmember_spectra(M_mean, P, L, out_dir,  M_true=None, name="endmember_spectra.png", logger: Logger = None):
        fig, axes = plt.subplots(1, P, figsize=(4 * P, 3), sharey=True)
        if P == 1:
            axes = [axes]
        wl = np.arange(L)

        for i, ax in enumerate(axes):
            ax.plot(wl, M_mean[:, i], label="Estimated", color="#2d6a9f", lw=1.5)
            if M_true is not None:
                ax.plot(wl, M_true[:, i], label="GT", color="#e74c3c", lw=1.2, ls="--")

            ax.set_title(f"Endmember {i+1}", fontsize=12)
            ax.set_xlabel("Band")

            if i == 0:
                ax.set_ylabel("Reflectance")
            ax.legend(fontsize=8)

        plt.suptitle("Estimated Endmember Spectra", fontsize=14, fontweight="bold")
        plt.tight_layout()

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        plt.savefig(path, dpi=300)
        plt.close()
        if logger:
            logger.result(f"Saved {name}")
        else:
            print(f"Saved {name}")

    @staticmethod
    def save_loss_curve(losses, out_dir, name="loss_curve.png", logger: Logger = None):
        plt.figure(figsize=(8, 4))
        plt.plot(losses, color="#9b59b6", lw=1.2)
        plt.xlabel("Epoch")
        plt.ylabel("Total Loss")
        plt.title("Training Loss Curve")
        plt.yscale("log")
        plt.grid(alpha=0.3)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        plt.savefig(path, dpi=300)
        plt.close()
        if logger:
            logger.result(f"Saved {name}")
        else:
            print(f"Saved {name}")