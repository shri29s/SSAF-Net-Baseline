import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import PRESETS, DEVICE
from src.utils import Logger
from src.loader import load_data
from src.train import train
from src.utils import Plotter
from src.metrics import align_and_eval

def main(args):
    logger = Logger()

    preset = PRESETS.get(args.dataset, {})
    out_dir = os.path.join(args.out_dir, args.dataset or "custom") or os.path.join("results", args.dataset or "custom")
    os.makedirs(out_dir, exist_ok=True)
    
    # Setup logging to file and console
    log_file = logger.setup_logging(out_dir)
    logger.step_msg(f"Loading data from {args.dataset}")
    
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Log file: {log_file}")

    # Step 1: Load data
    Y_np, A_true, M_true, P_preset, H, W, L = load_data(args, preset, logger=logger)
    N = H * W
    logger.result(f"Data loaded successfully")

    # Step 2: Determine P
    logger.step_msg("Determining number of endmembers (P)")
    if args.p:
        P = args.p
        logger.info(f"User-specified P = {P}")
    elif P_preset:
        P = P_preset
        logger.info(f"Using preset P = {P} for dataset '{args.dataset}'")
    else:
        raise ValueError("Specify --p, or use a named dataset preset with a default P.")

    epochs = args.epochs or preset.get("epochs", 2000)
    lr = args.lr or preset.get("lr", 5e-3)

    # Steps 3-4: Train
    logger.step_msg(f"Training SSAFNet model")
    A_hat, M_mean, losses = train(
        Y_np, P, H, W, L, 
        out_dir, epochs=epochs, 
        lr=lr, J=args.z_dim, 
        lam_kl=args.lam_kl, 
        lam_sad=args.lam_sad, 
        lam_vol=args.lam_vol,
        logger=logger
    )

    # Step 5: Save plots
    logger.step_msg("Generating visualizations and saving results")
    Plotter.save_loss_curve(losses, out_dir)
    Plotter.save_abundance_maps(A_hat, P, out_dir)
    Plotter.save_endmember_spectra(M_mean, P, L, out_dir, M_true)

    # Step 6: Metrics (if GT available)
    if A_true is not None:
        logger.step_msg("Evaluating against ground truth")
        A_hat_flat = A_hat.reshape(P, N)
        metrics = align_and_eval(A_hat_flat, M_mean, A_true, M_true)

        logger.subsection(f"Quantitative Results - {args.dataset or 'custom'}")
        logger.info(f"aRMSE_A: {metrics['rmse_a']:.6f}")
        if "rmse_m" in metrics:
            logger.info(f"aRMSE_M: {metrics['rmse_m']:.6f}")
            logger.info(f"aSAD_M: {metrics['sad_m']:.4f} rad ({np.degrees(metrics['sad_m']):.2f}°)")
            for i, s in enumerate(metrics["sad_per_em"]):
                logger.info(f"  EM{i+1} SAD: {s:.4f} rad ({np.degrees(s):.2f}°)")

        if A_true is not None:
            fig, axes = plt.subplots(2, P, figsize=(4 * P, 7))
            order = metrics["order"]
            for i in range(P):
                ax = axes[0, i]
                ax.imshow(A_hat[order[i]].reshape(H, W), cmap="jet", vmin=0, vmax=1)
                ax.set_title(f"Est. EM #{i+1}", fontsize=10)
                ax.axis("off")

                ax = axes[1, i]
                ax.imshow(A_true[i].reshape(H, W), cmap="jet", vmin=0, vmax=1)
                ax.set_title(f"GT EM #{i+1}", fontsize=10)
                ax.axis("off")
            plt.suptitle("Abundance Map Comparison", fontsize=14, fontweight="bold")
            plt.tight_layout()

            os.makedirs(out_dir, exist_ok=True)
            cmp_path = os.path.join(out_dir, "abundance_comparison.png")
            plt.savefig(cmp_path, dpi=150)
            plt.close()
            logger.result(f"Saved abundance_comparison.png")
    else:
        logger.step_msg("No ground truth available - skipping quantitative evaluation")

    # Save numpy arrays
    logger.step_msg("Saving model outputs")
    np.save(os.path.join(out_dir, "A_hat.npy"), A_hat)
    np.save(os.path.join(out_dir, "M_mean.npy"), M_mean)
    logger.result(f"Saved A_hat.npy, M_mean.npy")
    logger.info(f"All outputs saved to: {out_dir}")
    logger.info(f"Log file saved to: {log_file}")

# CLI
def get_args():
    pa = argparse.ArgumentParser(description="SSAF-Net for Hyperspectral Unmixing")

    # Data
    pa.add_argument("--dataset", type=str, default="custom", help="Named preset: jasper|apex|dc2 (default: custom)")
    pa.add_argument("--mat", type=str, default=None, help="Path to .mat file")
    pa.add_argument("--npy", type=str, default=None, help="Path to .npy file")
    pa.add_argument("--npy_a", type=str, default=None, help="Path to abundance .npy file")
    pa.add_argument("--npy_m", type=str, default=None, help="Path to endmember .npy file")
    pa.add_argument("--y_key", type=str, default=None, help="Key for hyperspectral data in .mat")
    pa.add_argument("--gt_key", type=str, default=None, help="Key for abundance GT in .mat")
    pa.add_argument("--em_key", type=str, default=None, help="Key for endmember GT in .mat")

    # Dimensions (Override presets)
    pa.add_argument("--p", type=int, default=None, help="Number of endmembers (P)")
    pa.add_argument("--h", type=int, default=None, help="Height (H)")
    pa.add_argument("--w", type=int, default=None, help="Width (W)")

    # Training
    pa.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    pa.add_argument("--lr", type=float, default=None, help="Learning rate")
    pa.add_argument("--z_dim", type=int, default=4, help="Latent dimension J for EV-Net (default: 4)")
    pa.add_argument("--lam_kl", type=float, default=0.1, help="Weight for KL divergence loss (default: 0.1)")
    pa.add_argument("--lam_sad", type=float, default=0.1, help="Weight for SAD loss (default: 0.1)")
    pa.add_argument("--lam_vol", type=float, default=0.1, help="Weight for volume loss (default: 0.1)")

    # Output
    pa.add_argument("--out_dir", type=str, default="results", help="Directory to save results (default: results/{dataset})")

    return pa.parse_args()

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main(get_args())
