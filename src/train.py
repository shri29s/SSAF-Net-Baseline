from .utils import Logger, vca
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from time import time
from .config import PRESETS, DEVICE
from .loader import load_data
from .models import SSAFNet
from .losses import total_loss

# Train loop
def train(Y_np, P, H, W, L, out_dir,
          epochs = 2000,
          lr = 5e-3,
          J = 4,
          alpha = 0.5,
          lam_kl = 0.1,
          lam_sad = 0.1,
          lam_vol = 0.1, logger: Logger = None):
    """
    Y_np : (L, H, W)  float32, values in [0,1]
    Returns A_hat (P, H, W), M_mean (L, P), loss_history (list)
    """
    # VCA initialisation
    logger.subsection("Initializing endmembers with VCA")
    Y_flat = Y_np.reshape(L, -1) # (L, N)
    M0 = vca(Y_flat, P).to(DEVICE) # (L, P)
    logger.result(f"VCA initialization complete (P={P} endmembers)")

    logger.subsection("Building SSAFNet model")
    model = SSAFNet(L, P, J, M0).to(DEVICE)

    # Kaiming init
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)
    logger.result(f"Model created on {DEVICE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    Y_tensor = torch.tensor(Y_np, dtype=torch.float32).unsqueeze(0).to(DEVICE) # (1, L, H, W)
    losses = []

    logger.subsection(f"Training for {epochs} epochs (LR={lr})")
    tic = time()

    for epoch in tqdm(range(1, epochs + 1), desc="Training"):
        model.train()
        Y1, Y2, A1, A2, Mn, mu, logv = model(Y_tensor)
        loss = total_loss(Y_tensor, Y1, Y2, mu, logv, Mn, alpha, lam_kl, lam_sad, lam_vol)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
        if epoch % 100 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}/{epochs:5d} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    toc = time()
    logger.result(f"Training completed in {(toc - tic):.1f}s | Final loss: {losses[-1]:.6f}")

    # Extract results
    model.eval()
    with torch.no_grad():
        _, _, A1, _, Mn, _, _ = model(Y_tensor)

    A_hat = A1.squeeze(0).cpu().numpy() # (P, H, W)
    Mn_np = Mn.cpu().numpy() # (N, L, P)
    M_mean = Mn_np.mean(axis=0) # (L, P)
    logger.result(f"Extracted results: A_hat {A_hat.shape}, M_mean {M_mean.shape}")

    return A_hat, M_mean, losses