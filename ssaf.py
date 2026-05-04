import os
import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg") # No display needed

PRESETS = {
    "jasper": dict(
        y_key="Y", gt_key="A", em_key="M",
        p=4, h=100, w=100,
        epochs=2000, lr=5e-3
    ),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# P estimation (HySime + PCA scree)
def estimate_noise(Y: torch.Tensor):
    """
    Per-band noise estimation via multiple regression.
    Y : (L, N)
    Returns noise matrix (L, N) and noise covariance (L, L).
    """
    L, N = Y.shape
    noise = np.zeros_like(Y)
    for b in range(L):
        others = np.delete(Y, b, axis=0)
        w, _, _, _ = np.linalg.lstsq(others.T, Y[b], rcond=None)
        noise[b] = Y[b] - w @ others
    Rn = (noise @ noise.T) / N
    return noise, Rn

def hysime(Y, verbose=True):
    """
    HySime: Hyperspectral Signal Identification by Minimum Error.
    Bioucas-Dias & Nascimento, IEEE TGRS 2008.

    Y : (L, N)
    Returns estimated P (int).
    """
    L, N = Y.shape
    _, Rn = estimate_noise(Y) # Noise covariance
    Ry = (Y @ Y.T) / N # Total data covariance
    Rx = Ry - Rn # Signal covariance

    # Eigen decomposisiton of Ry (descending)
    vals_y, vecs = np.linalg.eigh(Ry)
    idx = np.argsort(vals_y)[::-1]
    vecs = vecs[:, idx]

    errors = []
    for k in range(1, L + 1):
        Ek = vecs[:, :k]
        proj_err = np.trace(Ry - Ek @ Ek.T @ Rx @ Ek @ Ek.T)
        noise_cont = 2 * np.trace(Ek.T @ Rn @ Ek)
        errors.append(proj_err + noise_cont)

    p_best = int(np.argmin(errors)) + 1
    return p_best, np.array(errors)

def pca_scree(Y, max_p=30):
    """
    Y : (L, N)
    Returns (elbow_p, explained_variance_ratios).
    """
    max_p = min(max_p, Y.shape[0], Y.shape[1])
    Yc = Y - Y.mean(axis=1, keepdims=True)
    cov = (Yc @ Yc.T) / Y.shape[1]
    vals = np.linalg.eigvalsh(cov)[::-1]
    vals = np.maximum(vals, 0)
    ratio = vals[:max_p] / vals.sum()

    if len(ratio) >= 3:
        d2 = np.diff(np.diff(ratio))
        elbow = int(np.argmax(d2)) + 2
    else:
        elbow = len(ratio)

    return elbow, ratio

def estimate_p(Y, out_dir, max_p=30, verbose=True):
    """
    Runs HySime + PCA, saves combined plot, returns recommended P.

    Y       : (L, N)
    out_dir : directory to save 'p_estimation.png'
    """
    max_p = min(max_p, Y.shape[0] - 1, Y.shape[1] - 1)

    # HySime
    print("Running HySime", end=" ", flush=True)
    p_hysime, hy_errors = hysime(Y, verbose=False)
    print(f"P = {p_hysime}")

    # PCA scree
    print("Running PCA scree", end=" ", flush=True)
    p_pca, pca_ratio = pca_scree(Y, max_p=max_p)
    pca_cum = np.cumsum(pca_ratio)
    print(f"P = {p_pca}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Endmember Number Estimation", fontsize=13)

    # Hysime error curve
    k_vals = np.arange(1, len(hy_errors) + 1)
    axes[0].plot(k_vals, hy_errors, color="#2d6a9f", lw=1.5)
    axes[0].axvline(p_hysime, color="#e74c3c", label=f"HySime: p={p_hysime}")
    axes[0].set_xlabel("Subspace dimension K")
    axes[0].set_ylabel("Error criterion")
    axes[0].set_title("HySime")
    axes[0].legend()
    axes[0].set_xlim(1, min(40, len(hy_errors)))

    # PCA scree
    xs = np.arange(1, len(pca_ratio) + 1)
    axes[1].bar(xs, pca_ratio * 100, color="#2ecc71", alpha=0.7)
    axes[1].axvline(p_pca, color="#e74c3c", ls="--", label=f"Elbow: p={p_pca}")
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Explained variance (%)")
    axes[1].set_title("PCA Scree")
    axes[1].legend()

    # Cumulative variance
    axes[2].plot(xs, pca_cum * 100, "o-", color="#9b59b6", lw=1.5, ms=4)
    for thr, col, lbl in [(99, "#e67e22", "99%"), (99.9, "#e74c3c", "99.9%")]:
        axes[2].axhline(thr, color=col, ls="--", alpha=0.7, label=lbl)
        idx99 = int(np.searchsorted(pca_cum, thr / 100))
        if idx99 < len(xs):
            axes[2].axvline(xs[idx99], color=col, ls=":", alpha=0.5)
    axes[2].set_xlabel("Components")
    axes[2].set_ylabel("Cumulative Variance (%)")
    axes[2].set_title("Cumulative Explained Variance")
    axes[2].legend()

    plt.tight_layout()
    save_path = os.path.join(out_dir, "p_estimate.png")
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

    # Decision
    if p_hysime == p_pca:
        P = p_hysime
        note = "both methods agree"
    else:
        P = p_hysime # HySime is more reliable for HSI
        note = f"HySime preferred (PCA value: {p_pca})"

    print(f"\nUsing P = {P} [{note}]")
    return P

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

# Attention modules
class ChannelAttention(nn.Module):
    def __init__(self, channels, k=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)

    def forward(self, x): # x (B, C, H, W)
        avg = x.mean(dim=[2, 3])
        mx = x.amax(dim=[2, 3])
        avg_out = self.conv(avg.unsqueeze(1)).squeeze(1)
        mx_out = self.conv(mx.unsqueeze(1)).squeeze(1)
        att = torch.sigmoid(avg_out + mx_out)
        return x * att[:, :, None, None]
    
class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k//2, bias=False)

    def forward(self, x): # x: (B, C, H, W)
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att
    

# Encoders
class EncoderSpa(nn.Module):
    # Spatial encoder (3x3 convs + Channel attention)
    def __int__(self, L, P):
        super.__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(L, 32, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU(0.2),
                                   nn.Dropout2d(0.1))
        self.cam1 = ChannelAttention(32)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(16), 
                                   nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 4, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(4), 
                                   nn.LeakyReLU(0.2))
        self.cam3 = ChannelAttention(4)
        self.conv4 = nn.Conv2d(4, P, 1, bias=False)

    def forward(self, x):
        h = self.cam1(self.conv1(x))
        h = self.conv2(h)
        h = self.cam3(self.conv3(h))
        h = self.conv4(h)
        return F.softmax(h, dim=1)

class EncoderSpe(nn.Module):
    # Spectral encoder (1x1 convs + Spatial attention)
    def __int__(self, L, P):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(L, 32, 1, bias=False),
                                   nn.BatchNorm2d(32), 
                                   nn.LeakyReLU(0.2))
        self.sam1 = SpatialAttention()
        self.conv2 = nn.Sequential(nn.Conv2d(32, 16, 1, bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.LeakyReLU(0.2))
        self.sam2 = SpatialAttention()
        self.conv3 = nn.Sequential(nn.Conv2d(16, 4, 1, bias=False),
                                   nn.BatchNorm2d(4), 
                                   nn.LeakyReLU(0.2))
        self.sam3 = SpatialAttention()
        self.conv4 = nn.Conv2d(4, P, 1, bias=False)

    def forward(self, x):
        h = self.sam1(self.conv1(x))
        h = self.sam2(self.conv2(h))
        h = self.sam3(self.conv3(h))
        h = self.conv4(h)
        return F.softmax(h, dim=1)
    
# EV-NET (VAE for per-pixel endmember variability)
class EVNet(nn.Module):
    # Endmember Variability Network
    # M_n = M0 @ psi_n + dM_n (Perturbed Prototype Model)
    def __init__(self, L, P, J=4, M0=None):
        super().__init__()
        self.L = L
        self.P = P
        self.J = J

        # Inference: pixel -> (mu_z, logvar_z)
        self.enc = nn.Sequential(
            nn.Linear(L, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 16), nn.LeakyReLU(0.2),
        )
        self.mu = nn.Linear(16, J)
        self.logv = nn.Linear(16, J)

        # Generative: z -> psi (PxP scaling), dM (Lxp perturbation)
        self.dec_psi = nn.Sequential(
            nn.Linear(J, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, P * P), nn.Softplus()
        )

        self.dec_dM = nn.Sequential(
            nn.Linear(J, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, L * P), nn.Sigmoid(),
        )

        if M0 is not None:
            self.register_buffer("M0", M0) # (L, P)
        else:
            self.register_buffer("M0", torch.eye(L, P))

    def reparameterise(self, mu, logv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logv)
    
    def forward(self, y): # y: (N, L)
        N = y.shape[0]
        h = self.enc(y)
        mu = self.mu(h)
        logv = self.logv(h)
        z = self.reparameterise(mu, logv) # (N, J)

        psi = self.dec_psi(z).view(N, self.P, self.P) # (N, P, P)
        dM = self.dec_dM(z).view(N, self.L, self.P)

        M0_exp = self.M0.unsqueeze(0).expand(N, -1, -1)
        Mn = torch.bmm(M0_exp, psi) + dM
        Mn = torch.clamp(Mn, 0, 1)
        return Mn, mu, logv

