"""
ssaf_pipeline.py — Complete end-to-end Hyperspectral Unmixing with SSAF-Net
============================================================================

Single-file, no external model dependencies.

Usage
-----
  python ssaf_pipeline.py --mat jasper_ridge.mat
  python ssaf_pipeline.py --mat houston.mat     --dataset houston
  python ssaf_pipeline.py --mat dc2.mat         --dataset dc2
  python ssaf_pipeline.py --npy Y.npy           --p 5 --h 80 --w 80

What it does
------------
  1. Loads your hyperspectral cube (no GT needed)
  2. Estimates P automatically via HySime + PCA scree (saves plots)
  3. Initialises endmembers with VCA
  4. Trains SSAF-Net (spatial + spectral encoder + EV-Net VAE)
  5. Saves abundance maps, estimated endmember spectra, and all plots
  6. If GT is available, prints RMSE / SAD metrics
"""

import os
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# 0.  PER-DATASET PRESETS
# ════════════════════════════════════════════════════════════════════════════
PRESETS = {
    "jasper": dict(
        y_key="Y", gt_key="A", em_key="M",
        p=4, h=100, w=100,
        band_mask=np.r_[0:3, 107:112, 153:166, 219:224],
        epochs=2000, lr=5e-3,
    ),
    "houston": dict(
        y_key="Y", gt_key="A", em_key="M",
        p=4, h=170, w=170,
        band_mask=None,
        epochs=200, lr=5e-3,
    ),
    "apex": dict(
        y_key="Y", gt_key="A", em_key="M",
        p=4, h=110, w=110,
        band_mask=None,
        epochs=2000, lr=5e-3,
    ),
    "dc2": dict(
        y_key="Y", gt_key="A", em_key="M",
        p=3, h=50, w=50,
        band_mask=None,
        epochs=200, lr=5e-3,
    ),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ════════════════════════════════════════════════════════════════════════════
# 1.  P ESTIMATION  (HySime + PCA scree)
# ════════════════════════════════════════════════════════════════════════════

def estimate_noise(Y):
    """
    Per-band noise estimation via multiple regression.
    Y : (L, N)
    Returns noise matrix (L, N) and noise covariance (L, L).
    """
    L, N = Y.shape
    noise = np.zeros_like(Y)
    for b in range(L):
        others = np.delete(Y, b, axis=0)                # (L-1, N)
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
    _, Rn = estimate_noise(Y)
    Ry    = (Y @ Y.T) / N
    Rx    = Ry - Rn

    # Eigendecomposition of Ry (descending)
    vals_y, vecs = np.linalg.eigh(Ry)
    idx    = np.argsort(vals_y)[::-1]
    vecs   = vecs[:, idx]

    errors = []
    for k in range(1, L + 1):
        Ek         = vecs[:, :k]
        proj_err   = np.trace(Ry - Ek @ Ek.T @ Rx @ Ek @ Ek.T)
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
    Yc    = Y - Y.mean(axis=1, keepdims=True)
    cov   = (Yc @ Yc.T) / Y.shape[1]
    vals  = np.linalg.eigvalsh(cov)[::-1]          # descending
    vals  = np.maximum(vals, 0)
    ratio = vals[:max_p] / vals.sum()

    # Elbow: largest second-difference
    if len(ratio) >= 3:
        d2     = np.diff(np.diff(ratio))
        elbow  = int(np.argmax(d2)) + 2            # 1-indexed
    else:
        elbow  = len(ratio)

    return elbow, ratio


def estimate_p(Y, out_dir, max_p=30, verbose=True):
    """
    Runs HySime + PCA, saves combined plot, returns recommended P.

    Y       : (L, N)
    out_dir : directory to save 'p_estimation.png'
    """
    max_p = min(max_p, Y.shape[0] - 1, Y.shape[1] - 1)

    # ── HySime ──────────────────────────────────────────────────────────
    print("  Running HySime …", end=" ", flush=True)
    p_hysime, hy_errors = hysime(Y, verbose=False)
    print(f"P = {p_hysime}")

    # ── PCA scree ────────────────────────────────────────────────────────
    print("  Running PCA scree …", end=" ", flush=True)
    p_pca, pca_ratio = pca_scree(Y, max_p=max_p)
    pca_cum = np.cumsum(pca_ratio)
    print(f"P = {p_pca}")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Endmember Number Estimation", fontsize=13, fontweight="bold")

    # HySime error curve
    k_vals = np.arange(1, len(hy_errors) + 1)
    axes[0].plot(k_vals, hy_errors, color="#2d6a9f", lw=1.5)
    axes[0].axvline(p_hysime, color="#e74c3c", ls="--",
                    label=f"HySime → P = {p_hysime}")
    axes[0].set_xlabel("Subspace dimension k")
    axes[0].set_ylabel("Error criterion")
    axes[0].set_title("HySime")
    axes[0].legend()
    axes[0].set_xlim(1, min(40, len(hy_errors)))

    # PCA scree
    xs = np.arange(1, len(pca_ratio) + 1)
    axes[1].bar(xs, pca_ratio * 100, color="#2ecc71", alpha=0.7)
    axes[1].axvline(p_pca, color="#e74c3c", ls="--",
                    label=f"Elbow → P = {p_pca}")
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Explained Variance (%)")
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
    save_path = os.path.join(out_dir, "p_estimation.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")

    # ── Decision ─────────────────────────────────────────────────────────
    if p_hysime == p_pca:
        P = p_hysime
        note = "both methods agree"
    else:
        P = p_hysime        # HySime is more reliable for HSI
        note = f"HySime preferred (PCA said {p_pca})"

    print(f"\n  ✓ Using P = {P}  [{note}]")
    return P


# ════════════════════════════════════════════════════════════════════════════
# 2.  VCA  (endmember initialisation)
# ════════════════════════════════════════════════════════════════════════════

def vca(Y, p):
    """
    Vertex Component Analysis.
    Y : (L, N)   → returns M0 : (L, P) as float32 torch tensor
    """
    L, N = Y.shape
    R    = np.zeros((L, p), dtype=np.float32)

    for i in range(p):
        f = np.random.randn(N)
        if i > 0:
            U = np.linalg.svd(R[:, :i], full_matrices=False)[0]
            f = f - U @ (U.T @ Y) @ (np.linalg.pinv(Y) @ f)
        idx      = np.argmax(np.abs(Y.T @ f))
        R[:, i]  = Y[:, idx]

    return torch.tensor(R, dtype=torch.float32)


# ════════════════════════════════════════════════════════════════════════════
# 3.  ATTENTION MODULES
# ════════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    def __init__(self, channels, k=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)

    def forward(self, x):           # x: (B, C, H, W)
        avg     = x.mean(dim=[2, 3])
        mx      = x.amax(dim=[2, 3])
        avg_out = self.conv(avg.unsqueeze(1)).squeeze(1)
        mx_out  = self.conv(mx.unsqueeze(1)).squeeze(1)
        att     = torch.sigmoid(avg_out + mx_out)
        return x * att[:, :, None, None]


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k//2, bias=False)

    def forward(self, x):           # x: (B, C, H, W)
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


# ════════════════════════════════════════════════════════════════════════════
# 4.  ENCODERS
# ════════════════════════════════════════════════════════════════════════════

class EncoderSpa(nn.Module):
    """Spatial encoder: 3×3 convs + Channel Attention."""
    def __init__(self, L, P):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(L,  32, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
                                   nn.Dropout2d(0.1))
        self.cam1  = ChannelAttention(32)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(16), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(16,  4, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(4),  nn.LeakyReLU(0.2))
        self.cam3  = ChannelAttention(4)
        self.conv4 = nn.Conv2d(4, P, 1, bias=False)

    def forward(self, x):
        h = self.cam1(self.conv1(x))
        h = self.conv2(h)
        h = self.cam3(self.conv3(h))
        return F.softmax(self.conv4(h), dim=1)   # (B, P, H, W)


class EncoderSpe(nn.Module):
    """Spectral encoder: 1×1 convs + Spatial Attention."""
    def __init__(self, L, P):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(L,  32, 1, bias=False),
                                   nn.BatchNorm2d(32), nn.LeakyReLU(0.2))
        self.sam1  = SpatialAttention()
        self.conv2 = nn.Sequential(nn.Conv2d(32, 16, 1, bias=False),
                                   nn.BatchNorm2d(16), nn.LeakyReLU(0.2))
        self.sam2  = SpatialAttention()
        self.conv3 = nn.Sequential(nn.Conv2d(16,  4, 1, bias=False),
                                   nn.BatchNorm2d(4),  nn.LeakyReLU(0.2))
        self.sam3  = SpatialAttention()
        self.conv4 = nn.Conv2d(4, P, 1, bias=False)

    def forward(self, x):
        h = self.sam1(self.conv1(x))
        h = self.sam2(self.conv2(h))
        h = self.sam3(self.conv3(h))
        return F.softmax(self.conv4(h), dim=1)   # (B, P, H, W)


# ════════════════════════════════════════════════════════════════════════════
# 5.  EV-NET  (VAE for per-pixel endmember variability)
# ════════════════════════════════════════════════════════════════════════════

class EVNet(nn.Module):
    """
    Endmember Variability Network.
    Models M_n = M0 @ psi_n + dM_n  (Perturbed Prototype Model).
    """
    def __init__(self, L, P, J=4, M0=None):
        super().__init__()
        self.L, self.P, self.J = L, P, J

        # Inference: pixel → (mu_z, logvar_z)
        self.enc = nn.Sequential(
            nn.Linear(L, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 16), nn.LeakyReLU(0.2),
        )
        self.mu   = nn.Linear(16, J)
        self.logv = nn.Linear(16, J)

        # Generative: z → psi (P×P scaling), dM (L×P perturbation)
        self.dec_psi = nn.Sequential(
            nn.Linear(J, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, P * P), nn.Softplus(),
        )
        self.dec_dM = nn.Sequential(
            nn.Linear(J, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, L * P), nn.Sigmoid(),
        )

        if M0 is not None:
            self.register_buffer("M0", M0)   # (L, P)
        else:
            self.register_buffer("M0", torch.eye(L, P))

    def reparameterise(self, mu, logv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logv)

    def forward(self, y):          # y: (N, L)
        N = y.shape[0]
        h    = self.enc(y)
        mu   = self.mu(h)
        logv = self.logv(h)
        z    = self.reparameterise(mu, logv)   # (N, J)

        psi  = self.dec_psi(z).view(N, self.P, self.P)        # (N, P, P)
        dM   = self.dec_dM(z).view(N, self.L, self.P)         # (N, L, P)

        M0_exp = self.M0.unsqueeze(0).expand(N, -1, -1)       # (N, L, P)
        Mn     = torch.bmm(M0_exp, psi) + dM
        Mn     = torch.clamp(Mn, 0, 1)

        return Mn, mu, logv


# ════════════════════════════════════════════════════════════════════════════
# 6.  SSAF-NET
# ════════════════════════════════════════════════════════════════════════════

class SSAFNet(nn.Module):
    def __init__(self, L, P, J=4, M0=None):
        super().__init__()
        self.L, self.P = L, P
        self.enc_spa = EncoderSpa(L, P)
        self.enc_spe = EncoderSpe(L, P)
        self.ev_net  = EVNet(L, P, J, M0)

    def decode(self, A, Mn):
        """
        A  : (B, P, H, W)
        Mn : (N, L, P)   N = B*H*W
        → (B, L, H, W)
        """
        B, _, H, W = A.shape
        N = B * H * W
        a     = A.permute(0, 2, 3, 1).reshape(N, self.P, 1)   # (N, P, 1)
        y_hat = torch.bmm(Mn, a).squeeze(-1)                   # (N, L)
        return y_hat.view(B, H, W, self.L).permute(0, 3, 1, 2)# (B, L, H, W)

    def forward(self, x):           # x: (B, L, H, W)
        B, _, H, W = x.shape
        N = B * H * W

        A1              = self.enc_spa(x)
        y_flat          = x.permute(0, 2, 3, 1).reshape(N, self.L)
        Mn, mu, logv    = self.ev_net(y_flat)
        Y1              = self.decode(A1, Mn)
        A2              = self.enc_spe(Y1)
        Y2              = self.decode(A2, Mn)

        return Y1, Y2, A1, A2, Mn, mu, logv


# ════════════════════════════════════════════════════════════════════════════
# 7.  LOSS FUNCTIONS  (all self-supervised, no GT needed)
# ════════════════════════════════════════════════════════════════════════════

def kl_loss(mu, logv):
    return -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())


def sad_loss(Mn):
    """Angular spread of per-pixel endmembers around their mean — no GT."""
    mean_m = Mn.mean(dim=0, keepdim=True)
    cos    = F.cosine_similarity(Mn, mean_m.expand_as(Mn), dim=1)
    return torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)).mean()


def vol_loss(Mn):
    """Endmember compactness — no GT."""
    mean_p = Mn.mean(dim=2, keepdim=True)
    return ((Mn - mean_p) ** 2).mean()


def total_loss(Y, Y1, Y2, mu, logv, Mn,
               alpha=0.5, lam_kl=0.1, lam_sad=0.1, lam_vol=0.1):
    rec = alpha * F.mse_loss(Y1, Y) + (1 - alpha) * F.mse_loss(Y2, Y)
    return rec + lam_kl * kl_loss(mu, logv) \
               + lam_sad * sad_loss(Mn) \
               + lam_vol * vol_loss(Mn)


# ════════════════════════════════════════════════════════════════════════════
# 8.  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_data(args, preset):
    """
    Returns Y_np (L, H, W), A_true (P, N) or None, M_true (L, P) or None.
    """
    import scipy.io as sio

    p         = args.p  or preset.get("p")
    h         = args.h  or preset.get("h")
    w         = args.w  or preset.get("w")
    y_key     = args.y_key  or preset.get("y_key",  "Y")
    gt_key    = args.gt_key or preset.get("gt_key", "A")
    em_key    = args.em_key or preset.get("em_key", "M")
    band_mask = preset.get("band_mask", None)

    assert p and h and w, \
        "Specify --p, --h, --w or use a named --dataset preset."

    # ── Load raw array ────────────────────────────────────────────────────
    if args.mat:
        mat = sio.loadmat(args.mat)
        Y   = mat[y_key].astype(np.float32)
        A_true = mat[gt_key].astype(np.float32) if gt_key in mat else None
        M_true = mat[em_key].astype(np.float32) if em_key in mat else None
    elif args.npy:
        Y      = np.load(args.npy).astype(np.float32)
        A_true = np.load(args.npy_a).astype(np.float32) if args.npy_a else None
        M_true = np.load(args.npy_m).astype(np.float32) if args.npy_m else None
    else:
        raise ValueError("Provide --mat or --npy.")

    # ── Shape normalisation → (L, H, W) ──────────────────────────────────
    if Y.ndim == 3:
        if Y.shape[0] == h and Y.shape[1] == w:
            Y = Y.transpose(2, 0, 1)          # (H, W, L) → (L, H, W)
        # else assume already (L, H, W)
    elif Y.ndim == 2:
        # (L, N) or (N, L)
        if Y.shape[1] == h * w:
            Y = Y.reshape(-1, h, w)
        elif Y.shape[0] == h * w:
            Y = Y.T.reshape(-1, h, w)
        else:
            raise ValueError(f"Cannot reshape Y {Y.shape} to ({h},{w}).")

    # ── Band removal ──────────────────────────────────────────────────────
    if band_mask is not None:
        keep = np.setdiff1d(np.arange(Y.shape[0]), band_mask)
        Y    = Y[keep]

    # ── Normalise to [0, 1] ───────────────────────────────────────────────
    vmax = Y.max()
    if vmax > 0:
        Y /= vmax

    L = Y.shape[0]
    N = h * w

    # ── GT shape fixes ────────────────────────────────────────────────────
    if A_true is not None:
        try:
            A_true = A_true.reshape(p, N)
        except ValueError:
            A_true = A_true.T.reshape(p, N)

    if M_true is not None:
        try:
            M_true = M_true.reshape(L, p)
        except ValueError:
            M_true = M_true.T.reshape(L, p)

    print(f"  Cube   : L={L}, H={h}, W={w}, N={N}")
    print(f"  A_true : {'yes' if A_true is not None else 'no'}")
    print(f"  M_true : {'yes' if M_true is not None else 'no'}")
    return Y, A_true, M_true, p, h, w, L


# ════════════════════════════════════════════════════════════════════════════
# 9.  METRICS
# ════════════════════════════════════════════════════════════════════════════

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def sad_metric(m1, m2):
    """m1, m2: (L,) vectors"""
    cos = np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-10)
    return float(np.arccos(np.clip(cos, -1, 1)))


def align_and_eval(A_hat, M_hat_mean, A_true, M_true):
    """
    Hungarian-style greedy alignment of estimated vs. true endmembers.
    A_hat      : (P, N)
    M_hat_mean : (L, P)  — mean of per-pixel endmembers
    A_true     : (P, N)
    M_true     : (L, P) or None
    Returns dict of metrics.
    """
    P  = A_hat.shape[0]
    # Build cost matrix on abundance RMSE
    cost = np.array([[rmse(A_hat[i], A_true[j])
                      for j in range(P)] for i in range(P)])
    order = np.argmin(cost, axis=1)             # greedy row→col assignment

    A_aligned = A_hat[order]
    rmse_a    = rmse(A_aligned, A_true)

    out = dict(rmse_a=rmse_a, order=order)

    if M_true is not None:
        M_aligned = M_hat_mean[:, order]        # (L, P)
        sad_vals  = [sad_metric(M_aligned[:, i], M_true[:, i])
                     for i in range(P)]
        out["rmse_m"] = rmse(M_aligned, M_true)
        out["sad_m"]  = float(np.mean(sad_vals))
        out["sad_per_em"] = sad_vals

    return out


# ════════════════════════════════════════════════════════════════════════════
# 10.  PLOTTING  (saves all figures, never shows)
# ════════════════════════════════════════════════════════════════════════════

def save_abundance_maps(A_hat, P, out_dir, name="abundance_maps.png"):
    fig, axes = plt.subplots(1, P, figsize=(4 * P, 3.5))
    if P == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        im = ax.imshow(A_hat[i].reshape(-1, A_hat.shape[-1])
                       if A_hat.ndim == 2 else A_hat[i],
                       cmap="jet", vmin=0, vmax=1)
        ax.set_title(f"EM #{i+1}", fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("Estimated Abundance Maps", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def save_endmember_spectra(M_mean, P, L, out_dir, M_true=None,
                           name="endmember_spectra.png"):
    fig, axes = plt.subplots(1, P, figsize=(4 * P, 3), sharey=True)
    if P == 1:
        axes = [axes]
    wl = np.arange(L)
    for i, ax in enumerate(axes):
        ax.plot(wl, M_mean[:, i], color="#2d6a9f", lw=1.5, label="Estimated")
        if M_true is not None:
            ax.plot(wl, M_true[:, i], color="#e74c3c", lw=1.2,
                    ls="--", label="GT")
        ax.set_title(f"EM #{i+1}")
        ax.set_xlabel("Band")
        if i == 0:
            ax.set_ylabel("Reflectance")
        ax.legend(fontsize=7)
    plt.suptitle("Endmember Spectra", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def save_loss_curve(losses, out_dir, name="loss_curve.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color="#2d6a9f", lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Training Loss Curve")
    ax.set_yscale("log")
    plt.tight_layout()
    path = os.path.join(out_dir, name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# 11.  TRAIN
# ════════════════════════════════════════════════════════════════════════════

def train(Y_np, P, H, W, L, out_dir,
          epochs    = 2000,
          lr        = 5e-3,
          J         = 4,
          alpha     = 0.5,
          lam_kl    = 0.1,
          lam_sad   = 0.1,
          lam_vol   = 0.1):
    """
    Y_np : (L, H, W)  float32, values in [0,1]
    Returns A_hat (P, H, W), M_mean (L, P), loss_history (list)
    """
    # ── VCA init ────────────────────────────────────────────────────────
    print("\n[3] Initialising endmembers with VCA …")
    Y_flat = Y_np.reshape(L, -1)                               # (L, N)
    M0     = vca(Y_flat, P).to(DEVICE)                         # (L, P)

    # ── Model ────────────────────────────────────────────────────────────
    model = SSAFNet(L=L, P=P, J=J, M0=M0).to(DEVICE)

    # Kaiming init
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    Y_tensor = torch.tensor(Y_np).unsqueeze(0).to(DEVICE)     # (1, L, H, W)

    losses = []
    print(f"\n[4] Training SSAF-Net for {epochs} epochs on {DEVICE} …")
    tic = time()

    for epoch in tqdm(range(1, epochs + 1), desc="Training"):
        model.train()
        Y1, Y2, A1, A2, Mn, mu, logv = model(Y_tensor)
        loss = total_loss(Y_tensor, Y1, Y2, mu, logv, Mn,
                          alpha, lam_kl, lam_sad, lam_vol)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))

    toc = time()
    print(f"  Done in {toc - tic:.1f}s  |  Final loss: {losses[-1]:.6f}")

    # ── Extract results ───────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        _, _, A1, _, Mn, _, _ = model(Y_tensor)

    A_hat  = A1.squeeze(0).cpu().numpy()       # (P, H, W)
    Mn_np  = Mn.cpu().numpy()                  # (N, L, P)
    M_mean = Mn_np.mean(axis=0)                # (L, P)  — mean endmember

    return A_hat, M_mean, losses


# ════════════════════════════════════════════════════════════════════════════
# 12.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    preset  = PRESETS.get(args.dataset, {})
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Load data ────────────────────────────────────────────────
    print("\n[1] Loading data …")
    Y_np, A_true, M_true, P_preset, H, W, L = load_data(args, preset)
    N = H * W

    # ── Step 2: Estimate P (if not forced) ───────────────────────────────
    if args.p:
        P = args.p
        print(f"\n[2] Using user-specified P = {P}")
    elif P_preset:
        P = P_preset
        print(f"\n[2] Using preset P = {P}  (running estimation anyway for reference)")
        estimate_p(Y_np.reshape(L, -1), out_dir, max_p=min(30, L - 1))
    else:
        print("\n[2] Estimating P …")
        P = estimate_p(Y_np.reshape(L, -1), out_dir, max_p=min(30, L - 1))

    epochs = args.epochs or preset.get("epochs", 2000)
    lr     = args.lr     or preset.get("lr",     5e-3)

    # ── Steps 3-4: Train ─────────────────────────────────────────────────
    A_hat, M_mean, losses = train(
        Y_np, P, H, W, L, out_dir,
        epochs=epochs, lr=lr,
        J=args.z_dim, lam_kl=args.lam_kl,
        lam_sad=args.lam_sad, lam_vol=args.lam_vol,
    )

    # ── Step 5: Save plots ───────────────────────────────────────────────
    print("\n[5] Saving outputs …")
    save_loss_curve(losses, out_dir)
    save_abundance_maps(A_hat, P, out_dir)
    save_endmember_spectra(M_mean, P, L, out_dir, M_true)

    # ── Step 6: Metrics (if GT available) ────────────────────────────────
    if A_true is not None:
        print("\n[6] Evaluating …")
        A_hat_flat = A_hat.reshape(P, N)
        metrics    = align_and_eval(A_hat_flat, M_mean, A_true, M_true)

        print("=" * 55)
        print(f"  Dataset : {args.dataset}")
        print(f"  aRMSE_A : {metrics['rmse_a']:.6f}")
        if "rmse_m" in metrics:
            print(f"  aRMSE_M : {metrics['rmse_m']:.6f}")
            print(f"  aSAD_M  : {metrics['sad_m']:.6f}  (rad)")
            for i, s in enumerate(metrics["sad_per_em"]):
                print(f"    EM #{i+1}: SAD = {s:.4f} rad")
        print("=" * 55)

        # Save GT vs estimated abundance comparison
        if A_true is not None:
            fig, axes = plt.subplots(2, P, figsize=(4 * P, 7))
            order = metrics["order"]
            for i in range(P):
                axes[0, i].imshow(A_hat[order[i]].reshape(H, W),
                                  cmap="jet", vmin=0, vmax=1)
                axes[0, i].set_title(f"Est. EM #{i+1}")
                axes[0, i].axis("off")
                axes[1, i].imshow(A_true[i].reshape(H, W),
                                  cmap="jet", vmin=0, vmax=1)
                axes[1, i].set_title(f"GT EM #{i+1}")
                axes[1, i].axis("off")
            plt.suptitle("Estimated (top) vs GT (bottom) Abundances",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            cmp_path = os.path.join(out_dir, "abundance_comparison.png")
            plt.savefig(cmp_path, dpi=150)
            plt.close()
            print(f"  Saved → {cmp_path}")
    else:
        print("\n[6] No GT available — skipping metrics.")

    # ── Save numpy arrays ─────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "A_hat.npy"),  A_hat)
    np.save(os.path.join(out_dir, "M_mean.npy"), M_mean)
    print(f"\n  A_hat  saved → {out_dir}/A_hat.npy   shape {A_hat.shape}")
    print(f"  M_mean saved → {out_dir}/M_mean.npy  shape {M_mean.shape}")
    print(f"\n✓ All outputs in: {out_dir}/")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def get_args():
    pa = argparse.ArgumentParser(
        description="SSAF-Net: end-to-end hyperspectral unmixing")

    # Data
    pa.add_argument("--dataset",  default="custom",
                    help="Named preset: jasper|houston|apex|dc2  (default: custom)")
    pa.add_argument("--mat",      default=None, help=".mat file path")
    pa.add_argument("--npy",      default=None, help="Y .npy file path")
    pa.add_argument("--npy_a",    default=None, help="A_true .npy (optional)")
    pa.add_argument("--npy_m",    default=None, help="M_true .npy (optional)")
    pa.add_argument("--y_key",    default=None, help="Key for Y in .mat")
    pa.add_argument("--gt_key",   default=None, help="Key for A in .mat")
    pa.add_argument("--em_key",   default=None, help="Key for M in .mat")

    # Dimensions (override preset)
    pa.add_argument("--p",   type=int,   default=None,
                    help="Num endmembers. If omitted, estimated automatically.")
    pa.add_argument("--h",   type=int,   default=None, help="Image height")
    pa.add_argument("--w",   type=int,   default=None, help="Image width")

    # Training
    pa.add_argument("--epochs",   type=int,   default=None)
    pa.add_argument("--lr",       type=float, default=None)
    pa.add_argument("--z_dim",    type=int,   default=4,
                    help="Latent dim for EV-Net VAE (default 4)")
    pa.add_argument("--lam_kl",   type=float, default=0.1)
    pa.add_argument("--lam_sad",  type=float, default=0.1)
    pa.add_argument("--lam_vol",  type=float, default=0.1)

    # Output
    pa.add_argument("--out_dir",  default="results",
                    help="Folder for all saved plots and arrays")

    return pa.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main(get_args())