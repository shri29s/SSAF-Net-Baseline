# Data loading
from .utils import Logger
import scipy.io as sio
import numpy as np

def load_data(args, preset, logger: Logger = None):
    # Returns Y_np (L, H, W), A_true (P, N) or None, M_true (L, P) or None
    p = args.p or preset.get("p")
    h = args.h or preset.get("h")
    w = args.w or preset.get("w")

    y_key = args.y_key or preset.get("y_key", "Y")
    gt_key = args.gt_key or preset.get("gt_key", "A")
    em_key = args.em_key or preset.get("em_key", "M")
    band_mask = preset.get("band_mask", None)

    if h is None or w is None:
        raise ValueError("Specify --h and --w or use a named --dataset preset.")

    if args.mat:
        mat = sio.loadmat(args.mat)
        Y = mat[y_key].astype(np.float32)
        A_true = mat[gt_key].astype(np.float32) if gt_key in mat else None
        M_true = mat[em_key].astype(np.float32) if em_key in mat else None
    elif args.npy:
        Y = np.load(args.npy).astype(np.float32)
        A_true = np.load(args.npy_a).astype(np.float32) if args.npy_a else None
        M_true = np.load(args.npy_m).astype(np.float32) if args.npy_m else None
    else:
        raise ValueError("Provide --mat or --npy")
    
    # Shape Normalization
    if Y.ndim == 3:
        if Y.shape[0] == h and Y.shape[1] == w:
            Y = Y.transpose(2, 0, 1) # (L, H, W) → (L, H, W)
    elif Y.ndim == 2:
        if Y.shape[1] == h * w:
            Y = Y.reshape(-1, h, w) # (L, N) → (L, H, W)
        elif Y.shape[0] == h * w:
            Y = Y.T.reshape(-1, h, w)
        else:
            raise ValueError(f"Cannot reshape Y of shape {Y.shape} to (L, H, W) with H={h}, W={w}")
        
    # Band masking
    if band_mask is not None:
        keep = np.setdiff1d(np.arange(Y.shape[0]), band_mask)
        Y = Y[keep]

    # Normalization to [0, 1]
    vmax = Y.max()
    if vmax > 0:
        Y = Y / vmax

    L = Y.shape[0]
    N = h * w

    # GT shape fixes
    if A_true is not None:
        if p is None:
            raise ValueError("Provide --p when abundance ground truth is present.")
        try:
            A_true = A_true.reshape(p, N)
        except ValueError:
            A_true = A_true.T.reshape(p, N)

    if M_true is not None:
        if p is None:
            raise ValueError("Provide --p when endmember ground truth is present.")
        try:
            M_true = M_true.reshape(L, p)
        except ValueError:
            M_true = M_true.T.reshape(L, p) 

    if logger:
        logger.info(f"Cube dimensions: L={L}, H={h}, W={w}, N={N}")
        logger.info(f"GT available - Abundance: {'yes' if A_true is not None else 'no'}, Endmembers: {'yes' if M_true is not None else 'no'}")
    return Y, A_true, M_true, p, h, w, L