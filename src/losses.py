import torch
import torch.nn.functional as F

# Loss functions
def kl_loss(mu, logv):
    return -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())

def sad_loss(Mn):
    # Angular spread of per-pixel endmembers around their mean — no GT
    mean_m = Mn.mean(dim=0, keepdim=True)
    cos = F.cosine_similarity(Mn, mean_m.expand_as(Mn), dim=1)
    return torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)).mean()

def vol_loss(Mn):
    # Endmember compactness - no GT
    mean_p = Mn.mean(dim=2, keepdim=True)
    return ((Mn - mean_p) ** 2).mean()

def total_loss(Y, Y1, Y2, mu, logv, Mn, alpha=0.5, lam_kl=0.1, lam_sad=0.1, lam_vol=0.1):
    rec = alpha * F.mse_loss(Y1, Y) + (1 - alpha) * F.mse_loss(Y2, Y)
    return rec + lam_kl * kl_loss(mu, logv) + lam_sad * sad_loss(Mn) + lam_vol * vol_loss(Mn)