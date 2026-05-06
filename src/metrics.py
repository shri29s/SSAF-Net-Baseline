import numpy as np

# Metrics
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def sad_metric(m1, m2):
    # m1, m2: (L, ) vectors
    cos = np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-10)
    return float(np.arccos(np.clip(cos, -1 + 1e-6, 1 - 1e-6)))

def align_and_eval(A_hat, M_hat_mean, A_true, M_true):
    """
    Hungarian-style greedy alignment of estimated vs. true endmembers.
    A_hat      : (P, N)
    M_hat_mean : (L, P)  — mean of per-pixel endmembers
    A_true     : (P, N)
    M_true     : (L, P) or None
    Returns dict of metrics.
    """
    P = A_hat.shape[0]
    # Build cost matrix on abundance RMSE
    cost = np.array([[rmse(A_hat[i], A_true[j]) for j in range(P)] for i in range(P)])
    order = np.argmin(cost, axis=1)

    A_aligned = A_hat[order]
    rmse_a = rmse(A_aligned, A_true)
    out = dict(rmse_a=rmse_a, order=order)

    if M_true is not None:
        M_aligned = M_hat_mean[:, order]
        sad_vals = [sad_metric(M_aligned[:, i], M_true[:, i]) for i in range(P)]
        out["rmse_m"] = rmse(M_aligned, M_true)
        out["sad_m"] = float(np.mean(sad_vals))
        out["sad_per_em"] = sad_vals
    return out