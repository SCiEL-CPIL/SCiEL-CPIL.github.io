import numpy as np
from scipy.optimize import differential_evolution

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    # Sparse support grid. 
    # We want to maximize P(X1+X2+X3 < 2X4).
    # This favors distributions with mass at 0 and at large values.
    support = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float64)
    n = len(support)

    si = support[:, None, None, None]
    sj = support[None, :, None, None]
    sk = support[None, None, :, None]
    sl = support[None, None, None, :]
    cond = (si + sj + sk) < (2.0 * sl)
    cond_per_l = [cond[:, :, :, l] for l in range(n)]

    def compute_P(w):
        P = 0.0
        for l in range(n):
            # w is a vector of length n
            # inner sum: sum_{i,j} w_i w_j cond[i,j,l]
            # We can precompute matrix M_l where M_l[i,j] = cond[i,j,l]
            # Then inner = w @ M_l @ w
            # But we have cond_per_l as 3D array [i,j,k] -> wait, cond is [i,j,k,l]
            # cond_per_l[l] is [i,j,k]? No.
            # si [n,1,1,1], sj [1,n,1,1], sk [1,1,n,1], sl [1,1,1,n]
            # cond [n,n,n,n]
            # cond_per_l[l] is [n,n,n] corresponding to i,j,k for fixed l.
            # We need sum_{i,j,k} w_i w_j w_k cond[i,j,k].
            # This is a cubic form.
            # Let's use einsum: 'ijk,i,j,k->'
            inner_l = np.einsum('ijk,i,j,k->', cond_per_l[l], w, w, w)
            P += w[l] * inner_l
        return P

    def objective(params):
        w = np.zeros(n)
        w[:n-1] = params
        w[n-1] = 1.0 - np.sum(params)
        if np.any(w < -1e-9):
            return 1e9
        w = np.clip(w, 0, None)
        total = np.sum(w)
        if total <= 1e-9:
            return 1e9
        w = w / total
        return -compute_P(w)

    bounds = [(0, 1)] * (n - 1)
    
    # Differential Evolution for global optimization
    # popsize=15, maxiter=1000, tol=1e-8
    # seed for reproducibility
    try:
        res = differential_evolution(objective, bounds, seed=42, maxiter=2000, tol=1e-10, polish=True)
        best_params = res.x
        best_score_val = -res.fun
    except Exception:
        # Fallback to uniform if DE fails
        best_params = np.ones(n-1) / n
        best_score_val = -1.0

    best_w = np.zeros(n)
    best_w[:n-1] = best_params
    best_w[n-1] = 1.0 - np.sum(best_params)
    best_w = np.clip(best_w, 0, None)
    total = np.sum(best_w)
    if total > 1e-9:
        best_w = best_w / total
    else:
        best_w = np.ones(n) / n

    return support, best_w
