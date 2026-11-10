# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _union_length_vec(starts):
    """
    Compute total length of union of 1-D intervals [starts[i], starts[i]+1/N].
    Vectorized merge-sweep.
    """
    n = len(starts)
    if n == 0:
        return 0.0
    order = np.argsort(starts)
    s_sorted = starts[order]
    e_sorted = s_sorted + 0.125  # 1/8 width
    
    # Sweep: track current rightmost end
    # Contribution of each interval is max(0, e_i - max(s_i, prev_end))
    # but we need the merged union length.
    # Standard sweep:
    total = 0.0
    cs = s_sorted[0]
    ce = e_sorted[0]
    for k in range(1, n):
        if s_sorted[k] > ce:
            total += ce - cs
            cs = s_sorted[k]
            ce = e_sorted[k]
        else:
            if e_sorted[k] > ce:
                ce = e_sorted[k]
    total += ce - cs
    return total


def _exact_volume(pos_x, pos_y, N=8, n_c=300):
    """
    Exact union volume via product structure: integral of Lx(c)*Ly(c) dc.
    Vectorized over c for speed.
    """
    invN = 1.0 / N
    c_samples = np.linspace(0, 1, n_c)
    # c_samples shape: (n_c,)
    # i_indices: (N,)
    i_idx = np.arange(N)
    j_idx = np.arange(N)
    
    # x_starts[c, i] = pos_x[i] + c * i_idx[i] * invN
    # Shape: (n_c, N)
    x_starts = pos_x[np.newaxis, :] + c_samples[:, np.newaxis] * (i_idx[np.newaxis, :] * invN)
    y_starts = pos_y[np.newaxis, :] + c_samples[:, np.newaxis] * (j_idx[np.newaxis, :] * invN)
    
    # Compute union length for each c
    # For each c, sort x_starts and merge
    Lx = np.zeros(n_c)
    Ly = np.zeros(n_c)
    
    for k in range(n_c):
        Lx[k] = _union_length_vec(x_starts[k])
        Ly[k] = _union_length_vec(y_starts[k])
    
    vol = np.sum(Lx * Ly) * (1.0 / n_c)
    return vol


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 620.0  # Safety margin from 630s limit

    # Keich binary construction as initial guess
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)

    best_x = keich.copy()
    best_y = keich.copy()
    best_vol = _exact_volume(best_x, best_y, N, n_c=200)
    
    # Track best found
    best_params = np.concatenate([best_x, best_y])

    def objective(params):
        if time.time() > deadline - 5:
            return best_vol
        px = params[:N]
        py = params[N:]
        return _exact_volume(px, py, N, n_c=200)

    rng = np.random.default_rng(12345)

    # Phase 1: Coarse search with Nelder-Mead
    # Multiple restarts with decreasing scales
    n_restarts = 0
    while time.time() < deadline - 120:
        if n_restarts < 10:
            scale = 0.5
        elif n_restarts < 30:
            scale = 0.1
        else:
            scale = 0.01
        
        init = np.concatenate([best_x, best_y]) + rng.normal(0, scale, 2 * N)
        
        result = minimize(
            objective, init, method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                     'adaptive': True, 'maxfev': 3000}
        )
        
        if result.fun < best_vol:
            best_vol = result.fun
            best_x = result.x[:N].copy()
            best_y = result.x[N:].copy()
            best_params = result.x.copy()
        
        n_restarts += 1

    # Phase 2: Fine-tuning with higher resolution c-sampling
    if time.time() < deadline - 30:
        def fine_objective(params):
            if time.time() > deadline - 5:
                return best_vol
            px = params[:N]
            py = params[N:]
            return _exact_volume(px, py, N, n_c=500)
        
        # A few fine-tuning steps
        for ft in range(3):
            if time.time() > deadline - 15:
                break
            result = minimize(
                fine_objective, best_params, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14,
                         'adaptive': True, 'maxfev': 5000}
            )
            if result.fun < best_vol:
                best_vol = result.fun
                best_x = result.x[:N].copy()
                best_y = result.x[N:].copy()
                best_params = result.x.copy()

    # Construct output
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
