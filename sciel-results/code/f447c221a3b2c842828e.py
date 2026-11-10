# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_1d_union_length(p):
    """Compute the total length of the union of intervals
    [p[i] + c*i/N, p[i] + 1/N + c*i/N] for c in [0,1].
    
    Each interval in c is [p[i], p[i] + 1/N + i/N] at c=0 to
    [p[i] + i/N, p[i] + 1/N + 2*i/N] at c=1.
    The union over c of these moving intervals equals the union of
    the "envelope" which is the union of all [p[i] + t*i/N, p[i] + 1/N + t*i/N]
    for t in [0,1], which is just the interval [p[i], p[i] + 1/N + i/N]
    unioned with [p[i] + i/N, p[i] + 1/N + 2*i/N] ... 
    
    Actually, for a fixed i, as c varies from 0 to 1, the interval
    [p[i] + c*i/N, p[i] + 1/N + c*i/N] sweeps from
    [p[i], p[i]+1/N] to [p[i]+i/N, p[i]+1/N+i/N].
    The union over c in [0,1] of these intervals is
    [p[i], p[i] + 1/N + i/N] (the full swept region).
    
    So the 1-D union length is the length of the union of intervals
    [p[i], p[i] + (1+i)/N] for i = 0, ..., N-1.
    """
    N = 8
    starts = p.copy()
    ends = p + (1.0 + np.arange(N)) / N
    # Merge intervals
    order = np.argsort(starts)
    s = starts[order]
    e = ends[order]
    total = 0.0
    cur_s = s[0]
    cur_e = e[0]
    for i in range(1, N):
        if s[i] <= cur_e:
            cur_e = max(cur_e, e[i])
        else:
            total += cur_e - cur_s
            cur_s = s[i]
            cur_e = e[i]
    total += cur_e - cur_s
    return total


def _optimize_1d(deadline):
    """Optimize 1-D base positions to minimize union length."""
    N = 8
    best_p = None
    best_len = np.inf
    
    # Multiple random restarts
    rng = np.random.default_rng(42)
    n_restarts = 30
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initial guess in [0, 1]
        p0 = rng.uniform(0, 1, N)
        
        res = minimize(
            _compute_1d_union_length,
            p0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        if res.fun < best_len:
            best_len = res.fun
            best_p = res.x.copy()
    
    return best_p, best_len


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Returns positions of shape (64, 2) minimizing the union volume.
    """
    start_time = time.time()
    deadline = start_time + 600  # safety margin before 630s kill
    
    N = 8
    
    # Optimize 1-D problem for x and y (they're identical by symmetry)
    p_opt, opt_len = _optimize_1d(deadline)
    
    if p_opt is None:
        # Fallback: Keich construction
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        p_opt = keich
    
    # Build the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = p_opt[i_flat]
    base_y = p_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
