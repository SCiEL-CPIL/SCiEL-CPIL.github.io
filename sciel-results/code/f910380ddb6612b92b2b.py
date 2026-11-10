import numpy as np
import time
from scipy.optimize import minimize

def _union_length_fast(intervals):
    """
    intervals: np.array of shape (N, 2) with columns [left, right]
    returns: length of the union of these intervals
    """
    N = intervals.shape[0]
    if N == 0:
        return 0.0
    
    lefts = intervals[:, 0]
    rights = intervals[:, 1]
    
    events = []
    for i in range(N):
        events.append((lefts[i], 1))
        events.append((rights[i], -1))
    
    events.sort(key=lambda x: x[0])
    
    total_len = 0.0
    active = 0
    prev_coord = events[0][0]
    
    for coord, delta in events:
        if active > 0:
            total_len += coord - prev_coord
        active += delta
        prev_coord = coord
        
    return total_len

def _objective(X):
    """
    X: np.array of shape (8,)
    Returns approximate volume integral
    """
    N = 8
    M = 500
    c_vals = np.linspace(0, 1, M)
    
    lengths = np.zeros(M)
    
    i_fracs = np.arange(N) / N
    
    for idx, c in enumerate(c_vals):
        lefts = X + c * i_fracs
        rights = lefts + 1.0 / N
        intervals = np.stack([lefts, rights], axis=1)
        lengths[idx] = _union_length_fast(intervals)
        
    if M > 1:
        integral = (0.5 * lengths[0] + np.sum(lengths[1:-1]) + 0.5 * lengths[-1]) * (1.0 / (M - 1))
    else:
        integral = lengths[0]
        
    return integral

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    best_X = keich.copy()
    best_val = _objective(best_X)
    
    start_time = time.time()
    deadline = 600 - 60  # 540s safety margin
    
    rng = np.random.default_rng(42)
    
    bounds = [(-10, 10) for _ in range(N)]
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        if restart == 0:
            X0 = keich.copy()
        else:
            X0 = keich + rng.normal(0, 0.1, size=N)
            
        res = minimize(_objective, X0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success and np.all(np.isfinite(res.x)):
            if res.fun < best_val:
                best_val = res.fun
                best_X = res.x.copy()
                
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_X[i_flat]
    base_y = best_X[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    if not np.all(np.isfinite(positions)):
        # Fallback to keich
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        
    return positions
