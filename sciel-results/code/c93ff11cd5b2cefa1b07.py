import numpy as np
from scipy.optimize import minimize
import time

def compute_union_length_scalar(c, X):
    N = 8
    i_vals = np.arange(N)
    starts = c * (i_vals / N) + X
    ends = starts + 1.0 / N
    order = np.argsort(starts)
    s_sorted = starts[order]
    e_sorted = ends[order]
    
    total_len = 0.0
    current_start = s_sorted[0]
    current_end = e_sorted[0]
    for m in range(1, N):
        if s_sorted[m] > current_end:
            total_len += current_end - current_start
            current_start = s_sorted[m]
            current_end = e_sorted[m]
        else:
            current_end = max(current_end, e_sorted[m])
    total_len += current_end - current_start
    return total_len

def exact_objective(X):
    N = 8
    c_pts = [0.0, 1.0]
    for i in range(N):
        for j in range(N):
            if i == j: continue
            diff = (j - i) / N
            if diff == 0: continue
            # Left-Left
            c1 = (X[j] - X[i]) / diff
            if 0 <= c1 <= 1: c_pts.append(c1)
            # Left-Right
            c2 = (X[j] - X[i] + 1/N) / diff
            if 0 <= c2 <= 1: c_pts.append(c2)
            # Right-Left
            c3 = (X[j] - X[i] - 1/N) / diff
            if 0 <= c3 <= 1: c_pts.append(c3)
            # Right-Right
            c4 = (X[j] - X[i]) / diff
            if 0 <= c4 <= 1: c_pts.append(c4)
            
    c_pts = np.array(sorted(set(c_pts)))
    
    total_vol = 0.0
    for k in range(len(c_pts) - 1):
        c0 = c_pts[k]
        c1 = c_pts[k+1]
        if c1 - c0 < 1e-12: continue
        L0 = compute_union_length_scalar(c0, X)
        L1 = compute_union_length_scalar(c1, X)
        cmid = (c0 + c1) / 2
        Lmid = compute_union_length_scalar(cmid, X)
        total_vol += (c1 - c0) / 6.0 * (L0**2 + 4.0*Lmid**2 + L1**2)
        
    return total_vol

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    best_X = None
    best_val = np.inf
    
    start_time = time.time()
    timeout = 600.0
    margin = 30.0
    
    inits = []
    # 1. Keich
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    inits.append(keich)
    
    # 2. Random
    rng = np.random.default_rng(42)
    for _ in range(10):
        inits.append(rng.uniform(-1, 1, N))
        
    # 3. Zero
    inits.append(np.zeros(N))
    
    for X0 in inits:
        if time.time() - start_time > timeout - margin:
            break
            
        try:
            res = minimize(exact_objective, X0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_X = res.x.copy()
        except Exception:
            continue
            
    if best_X is None:
        best_X = np.zeros(N)
        
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_X[i_flat]
    base_y = best_X[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
