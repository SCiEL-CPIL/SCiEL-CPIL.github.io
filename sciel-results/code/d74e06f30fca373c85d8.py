import numpy as np
import time
from scipy.optimize import minimize

def _merge_intervals(Ls, Rs):
    """Merge intervals and return (merged_list, total_length, left_indices, right_indices)."""
    n = len(Ls)
    intervals = [(Ls[i], Rs[i], i, i) for i in range(n)]
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    cur_l, cur_r, cur_li, cur_ri = intervals[0]
    for l, r, li, ri in intervals[1:]:
        if l > cur_r + 1e-15:
            merged.append((cur_l, cur_r, cur_li, cur_ri))
            cur_l, cur_r, cur_li, cur_ri = l, r, li, ri
        else:
            if r > cur_r:
                cur_r = r
                cur_ri = ri
    merged.append((cur_l, cur_r, cur_li, cur_ri))
    
    total_len = sum(r - l for l, r, _, _ in merged)
    return merged, total_len

def _get_breakpoints(X, N=8):
    """Calculate all c in (0, 1) where the order of endpoints changes."""
    breakpoints = set([0.0, 1.0])
    for i in range(N):
        for j in range(i + 1, N):
            denom = float(i - j)
            c1 = N * (X[j] - X[i]) / denom
            if 0.0 < c1 < 1.0:
                breakpoints.add(c1)
            c2 = N * (X[j] + 1.0/N - X[i]) / denom
            if 0.0 < c2 < 1.0:
                breakpoints.add(c2)
            c3 = N * (X[j] - X[i] - 1.0/N) / denom
            if 0.0 < c3 < 1.0:
                breakpoints.add(c3)
            c4 = N * (X[j] - X[i]) / denom
            if 0.0 < c4 < 1.0:
                breakpoints.add(c4)
    return sorted(breakpoints)

def _compute_1d_cost_squared(X, N=8):
    """
    Compute integral of (Union Length)^2 over c in [0, 1].
    """
    if not np.all(np.isfinite(X)):
        return 1e10
    
    breakpoints = _get_breakpoints(X, N)
    
    total_integral = 0.0
    
    for k in range(len(breakpoints) - 1):
        c_start = breakpoints[k]
        c_end = breakpoints[k + 1]
        if c_end - c_start < 1e-15:
            continue
            
        c_mid = 0.5 * (c_start + c_end)
        
        Ls = np.array([X[i] + (i / N) * c_mid for i in range(N)])
        Rs = np.array([X[i] + (i / N) * c_mid + 1.0 / N for i in range(N)])
        
        merged, L_mid = _merge_intervals(Ls, Rs)
        
        slope = 0.0
        for l, r, li, ri in merged:
            slope += (ri - li) / N
        
        h = (c_end - c_start) / 2.0
        A = slope
        B = L_mid
        integral_segment = (2.0 * A * A * h**3) / 3.0 + 2.0 * B * B * h
        total_integral += integral_segment
        
    return total_integral

def _build_positions(X, N=8):
    """Build the 64x2 positions array from the 1D coordinate array X."""
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = X[i_flat]
    base_y = X[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    return positions

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570
    
    N = 8
    
    inits = [
        np.zeros(N),
        np.arange(N) / N,
        -np.arange(N) / N,
        np.random.default_rng(12345).uniform(-0.5, 0.5, N),
        np.array([0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5]) / N,
        np.array([0, 0.25, 0.5, 0.75, 0, 0.25, 0.5, 0.75]) / N,
        np.array([0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5]) / N,
        np.linspace(0, 0.5, N),
        -np.linspace(0, 0.5, N),
        np.array([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0]) / N,
    ]
    
    best_X = None
    best_cost = float('inf')
    
    # Phase 1: L-BFGS-B from multiple starts with numerical gradients
    for init in inits:
        if time.time() > deadline - 120:
            break
            
        try:
            res = minimize(_compute_1d_cost_squared, init, method='L-BFGS-B', 
                           options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8,
                                    'maxls': 50, 'maxfun': 20000})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass
    
    if best_X is None:
        best_X = np.zeros(N)
    
    # Phase 2: Refine with tighter tolerances
    if time.time() < deadline - 60:
        try:
            res = minimize(_compute_1d_cost_squared, best_X, method='L-BFGS-B', 
                           options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-10,
                                    'maxls': 100, 'maxfun': 50000})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Try Nelder-Mead as a backup polish
    if time.time() < deadline - 30:
        try:
            res = minimize(_compute_1d_cost_squared, best_X, method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final polish
    if time.time() < deadline - 10:
        try:
            res = minimize(_compute_1d_cost_squared, best_X, method='L-BFGS-B', 
                           options={'maxiter': 20000, 'ftol': 1e-15, 'gtol': 1e-12,
                                    'maxls': 100, 'maxfun': 100000})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass
    
    positions = _build_positions(best_X, N)
    
    if not np.all(np.isfinite(positions)):
        positions = _build_positions(np.zeros(N), N)
    
    return positions
