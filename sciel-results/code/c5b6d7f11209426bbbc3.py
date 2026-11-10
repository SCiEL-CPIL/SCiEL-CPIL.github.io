# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

def _area_fast(x, n=64, M=20000):
    """Compute union area via quadrature at M heights."""
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    y = (np.arange(M, dtype=np.float64) + 0.5) / M
    w = (1.0 - y) / n
    L = x[None, :] + a[None, :] * y[:, None]
    Ls = np.sort(L, axis=1)
    gaps = Ls[:, 1:] - Ls[:, :-1]
    union_len = w + np.sum(np.minimum(gaps, w[:, None]), axis=1)
    return float(np.mean(union_len))

def _aligned_positions(n=64):
    """
    Initialize such that all triangle apices converge to x=0.5 at y=1.
    Vertex at y=1 is (x_i + a_i, 1).
    We want x_i + a_i = 0.5 => x_i = 0.5 - a_i.
    """
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    return 0.5 - a

def _cyclic_line_search(x0, n=64, M=20000, max_rounds=50, time_budget=500.0, rng=None):
    """
    Cyclic coordinate descent with exact 1D line search (Brent).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    x = x0.astype(np.float64).copy()
    current_area = _area_fast(x, n, M)
    start_time = time.time()
    
    for round_idx in range(max_rounds):
        if time.time() - start_time > time_budget:
            break
        
        # Randomize coordinate order for this round to avoid cyclic bias
        order = rng.permutation(n)
        
        for idx in order:
            if time.time() - start_time > time_budget:
                break
            
            # Define objective for 1D search: area as function of x[idx]
            # We fix other coordinates and vary x[idx]
            # The range of x[idx] that makes sense is roughly [-1, 2]
            # Let's search in [-0.5, 1.5] to be safe, or tighter based on neighbors
            # Actually, since a_i is in (0,1], and we want to minimize overlap,
            # the optimal x_i is likely near 0.5 - a_i.
            # Let's search in [0.5 - a_i - 0.5, 0.5 - a_i + 0.5] = [0.5 - a_i - 0.5, 0.5 - a_i + 0.5]
            a_i = (idx + 1.0) / n
            center = 0.5 - a_i
            lower = center - 0.5
            upper = center + 0.5
            
            def objective(val):
                x_temp = x.copy()
                x_temp[idx] = val
                return _area_fast(x_temp, n, M)
            
            # Use Brent's method
            try:
                res = minimize_scalar(objective, bounds=(lower, upper), method='bounded', options={'xatol': 1e-6})
                if res.success and res.fun < current_area:
                    x[idx] = res.x
                    current_area = res.fun
            except Exception:
                pass
        
        # Check for convergence
        if current_area < 0.01: # Arbitrary low threshold, area is normalized-ish
             pass # Just log or break if very small
    
    return x, current_area

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    deadline = time.time() + 600.0
    safety = 30.0
    
    # 1. Start with Aligned Positions (proven good init)
    x_aligned = _aligned_positions(n)
    
    # 2. Perform Cyclic Line Search from Aligned
    # Budget most time here
    time_budget_main = 400.0
    x_ls, area_ls = _cyclic_line_search(x_aligned, n, 20000, max_rounds=100, time_budget=time_budget_main)
    
    best_x = x_ls
    best_area = area_ls
    
    # 3. If time remains, try a second start from a slightly perturbed aligned
    # or a different structure
    if time.time() < deadline - safety - 60.0:
        # Perturb aligned
        rng = np.random.default_rng(123)
        x_pert = x_aligned + rng.normal(0, 0.01, size=n)
        
        remaining = min(100.0, deadline - safety - time.time() - 10.0)
        if remaining > 30.0:
             x_ls2, area_ls2 = _cyclic_line_search(x_pert, n, 20000, max_rounds=50, time_budget=remaining)
             if area_ls2 < best_area:
                 best_x = x_ls2
                 best_area = area_ls2
    
    # 4. Final check
    if best_x is None:
        best_x = x_aligned
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
