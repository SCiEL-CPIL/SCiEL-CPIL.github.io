# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(x, a, y, w, L_base):
    """
    Compute the area of the union of triangles.
    x: (64,) array of base positions
    a: (64,) array of slopes (a_i)
    y: (20000,) array of heights
    w: (20000,) array of widths at height y
    L_base: (64, 20000) array where L_base[i, m] = a[i] * y[m]
    """
    # L[i, m] = x[i] + a[i] * y[m]
    L = L_base + x[:, np.newaxis]
    
    # Sort L along axis 0 (for each height m, sort the 64 left edges)
    L_sorted = np.sort(L, axis=0)
    
    # Differences between consecutive sorted edges
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # Union length at each height:
    # union_len[m] = w[m] + sum_i min(d[i, m], w[m])
    mins = np.minimum(d, w[np.newaxis, :])
    sum_mins = np.sum(mins, axis=0)
    union_lens = w + sum_mins
    
    # Area is mean of union lengths
    return np.mean(union_lens)

def _compute_area_smooth(x, a, y, w, L_base, alpha=20.0):
    """
    Smooth surrogate for the area computation using soft-min.
    This provides a differentiable objective for L-BFGS-B.
    """
    L = L_base + x[:, np.newaxis]
    
    # Sort to get consecutive differences
    L_sorted = np.sort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # min(d, w) = d - max(d-w, 0) ≈ d - alpha*log(1 + exp((d-w)/alpha))
    z = (d - w[np.newaxis, :]) / alpha
    z = np.clip(z, -50, 50)
    soft_max = alpha * np.log1p(np.exp(z))
    soft_min = d - soft_max
    
    sum_mins = np.sum(soft_min, axis=0)
    union_lens = w + sum_mins
    
    return np.mean(union_lens)

def _keich_construction(n=64):
    """Classical Keich construction."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        bin_str = bin(i)[2:]
        eps = np.zeros(k, dtype=int)
        for pos, bit in enumerate(bin_str):
            eps[k - 1 - pos] = int(bit)
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem, n = 64.
    Returns base positions x = (x_0, ..., x_63) minimizing the union area.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 630s limit
    
    n = 64
    num_heights = 20000
    
    # Precompute static parts
    a = np.arange(1, n + 1, dtype=np.float64) / n
    y = (np.arange(num_heights, dtype=np.float64) + 0.5) / num_heights
    w = (1.0 - y) / n
    
    # L_base[i, m] = a[i] * y[m]
    L_base = np.outer(a, y)
    
    # Initial guess: Keich construction
    x_keich = _keich_construction(n)
    
    best_x = x_keich.copy()
    best_area = _compute_area(x_keich, a, y, w, L_base)
    
    # Phase 1: L-BFGS-B with smooth surrogate
    def objective_smooth(x):
        return _compute_area_smooth(x, a, y, w, L_base, alpha=20.0)
    
    def objective_exact(x):
        return _compute_area(x, a, y, w, L_base)
    
    bounds = [(-2.0, 3.0)] * n
    
    try:
        # Phase 1: Optimize with smooth objective
        res1 = minimize(
            objective_smooth,
            x_keich,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 300,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxls': 50,
                'maxcor': 20
            }
        )
        
        if time.time() < deadline:
            x1 = res1.x
            area1 = objective_exact(x1)
            if np.all(np.isfinite(x1)) and area1 < best_area:
                best_x = x1.copy()
                best_area = area1
        
        # Phase 2: Polish with exact objective using L-BFGS-B
        if time.time() < deadline:
            res2 = minimize(
                objective_exact,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 500,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                    'maxls': 50,
                    'maxcor': 30
                }
            )
            
            x2 = res2.x
            area2 = objective_exact(x2)
            if np.all(np.isfinite(x2)) and area2 < best_area:
                best_x = x2.copy()
                best_area = area2
        
        # Phase 3: Coordinate descent polish
        if time.time() < deadline - 30:
            x_cd = best_x.copy()
            improved = True
            max_cd_iters = 50
            for cd_iter in range(max_cd_iters):
                if time.time() > deadline - 30:
                    break
                any_improve = False
                for i in range(n):
                    if time.time() > deadline - 30:
                        break
                    # Try small perturbations
                    for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001,
                                  0.02, -0.02, 0.0005, -0.0005]:
                        x_try = x_cd.copy()
                        x_try[i] += delta
                        area_try = objective_exact(x_try)
                        if area_try < best_area - 1e-15:
                            x_cd[i] = x_try[i]
                            best_area = area_try
                            any_improve = True
                            break
                if not any_improve:
                    break
            
            if np.all(np.isfinite(x_cd)) and best_area < _compute_area(best_x, a, y, w, L_base):
                best_x = x_cd.copy()
                
    except Exception as e:
        pass
    
    # Final verification
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
