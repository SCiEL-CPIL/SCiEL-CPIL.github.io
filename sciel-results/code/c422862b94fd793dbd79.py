# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _union_area(x):
    """Compute the union area of 64 triangles given base positions x."""
    n = 64
    M = 20000
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    L = x[:, None] + a[:, None] * y[None, :]  # shape (n, M)
    w_m = (1.0 - y) / n
    L_sorted = np.sort(L, axis=0)
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]
    min_gaps = np.minimum(gaps, w_m[None, :])
    union_lengths = w_m + min_gaps.sum(axis=0)
    return union_lengths.mean()

def _union_area_grad(x):
    """Compute area and analytical gradient of union area w.r.t. x."""
    n = 64
    M = 20000
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    L = x[:, None] + a[:, None] * y[None, :]  # shape (n, M)
    w_m = (1.0 - y) / n  # shape (M,)
    
    # Sort L and track which original index is at each sorted position
    L_sorted, sort_idx = np.sort(L, axis=0)  # both shape (n, M)
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]  # shape (n-1, M)
    
    # Active pairs: gap < w_m (the min is binding)
    active = (gaps < w_m[None, :])  # shape (n-1, M)
    
    # Union length
    min_gaps = np.where(active, gaps, w_m[None, :])
    union_lengths = w_m + min_gaps.sum(axis=0)
    area = union_lengths.mean()
    
    # Gradient computation:
    # d(area)/d(x_i) = (1/M) * sum_m [d(union_len_m)/d(x_i)]
    # union_len_m = w_m + sum_j min(gap_j, w_m)
    # For active pair j: min(gap_j, w_m) = gap_j = L_sorted[j+1] - L_sorted[j]
    #   d/d(x_i) = +1 if i is at sorted position j+1, -1 if at position j
    # For inactive pair j: min = w_m, derivative = 0
    #
    # So grad[i] = (1/M) * sum_m [sum over active j: (1 if sort_idx[j+1,m]==i) - (1 if sort_idx[j,m]==i)]
    #
    # Equivalently: for each m, the gradient contribution is:
    #   +1 for the triangle at sorted position (j+1) for each active j
    #   -1 for the triangle at sorted position j for each active j
    #
    # This means: for each m, count how many active pairs have this triangle as the upper (j+1) 
    # minus how many have it as the lower (j).
    #
    # Let's compute: for each m, the net gradient contribution for each sorted position k:
    #   +1 for each active j where j+1 == k (i.e., j = k-1, active)
    #   -1 for each active j where j == k (active)
    # So net for sorted position k: active[k-1] - active[k] (with active[-1]=0, active[n-1]=0)
    
    # Build gradient using bincount approach
    # For each m, the gradient at sorted position k is: (active[k-1,m] if k>0 else 0) - (active[k,m] if k<n-1 else 0)
    # Then map back to original indices via sort_idx
    
    # active shape: (n-1, M)
    # For sorted position 0: contribution = -active[0, m]
    # For sorted position k (1..n-2): contribution = active[k-1, m] - active[k, m]
    # For sorted position n-1: contribution = active[n-2, m]
    
    grad_sorted = np.zeros((n, M), dtype=np.float64)
    if n > 1:
        grad_sorted[0, :] = -active[0, :]
        if n > 2:
            grad_sorted[1:n-1, :] = active[0:n-2, :] - active[1:n-1, :]
        grad_sorted[n-1, :] = active[n-2, :]
    
    # Map from sorted positions to original indices
    # grad[i] = (1/M) * sum_m grad_sorted[sort_idx[i,m], m]
    # Use np.add.at for the scatter
    grad = np.zeros(n, dtype=np.float64)
    # sort_idx shape: (n, M), grad_sorted shape: (n, M)
    # For each m, for each sorted position k, add grad_sorted[k,m] to grad[sort_idx[k,m]]
    np.add.at(grad, sort_idx.ravel(), grad_sorted.ravel())
    grad /= M
    
    return area, grad

def _keich_initial():
    """Generate initial positions using Keich construction."""
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses L-BFGS-B with analytical gradient and multiple restarts.
    """
    start_time = time.time()
    deadline = start_time + 570  # 30s safety margin
    
    n = 64
    
    # Start with Keich construction
    best_x = _keich_initial()
    best_area, _ = _union_area_grad(best_x)
    
    np.random.seed(42)
    
    # Multiple restarts with L-BFGS-B
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            x0 = best_x.copy()
        else:
            # Perturb the best solution with decreasing scale
            scale = 0.5 / (restart + 1)
            x0 = best_x + np.random.randn(n) * scale
        
        # L-BFGS-B optimization with analytical gradient
        try:
            result = minimize(
                _union_area,
                x0,
                jac=_union_area_grad,
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 30:
        try:
            result = minimize(
                _union_area,
                best_x,
                jac=_union_area_grad,
                method='L-BFGS-B',
                options={
                    'maxiter': 5000,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 50
                }
            )
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Final validation
    final_area, _ = _union_area_grad(best_x)
    if not np.all(np.isfinite(best_x)) or final_area <= 0:
        return _keich_initial()
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
