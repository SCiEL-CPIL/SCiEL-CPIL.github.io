# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(x):
    """Compute the union area of 64 triangles given base positions x."""
    n = 64
    M = 20000
    w = 1.0 / n
    # y_m = (m + 0.5) / M
    y = (np.arange(M) + 0.5) / M  # shape (M,)
    a = (np.arange(n) + 1) / n    # shape (n,)
    # L_i = x_i + a_i * y_m, shape (n, M)
    L = x[:, None] + a[:, None] * y[None, :]  # shape (n, M)
    # w_m = (1 - y_m) / n, shape (M,)
    w_m = (1.0 - y) / n
    # Sort L along axis 0
    L_sorted = np.sort(L, axis=0)  # shape (n, M)
    # Union length: w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    # For each column m, gaps = L_sorted[1:, m] - L_sorted[:-1, m]
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]  # shape (n-1, M)
    # union_length_m = w_m + sum_j min(gaps[j, m], w_m)
    # But w_m varies with m, so we need to handle this carefully
    # union_length_m = w_m + sum_j min(gaps[j, m], w_m)
    min_gaps = np.minimum(gaps, w_m[None, :])  # shape (n-1, M)
    union_lengths = w_m + min_gaps.sum(axis=0)  # shape (M,)
    area = union_lengths.mean()
    return area

def _compute_area_grad(x):
    """Compute gradient of area w.r.t. x using finite differences or analytical."""
    n = 64
    M = 20000
    w = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    L = x[:, None] + a[:, None] * y[None, :]
    w_m = (1.0 - y) / n
    L_sorted = np.sort(L, axis=0)
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]
    min_gaps = np.minimum(gaps, w_m[None, :])
    union_lengths = w_m + min_gaps.sum(axis=0)
    area = union_lengths.mean()
    
    # Gradient via finite differences
    eps = 1e-7
    grad = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        area_plus = _compute_area(x_plus)
        area_minus = _compute_area(x_minus)
        grad[i] = (area_plus - area_minus) / (2 * eps)
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
    Uses Nelder-Mead optimization with multiple restarts and coordinate descent polishing.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    n = 64
    w = 1.0 / n
    M = 20000
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    
    # Precompute for faster area evaluation
    def area_fast(x):
        L = x[:, None] + a[:, None] * y[None, :]
        w_m = (1.0 - y) / n
        L_sorted = np.sort(L, axis=0)
        gaps = L_sorted[1:, :] - L_sorted[:-1, :]
        min_gaps = np.minimum(gaps, w_m[None, :])
        union_lengths = w_m + min_gaps.sum(axis=0)
        return union_lengths.mean()
    
    # Start with Keich construction
    best_x = _keich_initial()
    best_area = area_fast(best_x)
    
    # Try multiple restarts with perturbations
    np.random.seed(42)
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            x0 = best_x.copy()
        else:
            # Perturb the best solution
            scale = 0.1 / (restart + 1)
            x0 = best_x + np.random.randn(n) * scale
        
        # Nelder-Mead optimization
        try:
            result = minimize(
                area_fast,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
        
        # Coordinate descent polishing
        if time.time() > deadline - 30:
            break
        
        improved = True
        max_cd_iter = 100
        for cd_iter in range(max_cd_iter):
            if time.time() > deadline - 15:
                break
            if not improved:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline - 15:
                    break
                # Try small moves in coordinate i
                for delta in [1e-4, -1e-4, 1e-5, -1e-5, 1e-6, -1e-6]:
                    x_trial = best_x.copy()
                    x_trial[i] += delta
                    trial_area = area_fast(x_trial)
                    if trial_area < best_area - 1e-12:
                        best_x = x_trial
                        best_area = trial_area
                        improved = True
                        break
    
    # Final validation
    final_area = area_fast(best_x)
    if not np.all(np.isfinite(best_x)) or final_area <= 0:
        return _keich_initial()
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
