# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _union_area(x, n=64, M=20000):
    """
    Compute the area of the union of n triangles using the specified quadrature.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    
    # L_i(y) = x_i + a_i * y
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L along axis 0
    L_sorted = np.sort(L, axis=0)
    
    # Consecutive differences
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # Union length = w + sum over j of min(d_j, w)
    min_d_w = np.minimum(d, w[np.newaxis, :])
    
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    
    area = np.mean(union_lengths)
    return area

def _union_area_gradient(x, n=64, M=20000):
    """
    Compute the area and its subgradient with respect to x.
    Optimized for speed using np.bincount.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L and track indices
    sort_idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, sort_idx, axis=0)
    
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    min_d_w = np.minimum(d, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = np.mean(union_lengths)
    
    grad = np.zeros(n, dtype=np.float64)
    
    # mask for where d_j < w (active constraints)
    mask = d < w[np.newaxis, :]
    
    # Vectorized gradient accumulation
    # We need to add 1/M to grad[upper] and -1/M to grad[lower] for all active (j, m)
    # upper indices are sort_idx[j+1, m], lower are sort_idx[j, m]
    
    # Flatten the relevant indices
    # mask is (n-1, M)
    # sort_idx is (n, M)
    
    # Get indices where mask is True
    # We can use np.nonzero
    j_indices, m_indices = np.nonzero(mask)
    
    if len(j_indices) > 0:
        upper_idx = sort_idx[j_indices + 1, m_indices]
        lower_idx = sort_idx[j_indices, m_indices]
        
        # Use bincount for fast accumulation
        weight = 1.0 / M
        grad += np.bincount(upper_idx, weights=np.full_like(upper_idx, weight, dtype=np.float64), minlength=n)
        grad -= np.bincount(lower_idx, weights=np.full_like(lower_idx, weight, dtype=np.float64), minlength=n)
        
    return area, grad

def _keich_init(n):
    """
    Generate initial positions based on the Keich construction.
    """
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x0 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    return x0

def _random_init(n, scale=1.0):
    return np.random.uniform(-scale, scale, n)

def _sorted_uniform_init(n):
    return np.linspace(-1.0, 1.0, n)

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # 1. Generate multiple initial candidates
    candidates = [
        _keich_init(n),
        _random_init(n, scale=0.5),
        _random_init(n, scale=1.0),
        _sorted_uniform_init(n),
        # A specific heuristic: spread out
        np.linspace(-0.5, 0.5, n) + 0.1 * np.random.randn(n),
    ]
    
    best_x = None
    best_area = np.inf
    
    # 2. Optimize each candidate
    for i, x0 in enumerate(candidates):
        if time.time() > deadline - 120:
            break
            
        x0 = x0.copy()
        try:
            # Local Optimization with L-BFGS-B
            res = minimize(
                _union_area_gradient,
                x0,
                args=(n, M),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'gtol': 1e-5}
            )
            area = res.fun
            if area < best_area:
                best_area = area
                best_x = res.x.copy()
        except Exception:
            # Fallback to direct evaluation if optimizer fails
            area = _union_area(x0, n, M)
            if area < best_area:
                best_area = area
                best_x = x0.copy()
                
    # If no valid result found, return Keich
    if best_x is None:
        return _keich_init(n)
        
    # 3. Basin Hopping / Multi-start refinement
    # Use the best found so far as the center
    current_x = best_x.copy()
    current_area = best_area
    step_size = 0.5
    
    # Perform a few local searches around the best point
    for i in range(20):
        if time.time() > deadline - 60:
            break
            
        # Generate a new candidate near the current best
        x_new = current_x + step_size * np.random.randn(n)
        
        try:
            res = minimize(
                _union_area_gradient,
                x_new,
                args=(n, M),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 30, 'gtol': 1e-5}
            )
            area_new = res.fun
            x_new = res.x
            
            if area_new < best_area:
                best_area = area_new
                best_x = x_new.copy()
                current_x = x_new.copy()
                step_size *= 0.9
            else:
                # If we didn't improve, maybe try a slightly larger step or keep
                # Simple strategy: keep step size, or slightly increase if stuck for a while
                pass
        except Exception:
            pass

    # 4. Final Polish with higher precision
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _union_area_gradient,
                best_x,
                args=(n, M),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'gtol': 1e-8}
            )
            if res.fun < best_area:
                best_x = res.x.copy()
                best_area = res.fun
        except Exception:
            pass
            
    # Validation
    if not np.all(np.isfinite(best_x)):
        return _keich_init(n)
    if best_area <= 0:
        return _keich_init(n)
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
