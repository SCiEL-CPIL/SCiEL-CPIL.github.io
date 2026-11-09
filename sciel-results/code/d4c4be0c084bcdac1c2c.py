# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, basinhopping

def _softplus(x):
    """Numerically stable softplus function."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def _compute_area(x, n=64, n_heights=20000):
    """Compute the exact area of the union of triangles for given base positions x."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
    return np.mean(union_lens)

def _smooth_objective(x, n=64, n_heights=2000, beta=50.0):
    """
    Smooth surrogate objective function using soft-min.
    Area is proportional to mean union length.
    Union length at height y is w + sum(min(gap_j, w)).
    min(gap, w) = gap - max(0, gap - w) = gap - softplus(gap - w).
    So union_len = w + sum(gaps) - sum(softplus(gaps - w)).
    Note: sum(gaps) is not constant because sorting changes, but for a fixed 
    permutation of L values, sum(gaps) is constant (L_max - L_min). 
    However, the permutation changes as x changes. 
    Actually, sum(gaps) = L_sorted[-1] - L_sorted[0].
    So union_len = w + (L_max - L_min) - sum(softplus(L_sorted[j+1] - L_sorted[j] - w)).
    """
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n
    
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    
    # L_max - L_min
    span = L_sorted[:, -1] - L_sorted[:, 0]
    
    gaps = np.diff(L_sorted, axis=1)
    
    # softplus(gaps - w)
    # w is shape (n_heights,), gaps is (n_heights, n-1)
    # We need to broadcast w
    sp = _softplus(gaps - w_vals[:, None])
    
    union_lens = w_vals + span - np.sum(sp, axis=1)
    
    return np.mean(union_lens)

def _smooth_objective_grad(x, n=64, n_heights=2000, beta=50.0):
    """
    Compute the gradient of the smooth objective.
    This is complex because of the sorting. We use numerical gradient for simplicity 
    or rely on L-BFGS-B's internal numerical gradient which is robust.
    Given the time budget, we will let L-BFGS-B compute the numerical gradient 
    of the smooth objective, which is much faster to evaluate than the non-smooth one 
    if we use fewer heights for the optimization phase.
    """
    pass

def _keich_initial(n=64):
    """Generate the Keich construction starting point."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return x0

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    deadline = time.time() + 610  # Safety margin before 630s kill
    
    # Start from the Keich construction
    x0 = _keich_initial(n)
    best_x = x0.copy()
    best_area = _compute_area(x0)
    
    np.random.seed(42)
    
    # Parameters for smooth optimization
    n_heights_opt = 1000  # Fewer heights for speed during optimization
    beta = 100.0  # Sharpness of soft-min
    
    def obj_smooth(x):
        return _smooth_objective(x, n=n, n_heights=n_heights_opt, beta=beta)
    
    # Strategy 1: L-BFGS-B from Keich point
    if time.time() < deadline - 60:
        try:
            res = minimize(
                obj_smooth, best_x, method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_area: # Note: res.fun is smooth approx, best_area is exact. 
                                    # We should compare exact areas.
                area_res = _compute_area(res.x)
                if area_res < best_area:
                    best_area = area_res
                    best_x = res.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Multiple restarts with L-BFGS-B
    # The landscape has many local minima. Random restarts help.
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Perturb the best solution found so far
        scale = 0.05 * (1.0 + i * 0.1)
        x_start = best_x + np.random.normal(0, scale, size=n)
        
        try:
            res = minimize(
                obj_smooth, x_start, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            area_res = _compute_area(res.x)
            if area_res < best_area:
                best_area = area_res
                best_x = res.x.copy()
        except Exception:
            pass

    # Strategy 3: Nelder-Mead polish on the exact non-smooth objective
    # This helps refine the solution in the neighborhood of the local minimum.
    if time.time() < deadline - 10:
        def obj_exact(x):
            return _compute_area(x, n=n, n_heights=2000) # Use fewer heights for speed in NM
        
        try:
            res = minimize(
                obj_exact, best_x, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            area_res = _compute_area(res.x)
            if area_res < best_area:
                best_area = area_res
                best_x = res.x.copy()
        except Exception:
            pass

    # Strategy 4: Another round of L-BFGS-B if time permits, starting from NM result
    if time.time() < deadline - 5:
        try:
            res = minimize(
                obj_smooth, best_x, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            area_res = _compute_area(res.x)
            if area_res < best_area:
                best_area = area_res
                best_x = res.x.copy()
        except Exception:
            pass

    # Final verification
    final_area = _compute_area(best_x)
    if final_area <= 0 or not np.all(np.isfinite(best_x)):
        # Fallback to Keich
        best_x = x0

    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
