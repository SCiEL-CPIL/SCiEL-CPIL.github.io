import numpy as np
import time
from scipy.optimize import minimize

def _calc_area_and_grad(x, n, y_vals, w_vals, a_vals):
    """
    Calculate the area and its gradient (subgradient) with respect to x.
    """
    M = len(y_vals)
    # L[i, m] = x_i + a_i * y_m
    L = x[:, None] + a_vals[:, None] * y_vals[None, :]
    
    # Sort L along axis 0 (triangle index) for each height m
    sorted_L = np.sort(L, axis=0)
    # Indices of the sort: idx[m, j] is the original index i such that L[i, m] = sorted_L[j, m]
    idx = np.argsort(L, axis=0)
    
    # Calculate gaps d_j = sorted_L[j+1, m] - sorted_L[j, m]
    d = np.diff(sorted_L, axis=0) # shape (n-1, M)
    
    # Mask for active gaps: d < w
    w_b = w_vals[None, :]
    active = d < w_b # shape (n-1, M)
    
    # Union length for each y_m:
    # U_m = w_m + sum_{j=0}^{n-2} min(d[j, m], w_m)
    min_terms = np.where(active, d, w_b)
    union_lengths = w_vals + np.sum(min_terms, axis=0) # shape (M,)
    
    area = np.mean(union_lengths)
    
    # Gradient calculation (subgradient)
    # d/d(L_sorted[j]) of min(d_j, w) is:
    # If d_j < w: d(min)/d(L_sorted[j+1]) = 1, d(min)/d(L_sorted[j]) = -1
    # If d_j >= w: 0
    # So grad_L[j+1, m] += active[j, m]
    #    grad_L[j, m] -= active[j, m]
    
    grad_L = np.zeros_like(sorted_L)
    grad_L[1:, :] += active
    grad_L[:-1, :] -= active
    
    # Map gradient back to original x indices
    # grad_x[i, m] = sum_{j: idx[m, j] == i} grad_L[j, m]
    grad_x = np.zeros_like(L)
    np.add.at(grad_x, (idx, np.arange(M)[None, :]), grad_L)
    
    # Average over y
    grad_x_mean = np.mean(grad_x, axis=1)
    
    return area, grad_x_mean

def _calc_area_fast(x, n, y_vals, w_vals, a_vals):
    """
    Calculate area without gradient for speed.
    """
    L = x[:, None] + a_vals[:, None] * y_vals[None, :]
    sorted_L = np.sort(L, axis=0)
    d = np.diff(sorted_L, axis=0)
    w_b = w_vals[None, :]
    union_lengths = w_vals + np.sum(np.minimum(d, w_b), axis=0)
    return np.mean(union_lengths)

def _keich_positions(n=64):
    """Generate Keich construction positions."""
    k = int(np.log2(n))
    positions = np.zeros(n)
    for i in range(n):
        bits = format(i, f'0{k}b')
        eps = np.array([int(b) for b in bits], dtype=float)
        j = np.arange(1, k + 1)
        positions[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return positions

def kakeya_needle_2d_positions64() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin for 600s limit
    
    n = 64
    a_vals = (np.arange(n) + 1) / n
    
    # Full quadrature for final evaluation
    M_full = 20000
    y_full = (np.arange(M_full) + 0.5) / M_full
    w_full = (1 - y_full) / n
    
    # Coarser quadrature for optimization
    M_coarse = 1000
    y_coarse = (np.arange(M_coarse) + 0.5) / M_coarse
    w_coarse = (1 - y_coarse) / n
    
    # Initial guess: Keich construction
    x0 = _keich_positions(n)
    
    best_x = x0.copy()
    best_area_full = _calc_area_fast(x0, n, y_full, w_full, a_vals)
    
    # Phase 1: Coarse optimization using Powell's method (derivative-free, robust to non-smoothness)
    # Powell's method is slower per iteration but often finds better minima in non-smooth landscapes
    # than gradient-based methods when gradients are subgradients.
    
    def objective_coarse(x):
        return _calc_area_fast(x, n, y_coarse, w_coarse, a_vals)
        
    np.random.seed(42)
    
    # Try a few random restarts with Powell
    best_coarse_x = x0.copy()
    best_coarse_area = objective_coarse(x0)
    
    n_restarts = 4
    for restart in range(n_restarts):
        if time.time() > deadline - 200: # Reserve 200s for fine tuning
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + np.random.normal(0, 0.02, n)
        
        try:
            # maxiter is tricky for Powell, it counts iterations of the main loop.
            # We rely on the timeout guard.
            res = minimize(
                objective_coarse,
                x_start,
                method='Powell',
                options={'maxiter': 50, 'xtol': 1e-8, 'ftol': 1e-8, 'disp': False}
            )
            if res.fun < best_coarse_area:
                best_coarse_area = res.fun
                best_coarse_x = res.x.copy()
        except Exception:
            pass
    
    # Evaluate best coarse solution on full grid
    area_full = _calc_area_fast(best_coarse_x, n, y_full, w_full, a_vals)
    if area_full < best_area_full:
        best_x = best_coarse_x
        best_area_full = area_full
    
    # Phase 2: Fine-tune on full grid with L-BFGS-B using subgradients
    # This polishes the solution found by Powell.
    
    def objective_and_grad_full(x):
        return _calc_area_and_grad(x, n, y_full, w_full, a_vals)

    if time.time() < deadline - 50:
        try:
            res = minimize(
                objective_and_grad_full,
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            if res.fun < best_area_full:
                best_x = res.x
                best_area_full = res.fun
        except Exception:
            pass
            
    # Phase 3: Local refinement with small perturbations (Random Walk)
    # This helps escape tiny local minima caused by the discrete quadrature grid.
    if time.time() < deadline - 10:
        current_x = best_x.copy()
        current_area = best_area_full
        step_size = 0.001
        
        # Use a fixed seed for reproducibility of the walk, or continue random state
        # We already set seed 42, so this is deterministic.
        
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            
            perturbation = np.random.normal(0, step_size, n)
            new_x = current_x + perturbation
            new_area = _calc_area_fast(new_x, n, y_full, w_full, a_vals)
            
            if new_area < current_area:
                current_x = new_x
                current_area = new_area
                step_size *= 0.95
            else:
                step_size *= 0.85
            
            if step_size < 1e-8:
                break
        
        if current_area < best_area_full:
            best_x = current_x
            best_area_full = current_area

    return np.array(best_x, dtype=np.float64)
