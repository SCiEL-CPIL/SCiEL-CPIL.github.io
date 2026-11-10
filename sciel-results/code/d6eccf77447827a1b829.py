# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _calc_area_fast(x, n, y_vals, w_vals, a_vals):
    """
    Calculate area using vectorized operations.
    x: shape (n,)
    """
    L = x[:, None] + a_vals[:, None] * y_vals[None, :]
    sorted_L = np.sort(L, axis=0)
    d = np.diff(sorted_L, axis=0)
    w_b = w_vals[None, :]
    union_lengths = w_vals + np.sum(np.minimum(d, w_b), axis=0)
    area = np.mean(union_lengths)
    return area

def _calc_area_and_grad(x, n, y_vals, w_vals, a_vals):
    """
    Calculate the area and its gradient with respect to x.
    """
    M = len(y_vals)
    L = x[:, None] + a_vals[:, None] * y_vals[None, :]
    
    sorted_L = np.sort(L, axis=0)
    idx = np.argsort(L, axis=0)
    
    d = np.diff(sorted_L, axis=0)
    w_b = w_vals[None, :]
    active = d < w_b
    
    min_terms = np.where(active, d, w_b)
    union_lengths = w_vals + np.sum(min_terms, axis=0)
    area = np.mean(union_lengths)
    
    grad_L = np.zeros_like(sorted_L)
    grad_L[1:, :] += active
    grad_L[:-1, :] -= active
    
    grad_x = np.zeros_like(L)
    np.add.at(grad_x, (idx, np.arange(M)[None, :]), grad_L)
    grad_x_mean = np.mean(grad_x, axis=1)
    
    return area, grad_x_mean

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    # Keich construction
    j = np.arange(1, k + 1)
    positions_keich = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions_keich.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions_keich, dtype=np.float64)
    
    # Full quadrature for final evaluation
    M_full = 20000
    y_full = (np.arange(M_full) + 0.5) / M_full
    w_full = (1 - y_full) / n
    a_vals = (np.arange(n) + 1) / n
    
    # Coarser quadrature for optimization (1000 points)
    M_coarse = 1000
    y_coarse = (np.arange(M_coarse) + 0.5) / M_coarse
    w_coarse = (1 - y_coarse) / n
    
    start_time = time.time()
    deadline = start_time + 580.0
    
    def objective_coarse(x):
        return _calc_area_fast(x, n, y_coarse, w_coarse, a_vals)
    
    def objective_and_grad_coarse(x):
        return _calc_area_and_grad(x, n, y_coarse, w_coarse, a_vals)
    
    best_x = x0.copy()
    best_area_full = _calc_area_fast(x0, n, y_full, w_full, a_vals)
    
    # Phase 1: Coarse optimization with multiple random starts
    np.random.seed(42)
    best_coarse_x = x0.copy()
    best_coarse_area = objective_coarse(x0)
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + np.random.normal(0, 0.05, n)
        
        try:
            res = minimize(
                objective_and_grad_coarse,
                x_start,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
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
    
    # Phase 2: Fine-tune on full grid
    if time.time() < deadline:
        try:
            res = minimize(
                lambda x: _calc_area_and_grad(x, n, y_full, w_full, a_vals),
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_area_full:
                best_x = res.x
                best_area_full = res.fun
        except Exception:
            pass
    
    # Phase 3: Local refinement with small perturbations
    if time.time() < deadline:
        current_x = best_x.copy()
        current_area = best_area_full
        step_size = 0.005
        
        for _ in range(50):
            if time.time() > deadline:
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
            
            if step_size < 1e-7:
                break
        
        if current_area < best_area_full:
            best_x = current_x
            best_area_full = current_area
    
    return best_x

# EVOLVE-BLOCK-END
