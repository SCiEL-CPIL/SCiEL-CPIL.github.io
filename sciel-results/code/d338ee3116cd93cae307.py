# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _calc_area_and_grad(x, n, y_vals, w_vals, a_vals):
    """
    Calculate the area and its gradient with respect to x.
    
    x: shape (n,)
    y_vals: shape (M,)
    w_vals: shape (M,)
    a_vals: shape (n,)
    """
    M = len(y_vals)
    # L[i, m] = x_i + a_i * y_m
    L = x[:, None] + a_vals[:, None] * y_vals[None, :]
    
    # Sort L along axis 0 (triangle index)
    sorted_L = np.sort(L, axis=0)
    idx = np.argsort(L, axis=0)
    
    # Calculate gaps d_j = sorted_L[j+1, m] - sorted_L[j, m]
    d = np.diff(sorted_L, axis=0) # shape (n-1, M)
    
    # Mask for active gaps: d < w
    w_b = w_vals[None, :]
    active = d < w_b # shape (n-1, M)
    
    # Union length for each y_m:
    min_terms = np.where(active, d, w_b)
    union_lengths = w_vals + np.sum(min_terms, axis=0) # shape (M,)
    
    area = np.mean(union_lengths)
    
    # Gradient calculation
    # grad_L[j, m] = - 1(d[j-1, m] < w_m) + 1(d[j, m] < w_m)
    grad_L = np.zeros_like(sorted_L)
    
    # Add +1 to rows 1..n-1 where active is true
    grad_L[1:, :] += active
    # Add -1 to rows 0..n-2 where active is true
    grad_L[:-1, :] -= active
    
    # Map back to x
    grad_x = np.zeros_like(L)
    np.add.at(grad_x, (idx, np.arange(M)[None, :]), grad_L)
    
    # Average over y
    grad_x_mean = np.mean(grad_x, axis=1)
    
    return area, grad_x_mean

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    # 1. Initial guess: Keich construction
    j = np.arange(1, k + 1)
    positions_keich = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions_keich.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions_keich, dtype=np.float64)
    
    # 2. Setup quadrature points
    M = 20000
    y_vals = (np.arange(M) + 0.5) / M
    w_vals = (1 - y_vals) / n
    a_vals = (np.arange(n) + 1) / n
    
    # 3. Optimization
    start_time = time.time()
    deadline = start_time + 590.0 # 10s safety margin
    
    # Define objective and gradient for scipy
    def objective_and_grad(x):
        area, grad = _calc_area_and_grad(x, n, y_vals, w_vals, a_vals)
        return area, grad
        
    best_x = x0
    best_area, _ = objective_and_grad(best_x)
    
    # Use L-BFGS-B
    try:
        res = minimize(
            objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.success or res.fun < best_area:
            best_x = res.x
            best_area = res.fun
    except Exception:
        pass
        
    # Check deadline
    if time.time() > deadline:
        return best_x
        
    # 4. Local search / refinement phase
    # If time permits, try to improve further with a local search
    # This can help escape flat regions or find better local minima
    # We use a simple random perturbation approach with acceptance if improved
    
    np.random.seed(42) # For reproducibility
    current_x = best_x.copy()
    current_area = best_area
    step_size = 0.01
    
    # Try a few random restarts or local improvements
    for _ in range(20):
        if time.time() > deadline:
            break
            
        # Generate a perturbation
        # Use a small random perturbation
        perturbation = np.random.normal(0, step_size, n)
        new_x = current_x + perturbation
        
        new_area, _ = objective_and_grad(new_x)
        
        if new_area < current_area:
            current_x = new_x
            current_area = new_area
            # Optionally reduce step size if improving
            step_size *= 0.9
        else:
            # If not improving, try a smaller step or different direction
            step_size *= 0.8
            
        if step_size < 1e-6:
            break
            
    if current_area < best_area:
        best_x = current_x
        best_area = current_area
        
    return best_x

# EVOLVE-BLOCK-END
