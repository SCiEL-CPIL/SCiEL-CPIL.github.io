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
    # x shape (n,), a shape (n,), y shape (M,)
    # L = x[:, None] + a[:, None] * y[None, :]
    L = x[:, None] + a_vals[:, None] * y_vals[None, :]
    
    # Sort L along axis 0 (triangle index)
    # We need the sorted L and the indices to map gradients back
    sorted_L = np.sort(L, axis=0)
    # Indices of the sort: idx[m, j] is the original index i such that L[i, m] = sorted_L[j, m]
    idx = np.argsort(L, axis=0)
    
    # Calculate gaps d_j = sorted_L[j+1, m] - sorted_L[j, m]
    d = np.diff(sorted_L, axis=0) # shape (n-1, M)
    
    # Mask for active gaps: d < w
    # w_vals shape (M,), need to broadcast to (n-1, M)
    w_b = w_vals[None, :]
    active = d < w_b # shape (n-1, M)
    
    # Union length for each y_m:
    # U_m = w_m + sum_{j=0}^{n-2} min(d[j, m], w_m)
    # min(d, w) = d * active + w * (1 - active)
    min_terms = np.where(active, d, w_b)
    union_lengths = w_vals + np.sum(min_terms, axis=0) # shape (M,)
    
    area = np.mean(union_lengths)
    
    # Gradient calculation
    # grad_L[j, m] = - 1(d[j-1, m] < w_m) + 1(d[j, m] < w_m)
    # Let's build grad_L of shape (n, M)
    grad_L = np.zeros_like(sorted_L)
    
    # Contribution from d[j, m] to grad_L[j+1, m] is +1 if active
    # Contribution from d[j, m] to grad_L[j, m] is -1 if active
    
    # active shape (n-1, M)
    # Add +1 to rows 1..n-1 where active is true
    # grad_L[1:, :] += active.T ? No, active is (n-1, M). 
    # We want to add to grad_L[j+1, m] for each j, m.
    # So grad_L[1:, :] += active (broadcasting along axis 0? No, active is (n-1, M), grad_L[1:] is (n-1, M)).
    grad_L[1:, :] += active
    
    # Add -1 to rows 0..n-2 where active is true
    # grad_L[:-1, :] -= active
    grad_L[:-1, :] -= active
    
    # Now grad_L is gradient w.r.t sorted_L.
    # We need to map back to x.
    # grad_x[i, m] = sum_{j: idx[m, j] == i} grad_L[j, m]
    # Since idx is a permutation, for each m, grad_x[idx[m, :], m] = grad_L[:, m]
    
    # Efficient scatter:
    # We can use np.add.at
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
    # maxiter should be sufficient for 64 vars
    # ftol is relative tolerance
    try:
        res = minimize(
            objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True, # We return (f, g)
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.success or res.fun < best_area:
            best_x = res.x
            best_area = res.fun
    except Exception:
        pass
        
    # If time allows, we could try a few more restarts or a different method, 
    # but L-BFGS-B on a convex piecewise linear function from a good start 
    # should be very fast.
    
    # Check deadline
    if time.time() > deadline:
        return best_x
        
    return best_x

# EVOLVE-BLOCK-END
