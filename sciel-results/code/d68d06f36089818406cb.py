# EVOLVE-BLOCK-START
import numpy as np
import time

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

def _area_coarse(x, n=64, M=500):
    """Compute union area via quadrature at M heights (fast, for gradient descent)."""
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

def _gradient_descent_smooth(x0, n=64, M_coarse=500, time_budget=400.0, lr_init=0.01, lr_decay=0.999):
    """
    Gradient descent using finite differences on a coarse area metric.
    This allows many more steps than Brent line search on the fine metric.
    """
    x = x0.astype(np.float64).copy()
    start_time = time.time()
    current_area = _area_coarse(x, n, M_coarse)
    
    # Precompute constants
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    y = (np.arange(M_coarse, dtype=np.float64) + 0.5) / M_coarse
    w = (1.0 - y) / n
    
    # Helper to compute area vectorized for gradient calculation
    # We can't easily vectorize the sort for gradients without significant memory overhead,
    # so we use standard finite differences but with the fast coarse metric.
    
    lr = lr_init
    epsilon = 1e-5
    
    while time.time() - start_time < time_budget:
        if lr < 1e-8:
            break
            
        grad = np.zeros(n, dtype=np.float64)
        
        # Calculate gradient
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += epsilon
            area_plus = _area_coarse(x_plus, n, M_coarse)
            
            x_minus = x.copy()
            x_minus[i] -= epsilon
            area_minus = _area_coarse(x_minus, n, M_coarse)
            
            grad[i] = (area_plus - area_minus) / (2 * epsilon)
            
        # Update
        x -= lr * grad
        new_area = _area_coarse(x, n, M_coarse)
        
        # Simple adaptive learning rate: if area increased, reduce lr and revert
        if new_area > current_area:
            lr *= 0.5
            x = x + lr * grad # revert
            # If lr gets too small, stop
            if lr < 1e-8:
                break
        else:
            current_area = new_area
            # If area decreased, we can keep or slightly increase lr, but be conservative
            # lr = lr * 1.05 
            lr *= lr_decay # Decay to ensure convergence
            
    return x, current_area

def _refine_with_brent(x0, n=64, M_fine=20000, time_budget=100.0, rng=None):
    """
    Refine the solution using Brent's method on the high-precision metric.
    """
    if rng is None:
        rng = np.random.default_rng(0)
        
    from scipy.optimize import minimize_scalar
    
    x = x0.astype(np.float64).copy()
    current_area = _area_fast(x, n, M_fine)
    start_time = time.time()
    
    max_rounds = 20 # Fewer rounds because we start from a good GD solution
    
    for round_idx in range(max_rounds):
        if time.time() - start_time > time_budget:
            break
        
        order = rng.permutation(n)
        
        for idx in order:
            if time.time() - start_time > time_budget:
                break
            
            a_i = (idx + 1.0) / n
            center = 0.5 - a_i
            lower = center - 0.5
            upper = center + 0.5
            
            def objective(val):
                x_temp = x.copy()
                x_temp[idx] = val
                return _area_fast(x_temp, n, M_fine)
            
            try:
                res = minimize_scalar(objective, bounds=(lower, upper), method='bounded', options={'xatol': 1e-7})
                if res.success and res.fun < current_area:
                    x[idx] = res.x
                    current_area = res.fun
            except Exception:
                pass
                
    return x, current_area

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    deadline = time.time() + 630.0
    safety = 30.0
    
    # 1. Start with Aligned Positions
    x_aligned = _aligned_positions(n)
    
    # 2. Gradient Descent on Coarse Metric
    # Budget ~400s for GD. This should converge to a good local min quickly.
    time_budget_gd = 400.0
    x_gd, area_gd = _gradient_descent_smooth(x_aligned, n, 500, time_budget_gd)
    
    best_x = x_gd
    best_area = area_gd # Note: this is coarse area, need to check fine area
    
    # Evaluate fine area for the GD result
    fine_area_gd = _area_fast(best_x, n, 20000)
    
    # 3. Refine with Brent on Fine Metric
    # Use remaining time for refinement
    remaining_time = deadline - safety - time.time()
    if remaining_time > 20.0:
        time_budget_refine = min(remaining_time - 10.0, 150.0) # Cap refinement time
        x_ref, area_ref = _refine_with_brent(best_x, n, 20000, time_budget_refine)
        
        if area_ref < fine_area_gd:
            best_x = x_ref
            best_area = area_ref # This is fine area now
        
    # 4. Final check
    if best_x is None:
        best_x = x_aligned
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
