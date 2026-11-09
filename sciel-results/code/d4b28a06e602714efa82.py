# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys

def _smooth_min(a, b, beta):
    """
    Smooth approximation of min(a, b) using Log-Sum-Exp.
    a, b are arrays.
    """
    # To avoid overflow/underflow, we use the stable logsumexp form.
    # min(a, b) = -1/beta * log(exp(-beta*a) + exp(-beta*b))
    # Let m = max(a, b). 
    # log(exp(-beta*a) + exp(-beta*b)) = -beta*m + log(exp(-beta*(a-m)) + exp(-beta*(b-m)))
    # Since a-m <= 0 and b-m <= 0, the exponents are <= 0, so exp is stable.
    
    m = np.maximum(a, b)
    # If a or b is inf, handle separately? Problem guarantees finite.
    # exp(-beta * (a - m))
    # Note: if a == m, exp(0) = 1. If a < m, exp(negative) < 1.
    
    # Vectorized calculation
    # We need to be careful with broadcasting.
    # a and b have same shape.
    
    term1 = np.exp(-beta * (a - m))
    term2 = np.exp(-beta * (b - m))
    log_sum = np.log(term1 + term2)
    
    return m - log_sum / beta

def _area_smooth(x, y_vals, a, n, beta):
    """
    Compute the smooth approximation of the area of the union of triangles.
    """
    # L_i = x_i + a_i * y
    # Shape: x is (n,), y_vals is (M,), a is (n,)
    # L is (M, n)
    L = x[None, :] + a[None, :] * y_vals[:, None]
    
    # Sort L along axis 1
    L_sorted = np.sort(L, axis=1)
    
    # Gaps between consecutive sorted elements
    # gaps[i, j] = L_sorted[i, j+1] - L_sorted[i, j]
    # Shape: (M, n-1)
    gaps = np.diff(L_sorted, axis=1)
    
    # w is (1-y)/n, shape (M,)
    w_vec = (1.0 - y_vals) / n
    
    # We need sum over j of min(gaps[i, j], w[i])
    # gaps is (M, n-1), w_vec is (M,). 
    # We need to broadcast w_vec to (M, n-1)
    w_broadcast = w_vec[:, None]
    
    # Compute smooth min for each pair
    # smooth_min(gaps, w_broadcast, beta)
    mins = _smooth_min(gaps, w_broadcast, beta)
    
    # Sum over axis 1
    sum_mins = np.sum(mins, axis=1)
    
    # Union length at height i is w[i] + sum_mins[i]
    union_lens = w_vec + sum_mins
    
    # Area is mean of union lengths
    return np.mean(union_lens)

def _area_true(x, y_vals, a, n):
    """
    Compute the exact area of the union of triangles.
    """
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    w_vec = (1.0 - y_vals) / n
    w_broadcast = w_vec[:, None]
    
    # Exact min
    mins = np.minimum(gaps, w_broadcast)
    sum_mins = np.sum(mins, axis=1)
    union_lens = w_vec + sum_mins
    return np.mean(union_lens)

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    # Precompute constants
    a = np.array([(i + 1) / n for i in range(n)], dtype=np.float64)
    y_vals = np.array([(m + 0.5) / 20000 for m in range(20000)], dtype=np.float64)
    
    # Start from the Keich construction
    j = np.arange(1, k + 1)
    keich_positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        keich_positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0_keich = np.array(keich_positions, dtype=np.float64)
    
    # Generate a few random initializations around Keich
    # And some purely random ones
    rng = np.random.default_rng(42)
    initializations = [x0_keich]
    
    # Add perturbations
    for scale in [0.1, 0.01, 0.001]:
        x_pert = x0_keich + rng.normal(0, scale, n)
        initializations.append(x_pert)
        
    # Add random uniform in [0, 1]
    x_rand = rng.uniform(0, 1, n)
    initializations.append(x_rand)
    
    best_x = x0_keich.copy()
    best_area = _area_true(x0_keich, y_vals, a, n)
    
    deadline = time.time() + 580  # Safety margin
    
    # Annealing schedule for beta
    # Start with small beta (smooth), increase to large beta (sharp)
    # We will run optimization for each beta level.
    # To save time, we limit the number of iterations per beta level.
    
    beta_schedule = [10, 20, 50, 100, 200, 500, 1000]
    
    # We will try to optimize from the best current solution at each beta step.
    # This is a sequential annealing approach.
    
    current_x = x0_keich.copy()
    
    for beta in beta_schedule:
        if time.time() > deadline - 60: # Leave time for final polish and other starts
            break
            
        # Define the smooth objective
        def obj(x):
            return _area_smooth(x, y_vals, a, n, beta)
        
        # Use L-BFGS-B for gradient-based optimization
        # We need to provide a gradient? L-BFGS-B approximates it.
        # We can provide analytical gradient if we derive it, but numerical is safer for now.
        # However, for speed, we might want to limit maxiter.
        
        try:
            result = minimize(
                obj, current_x,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            
            if result.fun < obj(current_x):
                current_x = result.x.copy()
                
                # Check true area
                true_area = _area_true(current_x, y_vals, a, n)
                if true_area < best_area:
                    best_area = true_area
                    best_x = current_x.copy()
                    
        except Exception:
            pass
    
    # Try a few random restarts with the full annealing schedule if time permits
    # This helps escape local minima.
    
    time_remaining = deadline - time.time()
    
    # If we have significant time left, do random restarts
    # Each restart: pick a random init, run a shorter annealing schedule
    # We'll use a subset of the beta schedule for speed.
    
    if time_remaining > 120:
        # Use a coarser schedule for restarts
        beta_schedule_restart = [50, 200, 1000]
        
        # Number of restarts depends on time
        # Estimate time per restart: ~10-20s
        max_restarts = int((time_remaining - 60) / 20)
        max_restarts = max(1, min(max_restarts, 10))
        
        for i in range(max_restarts):
            if time.time() > deadline - 60:
                break
                
            # Random init
            x_start = rng.uniform(0, 1, n)
            # Or perturb best_x
            if i % 2 == 0:
                x_start = best_x + rng.normal(0, 0.05, n)
                
            current_x = x_start.copy()
            
            for beta in beta_schedule_restart:
                if time.time() > deadline - 60:
                    break
                    
                def obj(x):
                    return _area_smooth(x, y_vals, a, n, beta)
                
                try:
                    result = minimize(
                        obj, current_x,
                        method='L-BFGS-B',
                        options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5}
                    )
                    
                    if result.fun < obj(current_x):
                        current_x = result.x.copy()
                        
                        true_area = _area_true(current_x, y_vals, a, n)
                        if true_area < best_area:
                            best_area = true_area
                            best_x = current_x.copy()
                            
                except Exception:
                    pass
    
    # Final polish: Use Nelder-Mead on the true (non-smooth) objective
    # This can handle the kinks better than gradient methods in the final stage.
    
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _area_true, best_x, args=(y_vals, a, n),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
        
    # One more polish with tighter tolerance if time permits
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _area_true, best_x, args=(y_vals, a, n),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
        except Exception:
            pass

    # Ensure all values are finite
    if not np.all(np.isfinite(best_x)):
        return x0_keich.copy()
    
    return best_x


# EVOLVE-BLOCK-END
