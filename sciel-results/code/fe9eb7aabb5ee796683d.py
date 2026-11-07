"""Erdos minimum overlap: minimize max(correlate(h, 1-h, full))/N*2."""
import numpy as np
import time
from scipy.optimize import minimize, basinhopping

N = 100
DEADLINE = 145.0  # Safety margin below 150s kill
_START = time.time()

def remaining_time():
    return DEADLINE - (time.time() - _START)

def objective_with_penalty(h, penalty_weight=1e5):
    """
    Compute the raw value plus a penalty for violating the sum constraint.
    The raw value is max(correlate(h, 1-h, full)) / N * 2.
    """
    # Ensure h is finite
    if not np.all(np.isfinite(h)):
        return 1e10
    
    # Calculate raw value
    c = np.correlate(h, 1.0 - h, mode="full")
    raw_val = float(np.max(c)) / N * 2.0
    
    # Penalty for sum constraint
    # Target sum is N / 2.0
    sum_h = np.sum(h)
    target_sum = N / 2.0
    sum_penalty = penalty_weight * (sum_h - target_sum) ** 2
    
    # Penalty for bounds (though basinhopping/minimize should respect bounds, 
    # Powell might drift slightly if not careful, but we use bounds in minimize)
    
    return raw_val + sum_penalty

def project(h):
    """
    Project h to satisfy: h in [0,1]^N, sum(h) ~ N/2.
    This is used for the final result and potentially inside local search if needed.
    """
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(100):
        s = h.sum()
        if abs(s - target) < 1e-12:
            break
        # Adjust by uniform shift, then re-clip
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    best_h = np.full(N, 0.5)
    best_val = objective_with_penalty(best_h)
    
    # Initial candidates
    candidates = [
        np.full(N, 0.5),
        np.zeros(N), # Will be penalized, but good start for search? No, sum is 0.
    ]
    
    # Generate a few random starts to help basinhopping
    rng = np.random.default_rng(100)
    for _ in range(3):
        h_rand = rng.uniform(0, 1, N)
        # Project to satisfy sum constraint approximately for a better start
        h_proj = project(h_rand)
        candidates.append(h_proj)
    
    # Find the best initial candidate
    for h_cand in candidates:
        val = objective_with_penalty(h_cand)
        if val < best_val:
            best_val = val
            best_h = h_cand.copy()
    
    # Define the local optimizer
    # Powell is derivative-free and handles non-smooth functions (like max) well.
    def local_minimize(x):
        res = minimize(
            objective_with_penalty,
            x,
            method='Powell',
            bounds=[(0.0, 1.0)] * N,
            options={'maxiter': 100, 'ftol': 1e-12, 'xtol': 1e-12}
        )
        return res

    # Run Basinhopping
    # We need to be careful with the time budget.
    # Basinhopping performs many local minimizations.
    
    if remaining_time() > 20:
        try:
            res_basin = basinhopping(
                objective_with_penalty,
                best_h,
                minimizer=local_minimize,
                niter=50, # Adjust based on time
                stepsize=0.5,
                minimizer_kwargs={'method': 'Powell', 'bounds': [(0.0, 1.0)] * N},
                disp=False
            )
            
            if remaining_time() > 10:
                h_res = project(res_basin.x)
                # Evaluate the true objective (without penalty) for the projected result
                val_res = objective_with_penalty(h_res)
                if val_res < best_val:
                    best_val = val_res
                    best_h = h_res.copy()
            
        except Exception as e:
            pass # If basinhopping fails, we still have best_h

    # If time remains, try a few more random restarts with Powell
    while remaining_time() > 5:
        h_rand = rng.uniform(0, 1, N)
        h_rand = project(h_rand)
        
        res = minimize(
            objective_with_penalty,
            h_rand,
            method='Powell',
            bounds=[(0.0, 1.0)] * N,
            options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12}
        )
        
        if remaining_time() < 5:
            break
            
        h_res = project(res.x)
        val_res = objective_with_penalty(h_res)
        if val_res < best_val:
            best_val = val_res
            best_h = h_res.copy()

    # Final sanity check and projection
    best_h = project(best_h)
    assert np.all(np.isfinite(best_h)), "Non-finite values in result"
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1.0 + 1e-10), "Out of bounds"
    assert abs(best_h.sum() - N / 2.0) < 1e-9 * N, "Sum constraint violated"
    
    return best_h

# EVOLVE-BLOCK-START
# (construct is already defined above)
# EVOLVE-BLOCK-END
