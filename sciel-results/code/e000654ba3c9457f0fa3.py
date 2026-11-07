"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 100

def _objective(x):
    h = (x + 1.0) / 2.0
    # Correlation with 1-h
    corr = np.correlate(h, 1.0 - h, mode="full")
    val = np.max(corr) / N * 2.0
    # Penalty for sum constraint (should be 0)
    sum_penalty = np.abs(np.sum(h) - N / 2.0) * 100.0
    return val + sum_penalty

def construct():
    # Start from a perturbed seed to break symmetry
    x0 = np.zeros(N)
    # Add a small sinusoidal perturbation to break the flat minimum
    x0 = 0.1 * np.sin(np.linspace(0, 2 * np.pi * 5, N))
    
    # Optimization bounds in x-space: h in [0,1] -> x in [-1,1]
    bounds = [(-1.0, 1.0)] * N
    
    # Use BFGS with gradient computed via finite differences (L-BFGS-B is robust)
    res = minimize(
        _objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-9}
    )
    
    h_opt = (res.x + 1.0) / 2.0
    h_opt = np.clip(h_opt, 0.0, 1.0)
    
    # Ensure exact sum constraint by adjusting the largest element slightly
    # (The objective penalty helps, but this guarantees validity)
    current_sum = np.sum(h_opt)
    target_sum = N / 2.0
    diff = current_sum - target_sum
    
    if abs(diff) > 1e-12:
        # Adjust the element with the most "room" (closest to 0.5)
        idx = np.argmin(np.abs(h_opt - 0.5))
        h_opt[idx] -= diff
        # Re-clip if adjustment went out of bounds (shouldn't happen if diff is small)
        h_opt = np.clip(h_opt, 0.0, 1.0)
        
        # If clipping changed the sum again, distribute the remainder evenly
        # This is a rare edge case, but ensure validity
        current_sum = np.sum(h_opt)
        diff = current_sum - target_sum
        if abs(diff) > 1e-12:
            h_opt -= diff / N
            h_opt = np.clip(h_opt, 0.0, 1.0)

    # Final check: if it's worse than the seed, return the seed
    val_opt = np.max(np.correlate(h_opt, 1.0 - h_opt, mode="full")) / N * 2.0
    seed_val = 0.5
    
    if val_opt < seed_val:
        return h_opt
    else:
        return np.full(N, 0.5)
