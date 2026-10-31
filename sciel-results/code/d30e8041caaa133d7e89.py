"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import differential_evolution

N = 100

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h over all shifts."""
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for s in range(N):
        shifted = np.roll(one_minus_h, s)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _construct_h(w, c):
    """
    Construct h given width w and center c.
    h[i] = 0.5 + 0.5 * tanh((i - c) / w)
    """
    i = np.arange(N)
    # Avoid division by zero and ensure stability
    if w < 1e-6:
        w = 1e-6
    h = 0.5 + 0.5 * np.tanh((i - c) / w)
    return h

def _objective(params):
    """
    Objective function for joint optimization of w and c.
    Includes a penalty for violating the sum constraint.
    """
    w, c = params
    if w < 1e-6:
        w = 1e-6
    
    h = _construct_h(w, c)
    
    # Penalty for sum constraint
    current_sum = np.sum(h)
    sum_penalty = 1000.0 * (current_sum - N / 2.0) ** 2
    
    # Calculate overlap
    ov = _max_overlap(h)
    
    return ov + sum_penalty

def construct():
    """
    Construct the sequence of step heights.
    """
    # Define bounds for w and c
    # w: transition width. Can be very small (sharp step) to large (smooth).
    # log scale is better for w, but differential_evolution works in linear space.
    # We can optimize log_w and c, or just w and c.
    # Let's optimize w in [0.1, 20] and c in [-10, N+10].
    
    bounds = [(0.1, 20.0), (-10.0, N + 10.0)]
    
    # Use differential_evolution for global search
    # maxiter and popsize control the thoroughness
    # tol controls the convergence
    res = differential_evolution(
        _objective, 
        bounds, 
        seed=42, 
        maxiter=1000, 
        popsize=20, 
        tol=1e-8,
        polish=True # Use L-BFGS-B for final polish
    )
    
    best_w, best_c = res.x
    
    # Construct the final h
    h = _construct_h(best_w, best_c)
    
    # Final correction for exact sum
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    
    # The penalty method should have driven the sum error very close to 0.
    # We do a final precise adjustment.
    if abs(diff) > 1e-10:
        # Distribute difference evenly, then clip and re-adjust
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        
        # Iterative fix for exact sum
        for _ in range(50):
            current_sum = np.sum(h)
            diff = N / 2.0 - current_sum
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                mask = h < 1.0
                if not np.any(mask): break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                for idx in indices:
                    h[idx] = min(1.0, h[idx] + step)
            else:
                mask = h > 0.0
                if not np.any(mask): break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                for idx in indices:
                    h[idx] = max(0.0, h[idx] + step)

    return h

# EVOLVE-BLOCK-START
# (The construct function is defined above, but we need to keep the interface)
# The task says "Keep the function names and signatures of the seed program."
# The seed program has construct() -> sequence of step heights.
# So we just need to make sure construct() is defined and returns the right thing.
# EVOLVE-BLOCK-END
