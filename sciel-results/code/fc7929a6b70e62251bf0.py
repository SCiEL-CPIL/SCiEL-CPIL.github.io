"""Signed autocorrelation maximum absolute-value variant: minimize max|autocorr| / (integral)^2."""
import numpy as np
import time

N = 400
SEED = 42

def _score(x):
    """Compute max|autocorr| / (sum(x))^2. Lower is better."""
    s = x.sum()
    if s == 0:
        return float('inf')
    ac = np.correlate(x, x, mode='full')
    return np.max(np.abs(ac)) / (s * s)

def _flip_gain(x, i):
    """Compute change in score if we flip x[i]. Approximate via autocorr update."""
    # Flipping x[i] changes x by 2*x[i] at position i (since x[i] is ±1, new = -x[i])
    # delta = -2*x[i] at position i
    # This is expensive to compute exactly; use a heuristic instead.
    # For now, return 0 and use random flips.
    return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    rng = np.random.default_rng(SEED)
    
    # Initialize with random ±1 pattern
    x = rng.choice([-1.0, 1.0], size=N)
    best_x = x.copy()
    best_score = _score(x)
    
    # Local search: random restarts + hill climbing
    max_iter = 20000
    iter_count = 0
    
    while iter_count < max_iter:
        # Check time
        if time.time() > deadline - 5.0:
            break
        
        # Random restarts every 5000 iterations
        if iter_count % 5000 == 0 and iter_count > 0:
            x = rng.choice([-1.0, 1.0], size=N)
            current_score = _score(x)
            if current_score < best_score:
                best_x = x.copy()
                best_score = current_score
            iter_count += 1
            continue
        
        # Try random flips
        i = rng.integers(0, N)
        x[i] = -x[i]
        new_score = _score(x)
        
        if new_score < best_score:
            best_x = x.copy()
            best_score = new_score
        else:
            # Revert with some probability (simulated annealing style)
            if rng.random() > 0.3:
                x[i] = -x[i]
        
        iter_count += 1
    
    return best_x
