"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(arr):
    """
    Compute the C3' metric: max|autoconv| / (sum|arr|)^2.
    Lower is better.
    """
    c = np.convolve(arr, arr, mode='full')
    max_abs = np.max(np.abs(c))
    integral = np.sum(np.abs(arr))
    if integral == 0:
        return np.inf
    return max_abs / (integral * integral)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin from 150s limit
    
    # Initial best: constant sequence
    best_seq = np.ones(N)
    best_score = _score(best_seq)
    
    # Search for better sequences using random signs
    # For np.convolve, random signs can yield lower max absolute value
    # compared to the constant sequence which has a large peak.
    
    rng = np.random.default_rng(42)
    num_trials = 1000
    
    for i in range(num_trials):
        if time.time() > deadline:
            break
            
        # Generate random signs
        signs = rng.choice([-1, 1], size=N)
        
        # Evaluate
        score = _score(signs)
        
        if score < best_score:
            best_score = score
            best_seq = signs
            
            # If we get very good, we might stop early or continue
            if best_score < 0.001: # Heuristic threshold
                 pass # Continue to see if we can do better
    
    # Try a few specific patterns if time permits
    # e.g. Barker codes or similar, but random is usually competitive for large N
    
    return best_seq

# EVOLVE-BLOCK-END
