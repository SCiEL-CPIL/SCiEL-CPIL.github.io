"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_objective(x):
    """Compute the upper bound on C3 for a given sequence."""
    N = len(x)
    # Compute autocorrelation
    acf = np.correlate(x, x, mode='full')
    # The autocorrelation at lag 0 is at index N-1
    # We want to minimize some function of the autocorrelation
    # Try: sum of squares of non-zero lag autocorrelations
    # Or: maximum absolute value of non-zero lag autocorrelations
    # Let's try the sum of absolute values of non-zero lag autocorrelations
    non_zero_acf = np.concatenate([acf[:N-1], acf[N:]])
    return np.sum(np.abs(non_zero_acf))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with a random sequence
    np.random.seed(42)
    x = np.random.choice([-1, 1], size=N)
    best_x = x.copy()
    best_obj = _compute_objective(best_x)
    
    # Local search: try flipping individual bits
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            x[i] = -x[i]
            obj = _compute_objective(x)
            if obj < best_obj:
                best_obj = obj
                best_x = x.copy()
                improved = True
            else:
                x[i] = -x[i]  # Revert
    
    return best_x
# EVOLVE-BLOCK-END
