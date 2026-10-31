"""Optimized program for signed autocorrelation minimum peak-to-sum-ratio."""
import numpy as np
import time

N = 400

def compute_objective(x):
    """Compute max(|autoconv|) / (sum(x))^2"""
    n_fft = 2 * N
    X = np.fft.rfft(x, n=n_fft)
    autoconv = np.fft.irfft(X * np.conj(X), n=n_fft)[:N]
    peak = np.max(np.abs(autoconv))
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    return peak / (s * s)

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_x = np.ones(N)
    best_val = compute_objective(best_x)
    
    # Simulated Annealing on +/- 1 sequences
    # We try to minimize the peak autocorrelation.
    # If the denominator is (sum x)^2, we also want sum x to be large.
    # We use a combined objective for the search: peak / (sum^2)
    
    rng = np.random.default_rng(42)
    
    # Start with a random sequence biased towards +1
    x = rng.choice([-1.0, 1.0], size=N, p=[0.1, 0.9])
    current_val = compute_objective(x)
    
    T = 1.0
    T_min = 1e-6
    alpha = 0.999
    
    while time.time() < deadline:
        # Pick a random index to flip
        idx = rng.integers(0, N)
        old_val = x[idx]
        x[idx] = -old_val
        
        new_val = compute_objective(x)
        
        # Accept if better, or with probability exp(-delta/T)
        if new_val < current_val:
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_x = x.copy()
        else:
            delta = new_val - current_val
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_val = new_val
                else:
                    x[idx] = old_val
            
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart with a new random sequence to escape local minima
            x = rng.choice([-1.0, 1.0], size=N, p=[0.1, 0.9])
            current_val = compute_objective(x)
            T = 1.0

    return best_x

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
