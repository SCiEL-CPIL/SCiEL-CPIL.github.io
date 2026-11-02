"""
Optimized signed step function for third autocorrelation inequality.

Strategy:
- The score is the upper bound on C3, which is proportional to the maximum
  absolute value of the autocorrelation sequence.
- We use a combination of:
  1. A strong initial guess based on van der Corput / low-discrepancy sequences
     to get a good starting point with spectral flatness.
  2. Local search (greedy coordinate descent) to minimize the max autocorrelation.
  3. Simulated annealing to escape local minima.
- We track elapsed time and return the best result found before the deadline.

The interface: construct() -> sequence of signed step heights (length N=400)
Score is the upper bound on C3 (lower is better).
"""

import numpy as np
import time

N = 400

# Deadline: 150 seconds wall-clock, use 140 seconds for safety
DEADLINE = 140.0


def _autocorr_max_abs(x):
    """
    Compute the maximum absolute value of the autocorrelation of x.
    x is a 1D array of length N with values in {-1, +1}.
    
    Autocorrelation at lag k: sum_{i=0}^{N-1-k} x[i] * x[i+k]
    We compute all lags and return max |autocorr|.
    """
    N = len(x)
    # Use FFT-based autocorrelation for efficiency
    # Zero-pad to next power of 2 or 2*N for linear convolution
    size = 1
    while size < 2 * N:
        size *= 2
    f = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)[:N]
    return np.max(np.abs(ac))


def _vdC_sequence(n, base=2):
    """
    Generate the first n van der Corput numbers in [0, 1).
    These are low-discrepancy sequences.
    """
    seq = np.zeros(n)
    for i in range(n):
        f = 1.0
        num = i + 1
        v = 0.0
        while num > 0:
            num, r = divmod(num, base)
            f /= base
            v += f * r
        seq[i] = v
    return seq


def _initial_guess_vdc():
    """
    Generate an initial signed step function using van der Corput sequence.
    Map each value to -1 or +1 based on whether it's in the lower or upper half.
    """
    vdc = _vdC_sequence(N, base=2)
    # Threshold at 0.5
    x = np.where(vdc < 0.5, -1.0, 1.0)
    return x


def _greedy_polish(x, max_iter=50000):
    """
    Greedy coordinate descent: flip one element at a time if it reduces
    the max autocorrelation.
    """
    best_score = _autocorr_max_abs(x)
    current_score = best_score
    x = x.copy()
    
    for it in range(max_iter):
        improved = False
        # Try flipping each element
        for i in range(N):
            old_val = x[i]
            x[i] = -old_val
            new_score = _autocorr_max_abs(x)
            if new_score < current_score - 1e-12:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                improved = True
                break
            else:
                x[i] = old_val
        if not improved:
            break
    return x, best_score


def _simulated_annealing(x, start_temp=1.0, end_temp=0.01, max_iter=100000):
    """
    Simulated annealing to minimize max autocorrelation.
    """
    current_score = _autocorr_max_abs(x)
    best_score = current_score
    x = x.copy()
    best_x = x.copy()
    
    for it in range(max_iter):
        # Cooling schedule
        temp = start_temp * (end_temp / start_temp) ** (it / max_iter)
        
        # Random flip
        i = np.random.randint(N)
        old_val = x[i]
        x[i] = -old_val
        new_score = _autocorr_max_abs(x)
        
        delta = new_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
        else:
            x[i] = old_val
    
    return best_x, best_score


def construct():
    """
    Construct a signed step function of length N=400 that minimizes the
    upper bound on C3 (max absolute autocorrelation).
    
    Returns a numpy array of length N with values in {-1, +1}.
    """
    start_time = time.time()
    
    # Set random seed for reproducibility within this run
    np.random.seed(42)
    
    # Step 1: Initial guess from van der Corput sequence
    x = _initial_guess_vdc()
    best_x = x.copy()
    best_score = _autocorr_max_abs(x)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_x
    
    # Step 2: Greedy polish
    x, score = _greedy_polish(x, max_iter=20000)
    if score < best_score:
        best_score = score
        best_x = x.copy()
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_x
    
    # Step 3: Simulated annealing with multiple restarts
    # We'll do several short annealing runs from different starting points
    anneal_configs = [
        {'start_temp': 0.5, 'end_temp': 0.01, 'max_iter': 50000},
        {'start_temp': 1.0, 'end_temp': 0.005, 'max_iter': 50000},
        {'start_temp': 0.3, 'end_temp': 0.005, 'max_iter': 30000},
    ]
    
    for config in anneal_configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:  # Leave 10s buffer
            break
        
        # Start from best_x found so far
        x_anneal, score_anneal = _simulated_annealing(
            best_x, 
            start_temp=config['start_temp'], 
            end_temp=config['end_temp'], 
            max_iter=config['max_iter']
        )
        if score_anneal < best_score:
            best_score = score_anneal
            best_x = x_anneal.copy()
        
        # Greedy polish after annealing
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        x_polished, score_polished = _greedy_polish(best_x, max_iter=10000)
        if score_polished < best_score:
            best_score = score_polished
            best_x = x_polished.copy()
    
    # Final check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_x
    
    return best_x


# For testing / standalone execution
if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    score = _autocorr_max_abs(result)
    print(f"Result shape: {result.shape}")
    print(f"Min: {np.min(result)}, Max: {np.max(result)}")
    print(f"Score (max |autocorr|): {score}")
    print(f"Time: {elapsed:.2f}s")
