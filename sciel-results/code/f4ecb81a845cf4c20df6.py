"""Improved program for the third autocorrelation inequality task."""
import numpy as np
import time

N = 400
DEADLINE = 145  # safety margin before 150s kill


def _autocorr_max_abs(s):
    """Compute max |autocorrelation| over non-zero lags for a length-N ±1 sequence."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    # acf[k] = sum_{i=0}^{N-1-k} s[i]*s[i+k] for k=0..N-1
    # We want max |acf[k]| for k=1..N-1
    if N <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:])))


def _local_search(s, max_time=5.0):
    """Greedy local search: flip individual elements to reduce max |autocorr|."""
    s = s.copy()
    current_val = _autocorr_max_abs(s)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        # Try flipping each element
        best_gain = 0
        best_idx = -1
        for i in range(N):
            s[i] *= -1
            new_val = _autocorr_max_abs(s)
            s[i] *= -1  # revert
            gain = current_val - new_val
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx >= 0:
            s[best_idx] *= -1
            current_val -= best_gain
            improved = True
    return s, current_val


def _simulated_annealing(s, max_time=5.0):
    """Simulated annealing on ±1 sequence."""
    s = s.copy()
    current_val = _autocorr_max_abs(s)
    best_s = s.copy()
    best_val = current_val
    start = time.time()
    T = 10.0
    cooling = 0.99
    while (time.time() - start) < max_time:
        # Random flip
        i = np.random.randint(N)
        s[i] *= -1
        new_val = _autocorr_max_abs(s)
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_s = s.copy()
        else:
            s[i] *= -1  # revert
        T *= cooling
        if T < 0.01:
            T = 0.01
    return best_s, best_val


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    np.random.seed(42)
    
    best_s = None
    best_val = float('inf')
    
    # Try multiple random restarts with different strategies
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Generate random ±1 sequence
        s = np.random.choice([-1.0, 1.0], size=N)
        
        # Quick local search
        s_ls, val_ls = _local_search(s, max_time=3.0)
        if val_ls < best_val:
            best_val = val_ls
            best_s = s_ls.copy()
        
        if time.time() > deadline - 15:
            break
        
        # Simulated annealing on the locally optimized result
        s_sa, val_sa = _simulated_annealing(s_ls, max_time=3.0)
        if val_sa < best_val:
            best_val = val_sa
            best_s = s_sa.copy()
    
    # Final local search on the best found
    if best_s is not None and time.time() < deadline - 5:
        best_s, best_val = _local_search(best_s, max_time=5.0)
    
    if best_s is None:
        best_s = np.ones(N)
    
    return best_s
