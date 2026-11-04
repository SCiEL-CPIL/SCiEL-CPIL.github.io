"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _max_abs_autocorr(f):
    """Compute max |autocorrelation| for non-zero lags using FFT."""
    n = len(f)
    size = 1
    while size < 2 * n:
        size <<= 1
    f_pad = np.zeros(size)
    f_pad[:n] = f
    F = np.fft.rfft(f_pad)
    acf = np.fft.irfft(F * np.conj(F), size)
    # acf[0] is the DC term (sum f[i]^2 = n for ±1)
    # Non-zero lags: acf[1:n] and acf[size-n+1:size] (wrap-around)
    # For aperiodic autocorrelation with zero-padding:
    # acf[k] for k=1..n-1 is sum_{i=0}^{n-1-k} f[i]*f[i+k]
    # acf[size-k] for k=1..n-1 is sum_{i=k}^{n-1} f[i]*f[i-k]
    # We want max of |acf[k]| for all non-zero lags
    if n > 1:
        # Positive lags
        pos_lags = acf[1:n]
        # Negative lags (stored at the end due to FFT)
        neg_lags = acf[size-n+1:size]
        return float(np.max(np.abs(np.concatenate([pos_lags, neg_lags]))))
    return 0.0


def _max_abs_autocorr_vectorized(f):
    """Faster version: compute all autocorrelations and return max abs."""
    n = len(f)
    size = 1
    while size < 2 * n:
        size <<= 1
    f_pad = np.zeros(size)
    f_pad[:n] = f
    F = np.fft.rfft(f_pad)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, size)
    if n > 1:
        pos = acf[1:n]
        neg = acf[size-n+1:size]
        return float(np.max(np.abs(pos))) if len(pos) > 0 else 0.0
    return 0.0


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Use a good initial seed
    rng = np.random.RandomState(12345)
    
    best_f = None
    best_score = float('inf')
    
    # Multiple restarts with simulated annealing
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize: random ±1 sequence
        f = rng.choice([-1.0, 1.0], size=N)
        
        # Simulated annealing parameters
        T = 5.0  # Initial temperature
        T_min = 0.01
        cooling_rate = 0.995
        max_iters_per_restart = 50000
        
        current_score = _max_abs_autocorr(f)
        best_f_local = f.copy()
        best_score_local = current_score
        
        iters_this_restart = 0
        stagnant = 0
        
        for _ in range(max_iters_per_restart):
            if time.time() > deadline:
                break
            
            # Random perturbation: flip k random bits
            k = rng.randint(1, 6)  # Flip 1 to 5 bits
            indices = rng.choice(N, size=k, replace=False)
            f[indices] *= -1
            
            new_score = _max_abs_autocorr(f)
            
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
                current_score = new_score
                if new_score < best_score_local:
                    best_score_local = new_score
                    best_f_local = f.copy()
                    stagnant = 0
                else:
                    stagnant += 1
            else:
                # Reject: undo the flip
                f[indices] *= -1
                stagnant += 1
            
            # Cool down
            T *= cooling_rate
            
            iters_this_restart += 1
            
            # Early exit if stuck
            if stagnant > 5000 and T < 0.1:
                break
        
        # Update global best
        if best_score_local < best_score:
            best_score = best_score_local
            best_f = best_f_local
    
    # Final local polish on best solution
    if best_f is not None and time.time() < deadline:
        f = best_f.copy()
        current_score = _max_abs_autocorr(f)
        
        # Intensive local search: try single-bit flips
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                f[i] *= -1
                new_score = _max_abs_autocorr(f)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    f[i] *= -1
        
        if current_score < best_score:
            best_score = current_score
            best_f = f
    
    return best_f
# EVOLVE-BLOCK-END
