"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    Uses Logistic Map with irrational seeds for initialization.
    Uses an optimized greedy local search with incremental FFT updates to minimize
    the max absolute value of the third-order autocorrelation (C3 bound).
    """
    
    N_pad = 1 << (3 * N - 1).bit_length()
    # Precompute twiddle factors for incremental FFT update
    # omega = exp(-2j * pi / N_pad)
    # We need omega**i for i in range(N)
    # s is real, so we use rfft? 
    # If we use rfft, the update is slightly more complex due to symmetry.
    # Let's stick to complex fft for simplicity of update, N_pad=2048 is small.
    omega = np.exp(-2j * np.pi / N_pad)
    # Precompute powers of omega for indices 0..N-1
    omega_powers = omega ** np.arange(N)
    
    def compute_c3_bound_from_fft(S):
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))

    def generate_signs(seed):
        x = seed
        signs = np.zeros(N)
        for i in range(N):
            x = 4.0 * x * (1.0 - x)
            signs[i] = 1.0 if x > 0.5 else -1.0
        return signs

    seeds = [
        np.sqrt(2) % 1.0,
        np.sqrt(3) % 1.0,
        np.pi % 1.0,
        (1 + np.sqrt(5)) / 2 % 1.0,
        np.e % 1.0
    ]
    
    best_signs = None
    best_bound = float('inf')
    best_S = None
    
    # 1. Initialize with best of several chaotic seeds
    for seed in seeds:
        signs = generate_signs(seed)
        S = np.fft.fft(signs, N_pad)
        bound = compute_c3_bound_from_fft(S)
        if bound < best_bound:
            best_bound = bound
            best_signs = signs.copy()
            best_S = S.copy()
            
    signs = best_signs
    S = best_S
    current_bound = best_bound
    
    # 2. Optimized Greedy Local Search
    # Time limit: 110 seconds
    start_time = time.time()
    max_time = 110.0
    
    # Random number generator for restarts
    import random
    random.seed(42)
    
    while (time.time() - start_time) < max_time:
        improved = False
        
        # Greedy Pass
        for i in range(N):
            # Flip sign at i
            old_val = signs[i]
            signs[i] = -old_val
            
            # Update FFT incrementally
            # S_new = S_old - 2 * old_val * omega**i
            delta_S = -2.0 * old_val * omega_powers[i]
            S += delta_S
            
            # Compute new bound
            new_bound = compute_c3_bound_from_fft(S)
            
            if new_bound < current_bound:
                current_bound = new_bound
                improved = True
            else:
                # Revert
                signs[i] = old_val
                S -= delta_S
        
        if not improved:
            # Random Restart: Flip a random few bits to escape local minimum
            # Flip 1 bit
            j = random.randint(0, N - 1)
            old_val = signs[j]
            signs[j] = -old_val
            delta_S = -2.0 * old_val * omega_powers[j]
            S += delta_S
            # We don't update current_bound here, let the next greedy pass find the new local min
            # Or we can compute it to see if it's worse, but we accept it to escape.
            
    return signs
# EVOLVE-BLOCK-END
