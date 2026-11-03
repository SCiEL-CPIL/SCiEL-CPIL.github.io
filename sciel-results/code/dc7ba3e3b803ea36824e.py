# EVOLVE-BLOCK-START
import numpy as np
import math
import random
import time

N = 400
N_pad = 1 << (3 * N - 1).bit_length()

def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    Uses Simulated Annealing with an optimized FFT-based evaluation loop
    that maintains the frequency spectrum to avoid redundant forward FFTs.
    """
    
    def compute_c3_bound_from_spectrum(S):
        """
        Compute the C3 bound from the pre-computed FFT spectrum S.
        """
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))

    def compute_initial_spectrum(s):
        """
        Compute the FFT spectrum of the padded sequence s.
        """
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        return np.fft.fft(s_pad)

    def get_phase_factors(i):
        """
        Pre-computed or on-the-fly phase factors for index i.
        phase_k = exp(-1j * 2 * pi * k * i / N_pad)
        """
        k = np.arange(N_pad)
        return np.exp(-1j * 2 * np.pi * k * i / N_pad)

    # 1. Generate initial candidates
    seeds = [
        np.sqrt(2) % 1.0,
        np.sqrt(3) % 1.0,
        np.pi % 1.0,
        (1 + np.sqrt(5)) / 2 % 1.0,
        np.e % 1.0,
        0.123456789,
        0.987654321,
        0.5 + (1/np.sqrt(2)) % 1.0,
        0.33333333,
        0.66666666
    ]
    
    best_signs = None
    best_bound = float('inf')
    best_spectrum = None
    
    for seed in seeds:
        x = seed
        signs = np.zeros(N)
        for i in range(N):
            x = 4.0 * x * (1.0 - x)
            signs[i] = 1.0 if x > 0.5 else -1.0
            
        S = compute_initial_spectrum(signs)
        bound = compute_c3_bound_from_spectrum(S)
        
        if bound < best_bound:
            best_bound = bound
            best_signs = signs.copy()
            best_spectrum = S.copy()
            
    # 2. Simulated Annealing with Optimized Evaluation
    signs = best_signs
    S = best_spectrum
    current_bound = best_bound
    
    T_start = 5.0  # Higher start temp to explore more
    T_end = 0.01
    alpha = 0.9995 # Slower cooling for longer search
    max_evals = 200000 # Increased evals due to speedup
    
    # Pre-compute phase factors for all indices to speed up updates
    # phase_factors[i] is the vector of exp(-1j * 2 * pi * k * i / N_pad) for k=0..N_pad-1
    # This is a N x N_pad matrix. 400 * 4096 * 8 bytes (complex128) = ~13 MB. Acceptable.
    phase_factors = np.zeros((N, N_pad), dtype=np.complex128)
    k = np.arange(N_pad)
    for i in range(N):
        phase_factors[i] = np.exp(-1j * 2 * np.pi * k * i / N_pad)

    random.seed(42)
    
    start_time = time.time()
    evals = 0
    T = T_start
    
    while T > T_end and evals < max_evals:
        # Check time limit (leave 5s for final greedy)
        if time.time() - start_time > 100:
            break
            
        i = random.randint(0, N - 1)
        
        # Calculate new spectrum
        # S_new = S - 2 * s[i] * phase_factors[i]
        delta_S = -2.0 * signs[i] * phase_factors[i]
        S_new = S + delta_S
        
        # Evaluate new bound
        C3_new = np.fft.ifft(S_new ** 3).real
        new_bound = np.max(np.abs(C3_new))
        evals += 1
        
        delta = new_bound - current_bound
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            signs[i] *= -1.0
            S = S_new
            current_bound = new_bound
        else:
            # Revert (S and signs are unchanged)
            pass
            
        T *= alpha
        
    # 3. Final Greedy Pass (Polish)
    # Use the optimized evaluation here too
    improved = True
    while improved:
        improved = False
        for i in range(N):
            delta_S = -2.0 * signs[i] * phase_factors[i]
            S_new = S + delta_S
            C3_new = np.fft.ifft(S_new ** 3).real
            new_bound = np.max(np.abs(C3_new))
            
            if new_bound < current_bound:
                signs[i] *= -1.0
                S = S_new
                current_bound = new_bound
                improved = True
            # else: revert (do nothing)
                
    return signs
# EVOLVE-BLOCK-END
