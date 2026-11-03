# EVOLVE-BLOCK-START
import numpy as np
import math

def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    Uses Logistic Map for initialization and Simulated Annealing for optimization.
    """
    N = 400
    
    def compute_c3_bound(s):
        """
        Compute the upper bound on C3.
        """
        N_pad = 1 << (3 * N - 1).bit_length()
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))

    def logistic_map(x, r=4.0):
        return r * x * (1.0 - x)

    # Generate initial candidates
    seeds = [
        (np.sqrt(2) - 1) % 1.0,
        (np.sqrt(3) - 1) % 1.0,
        (np.pi - 3) % 1.0,
        (np.e - 2) % 1.0,
        (1/np.sqrt(5)) % 1.0,
        (np.sqrt(5) - 2) % 1.0,
        0.123456789012345,
        0.987654321098765,
        (np.sqrt(2) * np.pi) % 1.0,
        (np.log(2) * np.e) % 1.0
    ]
    
    best_signs = None
    best_bound = float('inf')
    
    for seed in seeds:
        x = seed
        signs = np.zeros(N)
        for i in range(N):
            x = logistic_map(x)
            signs[i] = 1.0 if x > 0.5 else -1.0
            
        bound = compute_c3_bound(signs)
        
        if bound < best_bound:
            best_bound = bound
            best_signs = signs.copy()
            
    # Simulated Annealing
    signs = best_signs.copy()
    current_bound = best_bound
    best_bound_sa = current_bound
    best_signs_sa = signs.copy()
    
    # SA Parameters
    T_start = 5.0
    T_stop = 0.05
    cooling_rate = 0.995
    T = T_start
    
    max_evals = 100000
    evals = 0
    steps_per_temp = 10 # 10 flips per temp step
    
    while T > T_stop and evals < max_evals:
        for _ in range(steps_per_temp):
            if evals >= max_evals:
                break
            
            # Flip 1 or 2 bits
            if np.random.rand() < 0.5:
                i1 = np.random.randint(N)
                signs[i1] *= -1.0
                new_bound = compute_c3_bound(signs)
                evals += 1
                delta = new_bound - current_bound
                if delta < 0 or np.random.rand() < math.exp(-delta / T):
                    current_bound = new_bound
                    if new_bound < best_bound_sa:
                        best_bound_sa = new_bound
                        best_signs_sa = signs.copy()
                else:
                    signs[i1] *= -1.0
            else:
                i1 = np.random.randint(N)
                i2 = np.random.randint(N)
                if i1 == i2:
                    continue
                signs[i1] *= -1.0
                signs[i2] *= -1.0
                new_bound = compute_c3_bound(signs)
                evals += 1
                delta = new_bound - current_bound
                if delta < 0 or np.random.rand() < math.exp(-delta / T):
                    current_bound = new_bound
                    if new_bound < best_bound_sa:
                        best_bound_sa = new_bound
                        best_signs_sa = signs.copy()
                else:
                    signs[i1] *= -1.0
                    signs[i2] *= -1.0
                    
        T *= cooling_rate
        
    return best_signs_sa
# EVOLVE-BLOCK-END
