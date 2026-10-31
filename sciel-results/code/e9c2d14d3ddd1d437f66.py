"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import time
import numpy as np

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Objective: Minimize sum of squares of non-zero autocorrelation lags
    # Metric: sum(R[1:]**2)
    
    def compute_initial_autocorr(f):
        n_fft = 2 * N
        f_ext = np.zeros(n_fft)
        f_ext[:N] = f
        F = np.fft.rfft(f_ext)
        power = F * np.conj(F)
        autocorr = np.fft.irfft(power, n=n_fft)[:N]
        return autocorr

    def get_score(R):
        return np.sum(R[1:] ** 2)

    rng = np.random.RandomState(42)
    
    best_score = float('inf')
    best_f = None
    best_R = None

    # Multiple restarts
    num_restarts = 50
    max_steps_per_restart = 50000
    
    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break
            
        # Initialize random sequence
        f = rng.choice([-1.0, 1.0], size=N)
        R = compute_initial_autocorr(f)
        current_score = get_score(R)
        
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
            best_R = R.copy()
            
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.9999
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline - 2:
                break
                
            i = rng.randint(N)
            val = f[i]
            
            # Calculate delta R
            # delta[k] = -2 * val * f[i+k] (if i+k < N) - 2 * f[i-k] * val (if i-k >= 0)
            # Note: f[i] is currently 'val'. We are flipping it to -val.
            # The change in R[k] is New - Old.
            # Old terms involving f[i]: val * f[i+k] and f[i-k] * val.
            # New terms involving -val: -val * f[i+k] and f[i-k] * (-val).
            # Diff: -2 * val * f[i+k] and -2 * f[i-k] * val.
            
            delta_R = np.zeros(N)
            
            # Part 1: i is the first index. k such that i+k < N.
            # k in [0, N-1-i]
            if N - i > 0:
                k1 = np.arange(0, N - i)
                delta_R[k1] -= 2 * val * f[i + k1]
                
            # Part 2: i is the second index. k such that i-k >= 0.
            # k in [0, i]
            if i >= 0:
                k2 = np.arange(0, i + 1)
                delta_R[k2] -= 2 * f[i - k2] * val
                
            # Correction for k=0: The term is f[i]^2.
            # Old: val^2. New: (-val)^2. Diff 0.
            # Our formula for k=0:
            # Part 1 (k=0): -2 * val * f[i] = -2 * val^2 = -2.
            # Part 2 (k=0): -2 * f[i] * val = -2 * val^2 = -2.
            # Total delta[0] = -4.
            # We need delta[0] = 0. So add 4.
            delta_R[0] += 4.0

            R_new = R + delta_R
            score_new = get_score(R_new)
            delta_e = score_new - current_score
            
            if delta_e < 0 or (T > 0 and rng.rand() < np.exp(-delta_e / T)):
                # Accept
                f[i] = -val
                R = R_new
                current_score = score_new
                
                if current_score < best_score:
                    best_score = current_score
                    best_f = f.copy()
                    best_R = R.copy()
            else:
                # Reject
                pass # f and R unchanged
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # Hill Climbing Polish
        improved = True
        hc_count = 0
        max_hc_steps = 10000
        while improved and hc_count < max_hc_steps and time.time() < deadline - 2:
            improved = False
            i = rng.randint(N)
            val = f[i]
            
            delta_R = np.zeros(N)
            if N - i > 0:
                k1 = np.arange(0, N - i)
                delta_R[k1] -= 2 * val * f[i + k1]
            if i >= 0:
                k2 = np.arange(0, i + 1)
                delta_R[k2] -= 2 * f[i - k2] * val
            delta_R[0] += 4.0

            R_new = R + delta_R
            score_new = get_score(R_new)
            
            if score_new < current_score:
                f[i] = -val
                R = R_new
                current_score = score_new
                improved = True
                hc_count += 1
                if current_score < best_score:
                    best_score = current_score
                    best_f = f.copy()
                    best_R = R.copy()
            else:
                hc_count += 1

    if best_f is None:
        best_f = np.ones(N)
        
    return best_f

# EVOLVE-BLOCK-END
