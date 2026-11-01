import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    # Compute max autocorrelation efficiently using FFT
    def max_autocorr(h):
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        return np.max(acf[:N])
    
    # Start with a random ±1 sequence
    rng = np.random.RandomState(42)
    best_h = rng.choice([1, -1], N)
    best_val = max_autocorr(best_h)
    
    # Multiple restarts with local search
    for seed in range(50):
        if time.time() >= deadline - 15.0:
            break
        
        rng = np.random.RandomState(seed)
        current_h = rng.choice([1, -1], N)
        current_val = max_autocorr(current_h)
        
        if current_val < best_val:
            best_h = current_h.copy()
            best_val = current_val
        
        # Local search from this starting point
        no_improve = 0
        for _ in range(10000):
            if time.time() >= deadline - 15.0:
                break
            idx = rng.randint(N)
            trial = current_h.copy()
            trial[idx] = -trial[idx]
            val = max_autocorr(trial)
            if val < current_val:
                current_val = val
                current_h = trial
                no_improve = 0
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                no_improve += 1
                if no_improve > 1000:
                    break
    
    return best_h
# EVOLVE-BLOCK-END
