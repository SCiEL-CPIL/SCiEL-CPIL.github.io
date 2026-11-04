import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 100.0  # 20s buffer
    
    # Initialize
    f = np.random.choice([-1, 1], size=N)
    
    def calc_peak(f):
        n_fft = 2 * N
        F = np.fft.fft(f, n=n_fft)
        power = F * np.conj(F)
        corr = np.fft.ifft(power).real
        return np.max(np.abs(corr))
    
    current_peak = calc_peak(f)
    best_peak = current_peak
    best_f = f.copy()
    
    T = 1.0
    cooling = 0.9995
    min_T = 1e-5
    no_improve_count = 0
    patience = 1000
    
    while time.time() - start_time < deadline:
        if T < min_T:
            break
            
        # Random move
        i = np.random.randint(N)
        f[i] *= -1
        new_peak = calc_peak(f)
        
        delta = new_peak - current_peak
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
                best_f = f.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            f[i] *= -1
            no_improve_count += 1
            
        T *= cooling
        
        # Kick if stuck
        if no_improve_count > patience:
            # Flip 10% of indices
            num_flips = N // 10
            indices = np.random.choice(N, num_flips, replace=False)
            f[indices] *= -1
            current_peak = calc_peak(f)
            if current_peak < best_peak:
                best_peak = current_peak
                best_f = f.copy()
            no_improve_count = 0
            T = 1.0 # Reset temperature
            
    return best_f
