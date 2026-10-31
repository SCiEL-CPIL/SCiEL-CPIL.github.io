import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Initialize with random +/- 1
    # Use a fixed seed for reproducibility? No, random is fine.
    f = np.random.choice([-1.0, 1.0], N)
    best_f = f.copy()
    
    # Calculate initial score
    def get_score(f):
        # Aperiodic autocorrelation
        n = len(f)
        f_pad = np.zeros(2 * n)
        f_pad[:n] = f
        F = np.fft.fft(f_pad)
        R = np.fft.ifft(F * np.conj(F)).real[:n]
        return np.sum(np.abs(R))
        
    best_score = get_score(best_f)
    
    # Hill Climbing
    step = 0
    while time.time() < deadline:
        idx = np.random.randint(N)
        f[idx] *= -1.0
        
        new_score = get_score(f)
        
        if new_score < best_score:
            best_score = new_score
            best_f = f.copy()
        else:
            f[idx] *= -1.0
            
        step += 1
        if step % 1000 == 0:
            # Optional: Check time more frequently
            pass
            
    return best_f
