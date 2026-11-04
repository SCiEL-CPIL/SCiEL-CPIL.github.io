import numpy as np
import time

N = 400

def _autocorr_peak(h):
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H)**2, n=fft_size)
    return np.max(np.abs(acf[1:n]))

def construct():
    start_time = time.time()
    deadline = start_time + 140
    polish_start = start_time + 128
    
    np.random.seed(42)
    
    def random_balanced():
        h = np.random.choice([-1.0, 1.0], size=N)
        n_pos = int(np.sum(h == 1.0))
        target = N // 2
        if n_pos > target:
            idx = np.where(h == 1.0)[0]
            np.random.shuffle(idx)
            h[idx[:n_pos - target]] = -1.0
        elif n_pos < target:
            idx = np.where(h == -1.0)[0]
            np.random.shuffle(idx)
            h[idx[:target - n_pos]] = 1.0
        return h
    
    # Multiple random initializations
    best_h = None
    best_peak = np.inf
    
    for _ in range(20):
        h = random_balanced()
        peak = _autocorr_peak(h)
        if peak < best_peak:
            best_peak = peak
            best_h = h.copy()
    
    # Simulated annealing
    h = best_h.copy()
    current_peak = best_peak
    T = 10.0
    min_T = 1e-3
    cooling = 0.9998
    
    while time.time() < polish_start:
        pos = np.random.randint(N)
        v = h[pos]
        h[pos] = -v
        new_peak = _autocorr_peak(h)
        
        if new_peak < current_peak:
            current_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
                best_h = h.copy()
        elif T > min_T and np.random.random() < np.exp((current_peak - new_peak) / T):
            current_peak = new_peak
        else:
            h[pos] = v
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Best flip local search
    h = best_h.copy()
    current_peak = best_peak
    for _ in range(500):
        if time.time() > deadline:
            break
        improved = False
        for pos in range(N):
            v = h[pos]
            h[pos] = -v
            new_peak = _autocorr_peak(h)
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_h = h.copy()
            else:
                h[pos] = v
        if not improved:
            break
    
    return best_h
