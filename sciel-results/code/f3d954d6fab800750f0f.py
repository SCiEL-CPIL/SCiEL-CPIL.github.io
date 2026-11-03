import numpy as np
import time

N = 400

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    
    def score(x):
        # Compute linear autocorrelation
        n = len(x)
        # autocorr[k] = sum_{i=0}^{n-1-k} x[i] * x[i+k]
        # Use FFT for efficiency
        # Zero-pad to 2n-1
        size = 2 * n - 1
        x_padded = np.zeros(size)
        x_padded[:n] = x
        X = np.fft.fft(x_padded)
        autocorr = np.fft.ifft(X * np.conj(X)).real[:n]
        max_abs = np.max(np.abs(autocorr))
        integral = np.sum(x)
        if abs(integral) < 1e-10:
            return float('inf')
        return max_abs / (integral * integral)
    
    best_x = np.ones(N)
    best_score = score(best_x)
    
    # Try random ±1 sequences
    rng = np.random.default_rng(42)
    for trial in range(5000):
        if time.time() > deadline - 5.0:
            break
        x = rng.choice([-1, 1], size=N)
        s = score(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()
    
    # Try local search from best found
    if time.time() < deadline - 10.0:
        x = best_x.copy()
        for step in range(100000):
            if time.time() > deadline - 5.0:
                break
            idx = rng.integers(0, N)
            x[idx] *= -1  # flip sign
            s = score(x)
            if s < best_score:
                best_score = s
                best_x = x.copy()
            else:
                x[idx] *= -1  # revert
    
    return best_x
