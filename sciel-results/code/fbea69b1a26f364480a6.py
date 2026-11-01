import numpy as np
import time

N = 400
DEADLINE = 145.0  # safety margin before 150s kill

def _autocorr(h):
    """Compute full autocorrelation of a 1D array via FFT."""
    n = len(h)
    # Zero-pad to avoid circular effects
    f = np.fft.rfft(h, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f), n=2 * n)[:n]
    return ac

def _objective(h):
    """Upper bound on C3: use L1 norm of autocorrelation (excluding DC)."""
    ac = _autocorr(h)
    # Exclude the DC term (lag 0) which is always N
    return np.sum(np.abs(ac)) - ac[0]

def _flip(h, idx):
    """Flip sign at position idx, return new array."""
    h_new = h.copy()
    h_new[idx] *= -1
    return h_new

def construct():
    start_time = time.time()
    best_h = np.ones(N, dtype=np.int8)
    best_val = _objective(best_h)
    
    # Try a few random initializations
    for trial in range(3):
        if time.time() - start_time > DEADLINE:
            break
        h = np.ones(N, dtype=np.int8)
        # Random signs
        flips = np.random.choice(N, size=N // 2, replace=False)
        h[flips] = -1
        val = _objective(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Simulated annealing
    h = best_h.copy()
    current_val = best_val
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < DEADLINE:
        # Flip a random position
        idx = np.random.randint(N)
        h_new = h.copy()
        h_new[idx] *= -1
        new_val = _objective(h_new)
        
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            h = h_new
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
        
        T = max(T * cooling, T_min)
        
        # Occasionally do a 2-flip move
        if np.random.random() < 0.3:
            i1 = np.random.randint(N)
            i2 = np.random.randint(N)
            if i1 != i2:
                h_new = h.copy()
                h_new[i1] *= -1
                h_new[i2] *= -1
                new_val = _objective(h_new)
                delta = new_val - current_val
                if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                    h = h_new
                    current_val = new_val
                    if current_val < best_val:
                        best_val = current_val
                        best_h = h.copy()
    
    # Final local search polish
    if time.time() - start_time < DEADLINE - 5:
        improved = True
        while improved and time.time() - start_time < DEADLINE - 5:
            improved = False
            for idx in range(N):
                if time.time() - start_time > DEADLINE - 5:
                    break
                h_new = h.copy()
                h_new[idx] *= -1
                new_val = _objective(h_new)
                if new_val < current_val:
                    h = h_new
                    current_val = new_val
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_h = h.copy()
    
    return best_h.astype(np.float64)
