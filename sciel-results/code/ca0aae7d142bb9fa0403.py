import numpy as np
import time

N = 400

def _autoconv(x):
    """Compute full autoconvolution of x."""
    n = len(x)
    # Use FFT for efficiency
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * np.conj(X), size)[:2 * n - 1]
    return ac

def _score(x):
    """Compute max|autoconv| / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)
    
    # Start with a random ±1 sequence
    np.random.seed(42)
    
    # Try several initial random sequences
    for init_trial in range(20):
        if time.time() > deadline - 30:
            break
        x = np.random.choice([-1.0, 1.0], size=N)
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
    
    # Simulated annealing / local search
    x = best_x.copy()
    current_score = best_score
    
    T = 1.0
    T_min = 1e-6
    cooling = 0.9995
    
    while time.time() < deadline - 5.0:
        if T < T_min:
            break
        
        # Randomly flip a few positions
        n_flips = np.random.randint(1, 6)
        indices = np.random.choice(N, size=n_flips, replace=False)
        x_new = x.copy()
        x_new[indices] *= -1.0
        
        new_score = _score(x_new)
        
        if new_score < current_score or np.random.random() < np.exp(-(new_score - current_score) / max(T, 1e-10)):
            x = x_new
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
        
        T *= cooling
        
        # Occasionally restart from best
        if np.random.random() < 0.01:
            x = best_x.copy()
            current_score = best_score
    
    # Final local refinement: single-flip search
    x = best_x.copy()
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
            x_old = x[i]
            x[i] = -x[i]
            sc = _score(x)
            if sc < best_score:
                best_score = sc
                best_x = x.copy()
                improved = True
            else:
                x[i] = x_old
    
    return best_x
