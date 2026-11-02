import numpy as np
import time

N = 400

def _peak_sidelobe(x):
    ac = np.correlate(x, x, mode='full')
    return np.max(np.abs(ac[N:2*N-1]))

def _greedy_polish(x):
    current_peak = _peak_sidelobe(x)
    improved = True
    max_iters = 100
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        best_peak = current_peak
        
        # Precompute ac
        ac = np.correlate(x, x, mode='full')
        sidelobes = ac[N:2*N-1]
        
        # Vectorized computation of change in autocorrelations for all flips
        i_idx = np.arange(N)[:, None] 
        k_idx = np.arange(1, N)[None, :]
        
        idx1 = i_idx + k_idx
        idx2 = i_idx - k_idx
        
        mask1 = (idx1 >= 0) & (idx1 < N)
        mask2 = (idx2 >= 0) & (idx2 < N)
        
        idx1_safe = np.where(mask1, idx1, 0)
        idx2_safe = np.where(mask2, idx2, 0)
        
        x_i = x[:, None]
        x_plus = x[idx1_safe]
        x_minus = x[idx2_safe]
        
        x_plus = np.where(mask1, x_plus, 0)
        x_minus = np.where(mask2, x_minus, 0)
        
        D = -2 * x_i * (x_plus + x_minus)
        
        S = sidelobes[None, :] + D
        peaks = np.max(np.abs(S), axis=1)
        
        best_i = np.argmin(peaks)
        best_peak = peaks[best_i]
        
        if best_peak < current_peak:
            x[best_i] *= -1
            current_peak = best_peak
            improved = True
            
    return x, current_peak

def _relax_and_discretize(x0, steps=50, lr=0.1):
    x = x0.astype(float)
    for _ in range(steps):
        ac = np.correlate(x, x, mode='full')
        sidelobes = ac[N:2*N-1]
        
        g = np.zeros(N)
        for j in range(N):
            k = np.arange(1, N)
            idx1 = j + k
            idx2 = j - k
            mask1 = idx1 < N
            mask2 = idx2 >= 0
            
            term1 = np.where(mask1, sidelobes * x[np.where(mask1, idx1, 0)], 0)
            term2 = np.where(mask2, sidelobes * x[np.where(mask2, idx2, 0)], 0)
            g[j] = 2 * (np.sum(term1) + np.sum(term2))
            
        x -= lr * g
        
    return np.sign(x)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_x = None
    best_score = np.inf
    
    np.random.seed(42)
    
    while time.time() < deadline:
        x = np.random.choice([-1, 1], size=N)
        
        x_relaxed = _relax_and_discretize(x, steps=50, lr=0.1)
        x_final, score = _greedy_polish(x_relaxed)
        
        if score < best_score:
            best_score = score
            best_x = x_final.copy()
            
        if best_score < 20:
            break

    if best_x is None:
        return np.ones(N)
        
    return best_x
