import numpy as np
import time

N = 400

def _compute_ac(h):
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _score_from_ac(ac, S):
    if abs(S) < 1e-9:
        return 1e10
    return np.max(np.abs(ac)) / (S * S)

def construct():
    t_start = time.time()
    deadline = t_start + 115.0
    
    # Initialize h
    rng = np.random.default_rng(42)
    h = rng.choice([-1, 1], size=N)
    
    S = np.sum(h)
    ac = _compute_ac(h)
    current_score = _score_from_ac(ac, S)
    
    best_h = h.copy()
    best_score = current_score
    
    # Pre-compute k
    k = np.arange(2 * N - 1)
    
    T_start = 1e-4
    T_end = 1e-8
    n_steps = 800000 # Higher steps with O(N) update
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
            
        progress = step / n_steps
        T = T_start * (T_end / T_start) ** progress
        
        j = rng.integers(0, N)
        v = h[j]
        
        # Efficient delta calculation
        # mask1: j + k < N  => k < N - j
        # mask2: j - k >= 0 => k <= j
        
        # We can create the delta_ac array directly
        delta_ac = np.zeros(2 * N - 1)
        
        # Add h[j+k] for valid k
        # Valid k for j+k: 0 <= k < N-j
        if N - j > 0:
            # idx1 = j + k
            # We need h[j+k]
            # k range: 0 to N-j-1
            # delta_ac[0:N-j] += h[j : N]
            delta_ac[:N-j] += h[j:]
            
        # Add h[j-k] for valid k
        # Valid k for j-k: 0 <= k <= j
        # idx2 = j - k
        # We need h[j-k]
        # k range: 0 to j
        # When k=0, idx=j. When k=j, idx=0.
        # So we are adding h[j], h[j-1], ..., h[0] to delta_ac[0], delta_ac[1], ..., delta_ac[j]
        # delta_ac[0:j+1] += h[j:-1:-1]
        if j >= 0:
            delta_ac[:j+1] += h[j:-1:-1]
            
        delta_ac *= -2 * v
        delta_ac[0] = 0
        
        new_ac = ac + delta_ac
        new_S = S - 2 * v
        
        new_score = _score_from_ac(new_ac, new_S)
        delta_score = new_score - current_score
        
        if delta_score < 0 or rng.random() < np.exp(-delta_score / T):
            h[j] = -v
            S = new_S
            ac = new_ac
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            pass
            
    return best_h
