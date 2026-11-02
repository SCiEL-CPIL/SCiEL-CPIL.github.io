import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = N / 2.0
TIME_LIMIT = 115.0

_START_TIME = None

def _time_remaining():
    if _START_TIME is None:
        return TIME_LIMIT
    return TIME_LIMIT - (time.time() - _START_TIME)

def _compute_overlaps(h):
    """
    Compute overlap for all shifts k.
    Overlap_k = sum_i h[i] * (1 - h[(i+k) % N])
    """
    H = np.fft.fft(h)
    # Autocorrelation C_k = sum_i h[i] h[(i+k)%N]
    # Note: np.fft.ifft(np.fft.fft(h) * np.fft.fft(h).conj()) gives circular autocorrelation
    autocorr = np.fft.ifft(H * np.conj(H)).real
    return HALF - autocorr

def _max_overlap_value(h):
    overlaps = _compute_overlaps(h)
    return float(np.max(overlaps))

def _project_to_simplex(v, target_sum):
    """
    Project v onto {x | 0 <= x <= 1, sum(x) = target_sum}.
    """
    n = len(v)
    if target_sum <= 0:
        return np.zeros(n)
    if target_sum >= n:
        return np.ones(n)
    
    w = np.zeros(n)
    active = np.ones(n, dtype=bool)
    v_active = v.copy()
    S_active = target_sum
    
    for _ in range(n):
        if not np.any(active):
            break
        v_act = v_active[active]
        S_act = S_active
        
        # Project v_act to {x >= 0, sum x = S_act}
        u = np.sort(v_act)[::-1]
        cssv = np.cumsum(u)
        rho_arr = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - S_act))[0]
        if len(rho_arr) == 0:
            theta = 0
        else:
            rho = rho_arr[-1]
            theta = (cssv[rho] - S_act) / (rho + 1.0)
            
        w_act = np.maximum(v_act - theta, 0)
        
        # Check for upper bound violation (> 1)
        viol = w_act > 1.0
        if np.any(viol):
            # Assign 1 to these
            idx_active = np.where(active)[0]
            idx_viol = idx_active[viol]
            w[idx_viol] = 1.0
            active[idx_viol] = False
            S_active -= np.count_nonzero(viol)
        else:
            w[active] = w_act
            break
            
    # Final adjustment for floating point errors
    diff = target_sum - np.sum(w)
    if abs(diff) > 1e-9:
        non_clipped = (w > 1e-6) & (w < 1 - 1e-6)
        if np.any(non_clipped):
            count = np.sum(non_clipped)
            w[non_clipped] += diff / count
            w = np.clip(w, 0, 1)
            
    # Ensure exact sum if possible (fix last element)
    if np.sum(w) != target_sum:
        diff = target_sum - np.sum(w)
        for i in range(n):
            if 0 < w[i] + diff < 1:
                w[i] += diff
                break
        if np.sum(w) != target_sum:
             w[0] += target_sum - np.sum(w)
             w = np.clip(w, 0, 1)
             
    return w

def _local_refine(h, max_iter=200, beta=10.0):
    """
    Refine h using gradient descent on soft max of overlaps.
    """
    current_score = _compute_max_overlap(h)
    lr = 0.05
    
    for i in range(max_iter):
        if _time_remaining() < 5:
            break
            
        g = 1.0 - h
        H = np.fft.fft(h)
        G = np.fft.fft(g)
        Corrs = np.fft.ifft(H * np.conj(G)).real
        
        # Soft max weights
        m = np.max(Corrs)
        exps = np.exp(beta * (Corrs - m))
        sum_exps = np.sum(exps)
        u = exps / sum_exps
        
        # Gradient of soft max
        U = np.fft.fft(u)
        D = np.fft.fft(1.0 - 2.0 * h)
        Grad = np.fft.ifft(U * D).real
        
        h_new = h - lr * Grad
        h_new = _project_to_simplex(h_new, HALF)
        
        score_new = _compute_max_overlap(h_new)
        
        if score_new < current_score:
            h = h_new
            current_score = score_new
            lr *= 0.99
        else:
            lr *= 0.9
            if lr < 1e-6:
                break
                
    return h, current_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Strategy:
    # 1. Try structured initializations
    # 2. Local refinement
    # 3. Brownian Bridge inspired random initializations
    
    candidates = []
    
    # 1. Constant
    candidates.append(np.full(N, 0.5))
    
    # 2. Sine waves
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        for phase in np.linspace(0, 2*np.pi, 8, endpoint=False):
            h = 0.5 + 0.3 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            h = _project_to_simplex(h, HALF)
            candidates.append(h)
            
    # 3. Random 0/1
    for _ in range(20):
        h = np.zeros(N)
        idx = np.random.choice(N, N//2, replace=False)
        h[idx] = 1.0
        candidates.append(h)
        
    # 4. Random uniform
    for _ in range(20):
        h = np.random.uniform(0, 1, N)
        h = _project_to_simplex(h, HALF)
        candidates.append(h)
        
    # 5. Brownian Bridge inspired
    # Generate a Brownian bridge, take cumulative sum, project to simplex
    for _ in range(20):
        # Brownian bridge path
        t = np.linspace(0, 1, N+1)
        W = np.cumsum(np.random.randn(N))
        # Bridge: W - t * W[-1]
        B = W - t[:-1] * W[-1]
        # Use B as deviations from 0.5
        h = 0.5 + 0.2 * B / (np.std(B) + 1e-9)
        h = _project_to_simplex(h, HALF)
        candidates.append(h)
        
    # Refine each candidate
    for h_init in candidates:
        if _time_remaining() < 10:
            break
            
        h_ref, score = _local_refine(h_init, max_iter=100)
        
        if score < best_score:
            best_score = score
            best_h = h_ref.copy()
            
    # If time remains, do more random restarts
    while _time_remaining() > 10:
        # Mix of random and Brownian bridge
        if np.random.random() < 0.5:
            h = np.random.uniform(0, 1, N)
        else:
            t = np.linspace(0, 1, N+1)
            W = np.cumsum(np.random.randn(N))
            B = W - t[:-1] * W[-1]
            h = 0.5 + 0.2 * B / (np.std(B) + 1e-9)
            
        h = _project_to_simplex(h, HALF)
        h_ref, score = _local_refine(h, max_iter=50)
        
        if score < best_score:
            best_score = score
            best_h = h_ref.copy()
            
    return best_h
