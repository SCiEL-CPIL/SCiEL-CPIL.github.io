"""Erdos minimum overlap - find step heights minimizing max overlap with shifted complement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
WIDTH = 2.0 / N

def compute_max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h) over all shifts (circular)."""
    N = len(h)
    width = 2.0 / N
    max_overlap = 0.0
    for s in range(1, N):
        h_shifted = np.roll(h, -s)
        overlap = np.sum(h * (1.0 - h_shifted)) * width
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _overlaps_fast(h):
    """Return array of overlaps for shifts 1..N-1 using FFT-based correlation."""
    N = len(h)
    width = 2.0 / N
    # overlap[s] = sum_i h[i] * (1 - h[(i+s)%N]) * width
    #            = (sum_i h[i]) - sum_i h[i]*h[(i+s)%N]
    #            = N/2 - sum_i h[i]*h[(i+s)%N]
    h_ext = np.zeros(2 * N)
    h_ext[:N] = h
    H = np.fft.fft(h, n=2*N)
    corr = np.fft.ifft(H * np.conj(H)).real[:N]
    # corr[s] = sum_i h[i]*h[(i-s) mod N]
    # We need sum_i h[i]*h[(i+s) mod N] = corr[(N-s) mod N]
    overlaps = np.empty(N - 1)
    for s in range(1, N):
        idx = (N - s) % N
        overlaps[s-1] = (N/2.0 - corr[idx]) * width
    return overlaps

def _soft_max(x, tau):
    """Smooth max via log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    m = np.max(x)
    if m == -np.inf:
        return m
    return m + tau * np.log(np.sum(np.exp((x - m) / tau)))

def _build_h_from_freq(coeffs, N_steps):
    """Build h from frequency coefficients: h[i] = 0.5 + sum_k c_k * cos(2*pi*k*i/N)."""
    i = np.arange(N_steps)
    h = np.full(N_steps, 0.5)
    for k, c in enumerate(coeffs, start=1):
        h += c * np.cos(2.0 * np.pi * k * i / N_steps)
    return h

def _project(h, N_steps):
    """Project h to [0,1] with sum exactly N/2. Returns clipped array."""
    h = np.clip(h, 0.0, 1.0)
    target = N_steps / 2.0
    diff = target - np.sum(h)
    if abs(diff) < 1e-15:
        return h
    if diff > 0:
        room = 1.0 - h
        total_room = np.sum(room)
        if total_room < 1e-15:
            return h
        h = h + diff * room / total_room
    else:
        room = h
        total_room = np.sum(room)
        if total_room < 1e-15:
            return h
        h = h + diff * room / total_room
    return np.clip(h, 0.0, 1.0)

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    best_h = np.full(N, 0.5)
    best_score = compute_max_overlap(best_h)
    
    # Try structured initial patterns
    initial_patterns = []
    if N % 4 == 0:
        p = np.tile(np.array([1.0, 1.0, 0.0, 0.0]), N // 4)
        initial_patterns.append(p)
    if N % 2 == 0:
        p = np.tile(np.array([1.0, 0.0]), N // 2)
        initial_patterns.append(p)
    p = np.zeros(N)
    p[:N//2] = 1.0
    initial_patterns.append(p)
    if N % 3 == 0:
        p = np.tile(np.array([1.0, 1.0, 0.0]), N // 3)
        initial_patterns.append(p)
    # 5-block: 2 ones, 3 zeros
    if N % 5 == 0:
        p = np.tile(np.array([1.0, 1.0, 0.0, 0.0, 0.0]), N // 5)
        initial_patterns.append(p)
    
    for p in initial_patterns:
        if abs(np.sum(p) - N/2) < 1e-10:
            s = compute_max_overlap(p)
            if s < best_score:
                best_score = s
                best_h = p.copy()
    
    # Frequency-based optimization using scipy
    K_max = 12
    
    for K in range(1, K_max + 1):
        if time.time() > deadline - 25:
            break
        
        for trial in range(5):
            if time.time() > deadline - 18:
                break
            
            rng = np.random.RandomState(42 + K * 10 + trial)
            
            if trial == 0:
                coeffs = np.zeros(K)
                coeffs[0] = 0.3
            elif trial == 1:
                coeffs = np.zeros(K)
                coeffs[0] = 0.2
                if K > 1:
                    coeffs[1] = 0.15
            elif trial == 2:
                coeffs = rng.randn(K) * 0.1
            elif trial == 3:
                coeffs = np.zeros(K)
                coeffs[0] = 0.25
                if K > 1:
                    coeffs[1] = -0.1
                if K > 2:
                    coeffs[2] = 0.05
            else:
                coeffs = np.zeros(K)
                coeffs[0] = 0.35
                if K > 1:
                    coeffs[1] = 0.1
            
            h = _build_h_from_freq(coeffs, N)
            h = _project(h, N)
            
            def obj(c):
                hh = _build_h_from_freq(c, N)
                hh = _project(hh, N)
                ov = _overlaps_fast(hh)
                return _soft_max(ov, 0.01)
            
            try:
                res = minimize(obj, coeffs, method='L-BFGS-B', 
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                h_opt = _build_h_from_freq(res.x, N)
                h_opt = _project(h_opt, N)
                score = compute_max_overlap(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
            except Exception:
                pass
        
        # Quick coordinate descent polish
        if time.time() < deadline - 12:
            h_cd = best_h.copy()
            for cd_iter in range(100):
                if time.time() > deadline - 8:
                    break
                improved = False
                for i in range(0, N-1, 2):
                    j = i + 1
                    h_old_i = h_cd[i]
                    h_old_j = h_cd[j]
                    for delta in [0.1, 0.2, 0.3, 0.5]:
                        h_new_i = np.clip(h_old_i + delta, 0, 1)
                        h_new_j = np.clip(h_old_j - delta, 0, 1)
                        h_cd[i] = h_new_i
                        h_cd[j] = h_new_j
                        score = compute_max_overlap(h_cd)
                        if score < best_score - 1e-12:
                            best_score = score
                            best_h = h_cd.copy()
                            improved = True
                        h_cd[i] = h_old_i
                        h_cd[j] = h_old_j
                    for delta in [0.1, 0.2, 0.3, 0.5]:
                        h_new_i = np.clip(h_old_i - delta, 0, 1)
                        h_new_j = np.clip(h_old_j + delta, 0, 1)
                        h_cd[i] = h_new_i
                        h_cd[j] = h_new_j
                        score = compute_max_overlap(h_cd)
                        if score < best_score - 1e-12:
                            best_score = score
                            best_h = h_cd.copy()
                            improved = True
                        h_cd[i] = h_old_i
                        h_cd[j] = h_old_j
                if not improved:
                    break
    
    # Ensure sum is exactly N/2
    if abs(np.sum(best_h) - N/2) > 1e-10:
        diff = N/2 - np.sum(best_h)
        if diff > 0:
            room = 1.0 - best_h
            total_room = np.sum(room)
            if total_room > 1e-15:
                best_h += diff * room / total_room
        else:
            room = best_h
            total_room = np.sum(room)
            if total_room > 1e-15:
                best_h += diff * room / total_room
        best_h = np.clip(best_h, 0.0, 1.0)
    
    if abs(np.sum(best_h) - N/2) > 1e-8:
        best_h = np.full(N, 0.5)
    
    return best_h
