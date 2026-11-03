import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# Precompute frequency indices for incremental FFT updates
_FREQ_IDX = np.arange(_FSIZE // 2 + 1)
_EXP_TABLE = np.exp(-2j * np.pi * _FREQ_IDX / _FSIZE)

def _autoconv_full(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2*N-1]

def _max_abs_ac(h):
    return np.max(np.abs(_autoconv_full(h)))

def compute_raw_value(x):
    """
    Computes the raw value: max(|autoconvolution|) / (sum(x))^2.
    Assumes x is a 1D numpy array.
    """
    X = np.fft.rfft(x, _FSIZE)
    conv = np.fft.irfft(X * X, _FSIZE)
    conv = conv[:2*N-1]
    max_abs_conv = np.max(np.abs(conv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs_conv / (integral * integral)

def _soft_max_abs(ac, beta=50.0):
    """Smooth approximation of max(|ac|) using log-sum-exp."""
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _smooth_obj(h, beta=50.0):
    """Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_full(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _local_search_flip(h, max_passes=10, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses incremental FFT and Sum updates for O(1) sum update and O(N) FFT update.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    current_sum = np.sum(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            h[i] = -old_val
            new_sum = current_sum - 2 * old_val
            
            # Update FFT incrementally
            delta = -2 * old_val * _EXP_TABLE[i]
            H_new = H + delta
            conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
            
            if abs(new_sum) < 1e-12:
                new_val = float('inf')
            else:
                new_val = np.max(np.abs(conv)) / (new_sum * new_sum)
            
            if new_val < best_val:
                best_val = new_val
                H = H_new
                current_sum = new_sum
                improved = True
            else:
                h[i] = old_val  # revert
        if not improved:
            break
    return h, best_val

def _local_search_block(h, deadline=None, block_sizes=[3,5,7,10]):
    """
    Block flip local search: flip contiguous blocks of signs.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    current_sum = np.sum(h)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for bs in block_sizes:
            step = max(1, (N - bs) // 80)
            for start in range(0, N - bs + 1, step):
                if deadline is not None and time.time() > deadline:
                    break
                block = h[start:start+bs].copy()
                h[start:start+bs] = -h[start:start+bs]
                new_sum = current_sum - 2 * np.sum(block)
                
                # Update FFT
                delta = np.zeros(_FSIZE//2+1, dtype=complex)
                for j in range(bs):
                    idx = start + j
                    delta += -2 * block[j] * _EXP_TABLE[idx]
                H_new = H + delta
                conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                
                if abs(new_sum) < 1e-12:
                    new_val = float('inf')
                else:
                    new_val = np.max(np.abs(conv)) / (new_sum * new_sum)
                
                if new_val < best_val:
                    best_val = new_val
                    H = H_new
                    current_sum = new_sum
                    improved = True
                else:
                    h[start:start+bs] = block  # revert
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    # 1. Baseline: All ones (Theoretical Lower Bound Candidate)
    best_x = np.ones(N, dtype=float)
    best_val = compute_raw_value(best_x)
    
    # 2. Generate diverse candidates
    candidates = [best_x.copy()]
    
    t = np.arange(N)
    # Chirp sequences
    for freq in [1.0, 2.0, 0.5, 3.0]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1
        candidates.append(x_chirp)
        
    # Linear chirp
    for freq in [1.0, 2.0, 0.5]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1
        candidates.append(x_lin)
        
    # Barker-like
    x_barker = np.array([1 if i % 4 in [0, 2] else -1 for i in range(N)], dtype=float)
    candidates.append(x_barker)
    
    # Random sequences
    rng = np.random.default_rng(42)
    for _ in range(15):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)
        
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    x_gold = np.sign(np.cos(2 * np.pi * phi * t))
    x_gold[x_gold == 0] = 1
    candidates.append(x_gold)
    
    # 3. Evaluate all candidates
    for x in candidates:
        val = compute_raw_value(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
    # 4. Smooth surrogate optimization
    scored_candidates = []
    for x in candidates:
        val = compute_raw_value(x)
        scored_candidates.append((val, x))
    scored_candidates.sort(key=lambda t: t[0])
    
    top_starts = scored_candidates[:8]
    
    for init_val, h_init in top_starts:
        if time.time() > deadline - 40:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        h_norm = h_init / s0
        
        for beta in [15, 30, 60, 120]:
            if time.time() > deadline - 20:
                break
            
            try:
                def obj(x, b=beta):
                    return _smooth_obj(x, beta=b)
                
                res = minimize(obj, h_norm, method='L-BFGS-B',
                               options={'maxiter': 500, 'disp': False, 'gtol': 1e-8})
                h_opt = res.x
                
                val = compute_raw_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_x = h_opt.copy()
                    
                h_bin = np.sign(h_opt)
                h_bin[h_bin == 0] = 1
                val_bin = compute_raw_value(h_bin)
                if val_bin < best_val:
                    best_val = val_bin
                    best_x = h_bin.copy()
                    
            except Exception:
                pass
    
    # 5. Local search refinement
    if time.time() < deadline - 15:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        
        h_work, val_work = _local_search_flip(h_work, max_passes=8, deadline=deadline - 8)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
            
        if time.time() < deadline - 5:
            h_work, val_work = _local_search_block(h_work, deadline=deadline - 2, 
                                                    block_sizes=[3, 5, 7, 11])
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
    
    # 6. Final random restarts
    if time.time() < deadline - 25:
        for _ in range(8):
            if time.time() > deadline - 12:
                break
            x_rand = rng.choice([1.0, -1.0], N)
            h_work, val_work = _local_search_flip(x_rand, max_passes=3, deadline=deadline - 6)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
                
    # 7. Final polish
    if time.time() < deadline - 3:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        h_work, val_work = _local_search_flip(h_work, max_passes=3, deadline=deadline - 1)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
            
    return best_x

if __name__ == "__main__":
    x = construct()
    val = compute_raw_value(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
