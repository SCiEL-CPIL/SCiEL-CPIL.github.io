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
    Uses vectorized FFT delta computation for speed.
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
            # Finer step for smaller blocks, coarser for larger
            if bs <= 4:
                step = max(1, (N - bs) // 150)
            elif bs <= 7:
                step = max(1, (N - bs) // 100)
            else:
                step = max(1, (N - bs) // 60)
            
            for start in range(0, N - bs + 1, step):
                if deadline is not None and time.time() > deadline:
                    break
                block = h[start:start+bs].copy()
                h[start:start+bs] = -h[start:start+bs]
                new_sum = current_sum - 2 * np.sum(block)
                
                # Vectorized FFT delta update
                idxs = np.arange(start, start + bs)
                delta = -2.0 * np.sum(block * _EXP_TABLE[idxs])
                # Need to add this to each frequency bin appropriately
                # Actually, the delta for each frequency k is sum_j (-2*block[j]*exp(-2pij*idx[j]/FSIZE))
                # This is -2 * sum_j block[j] * exp(-2pij*idx[j]/FSIZE) for each k
                # But _EXP_TABLE[idx] is exp(-2pij*idx/FSIZE), so we need:
                # delta[k] = -2 * sum_j block[j] * exp(-2pij*(start+j)/FSIZE)
                # = -2 * exp(-2pij*start/FSIZE) * sum_j block[j] * exp(-2pij*j/FSIZE)
                # = -2 * _EXP_TABLE[start] * sum_j block[j] * _EXP_TABLE[j]
                
                inner_sum = np.sum(block * _EXP_TABLE[:bs])
                delta_vec = -2.0 * _EXP_TABLE[start] * inner_sum
                H_new = H + delta_vec
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
            
    # 4. Smooth surrogate optimization with adaptive beta schedule
    scored_candidates = []
    for x in candidates:
        val = compute_raw_value(x)
        scored_candidates.append((val, x))
    scored_candidates.sort(key=lambda t: t[0])
    
    top_starts = scored_candidates[:10]
    
    for init_val, h_init in top_starts:
        if time.time() > deadline - 45:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        h_norm = h_init / s0
        
        # Adaptive beta schedule: start coarse, refine progressively
        for beta in [10, 25, 50, 100, 200, 400]:
            if time.time() > deadline - 20:
                break
            
            try:
                def obj(x, b=beta):
                    return _smooth_obj(x, beta=b)
                
                # More iterations for higher beta (finer optimization)
                max_iter = 300 if beta <= 50 else 800
                
                res = minimize(obj, h_norm, method='L-BFGS-B',
                               options={'maxiter': max_iter, 'disp': False, 'gtol': 1e-10,
                                        'ftol': 1e-12, 'maxls': 50})
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
                
                # Use the binary result as next starting point for higher beta
                h_norm = h_bin / np.sum(h_bin)
                
            except Exception:
                pass
    
    # 5. Peak-suppression phase: directly target argmax of autoconvolution
    if time.time() < deadline - 25:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        H = np.fft.rfft(h_work, _FSIZE)
        current_sum = np.sum(h_work)
        best_val_work = best_val
        
        # Multiple rounds of peak targeting
        for round_num in range(5):
            if time.time() > deadline - 10:
                break
            
            ac = np.fft.irfft(H * H, _FSIZE)[:2*N-1]
            abs_ac = np.abs(ac)
            
            # Find top peaks and try to suppress them
            peak_indices = np.argsort(abs_ac)[::-1][:20]
            
            improved_this_round = False
            for peak_idx in peak_indices:
                if time.time() > deadline - 8:
                    break
                
                # Try flipping signs near the peak location to reduce it
                # Peak at lag k corresponds to correlation between x[i] and x[i+k]
                # Flipping x[j] affects all lags involving j
                center = peak_idx % N
                
                for flip_offset in range(-3, 4):
                    if time.time() > deadline - 8:
                        break
                    
                    j = (center + flip_offset) % N
                    old_val = h_work[j]
                    h_work[j] = -old_val
                    new_sum = current_sum - 2 * old_val
                    
                    delta = -2 * old_val * _EXP_TABLE[j]
                    H_new = H + delta
                    conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                    
                    if abs(new_sum) < 1e-12:
                        new_val = float('inf')
                    else:
                        new_val = np.max(np.abs(conv)) / (new_sum * new_sum)
                    
                    if new_val < best_val_work:
                        best_val_work = new_val
                        H = H_new
                        current_sum = new_sum
                        improved_this_round = True
                    else:
                        h_work[j] = old_val
                
                if improved_this_round:
                    # Re-evaluate peaks after improvement
                    break
            
            if not improved_this_round:
                break
        
        if best_val_work < best_val:
            best_val = best_val_work
            best_x = h_work.copy()
    
    # 6. Local search refinement
    if time.time() < deadline - 15:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        
        h_work, val_work = _local_search_flip(h_work, max_passes=10, deadline=deadline - 8)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
            
        if time.time() < deadline - 5:
            h_work, val_work = _local_search_block(h_work, deadline=deadline - 2, 
                                                    block_sizes=[2, 3, 4, 5, 7, 9, 13])
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
    
    # 7. Final random restarts with structured perturbations
    if time.time() < deadline - 20:
        # Mix of pure random and structured perturbations
        for _ in range(12):
            if time.time() > deadline - 10:
                break
            
            perturbation_type = rng.integers(0, 3)
            if perturbation_type == 0:
                # Pure random
                x_rand = rng.choice([1.0, -1.0], N)
            elif perturbation_type == 1:
                # Flip a small random subset of current best
                x_rand = best_x.copy()
                n_flips = rng.integers(5, 25)
                flip_idx = rng.choice(N, n_flips, replace=False)
                x_rand[flip_idx] = -x_rand[flip_idx]
            else:
                # Flip a contiguous block
                x_rand = best_x.copy()
                bs = rng.integers(3, 15)
                start = rng.integers(0, N - bs)
                x_rand[start:start+bs] = -x_rand[start:start+bs]
            
            h_work, val_work = _local_search_flip(x_rand, max_passes=5, deadline=deadline - 5)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
                
    # 8. Final high-precision polish
    if time.time() < deadline - 3:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        
        # Multiple passes of flip search for fine-tuning
        for _ in range(3):
            if time.time() > deadline - 1:
                break
            h_work, val_work = _local_search_flip(h_work, max_passes=5, deadline=deadline - 1)
            if val_work >= best_val:
                break
            best_val = val_work
            best_x = h_work.copy()
        
        # Final block polish
        if time.time() < deadline - 2:
            h_work, val_work = _local_search_block(h_work, deadline=deadline - 1,
                                                    block_sizes=[2, 3, 4, 5, 7])
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
            
    return best_x

if __name__ == "__main__":
    x = construct()
    val = compute_raw_value(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
