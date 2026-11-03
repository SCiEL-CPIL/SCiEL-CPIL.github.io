import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# Precompute basis for FFT updates
_freqs = np.arange(_FSIZE // 2 + 1)
_basis = np.exp(-2j * np.pi * _freqs[None, :] * np.arange(N)[:, None] / _FSIZE)

def _autoconv_full(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2*N-1]

def compute_raw_value(x):
    """
    Computes the raw value: max(|autoconvolution|) / (sum(x))^2.
    """
    X = np.fft.rfft(x, _FSIZE)
    conv = np.fft.irfft(X * X, _FSIZE)
    conv = conv[:2*N-1]
    max_abs_conv = np.max(np.abs(conv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs_conv / (integral * integral)

def _soft_max_abs_and_grad(h, beta=50.0):
    """
    Computes the soft-max absolute autoconvolution value and its analytical gradient.
    Returns: value, gradient
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    
    ac = _autoconv_full(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    
    if m < 1e-15:
        S = 0.0
        grad_S = np.zeros_like(h)
    else:
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -50, 50)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        log_sum = np.log(sum_exp)
        S = m + log_sum / beta
        
        w = exp_scaled / sum_exp
        v = w * np.sign(ac)
        
        # Gradient of S w.r.t h: dS/dh_i = 2 * sum_k v_k h_{k-i}
        fsize_corr = 1
        while fsize_corr < 3 * N:
            fsize_corr *= 2
            
        V = np.fft.rfft(v, fsize_corr)
        H = np.fft.rfft(h, fsize_corr)
        C = np.fft.irfft(V * np.conj(H), fsize_corr)
        
        grad_S = 2 * C[:N]
    
    # Objective f = S / s^2
    # df/dh_i = (dS/dh_i * s - 2 * S) / s^3
    grad = (grad_S * s - 2 * S) / (s ** 3)
    
    return S / (s * s), grad

def _local_search_flip(h, max_passes=10, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses precomputed basis for incremental FFT updates.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    integral = np.sum(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            h[i] = -old_val
            delta = -2 * old_val * _basis[i]
            H_new = H + delta
            conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
            new_integral = integral - 2 * old_val
            if abs(new_integral) < 1e-12:
                new_val = float('inf')
            else:
                new_val = np.max(np.abs(conv)) / (new_integral * new_integral)
            
            if new_val < best_val:
                best_val = new_val
                H = H_new
                integral = new_integral
                improved = True
            else:
                h[i] = old_val
        if not improved:
            break
    return h, best_val

def _local_search_peak_targeted(h, deadline=None, max_passes=6):
    """
    Peak-targeted local search: identify lags where |autoconv| is near-max,
    then preferentially test flips at positions that strongly affect those lags.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    integral = np.sum(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Identify peak lags
        conv = np.fft.irfft(H * H, _FSIZE)[:2*N-1]
        abs_conv = np.abs(conv)
        m = np.max(abs_conv)
        if m < 1e-15:
            break
        # Lags within 20% of max
        threshold = 0.8 * m
        peak_lags = np.where(abs_conv >= threshold)[0]
        
        # For each position i, compute how much it contributes to peak lags
        # ac[k] = sum_j h[j] h[k-j], so d(ac[k])/d(h[i]) = 2*sum_j h[j] * [i==j or i==k-j]
        # More precisely: flipping h[i] changes ac[k] by -2*h[i]*h[k-i] (if 0<=k-i<N) plus boundary terms
        # Let's compute a "priority score" for each position
        priority = np.zeros(N)
        for k in peak_lags:
            # ac[k] = sum_{j=max(0,k-N+1)}^{min(N-1,k)} h[j]*h[k-j]
            # d(ac[k])/d(h[i]) = h[k-i] if 0<=k-i<N, else 0 (for the term where j=i)
            # plus h[i] * h[k-i] for the term where k-j=i, i.e., j=k-i
            # Actually ac[k] = sum_j h[j]*h[k-j], so:
            # term with j=i: h[i]*h[k-i] if 0<=k-i<N
            # term with k-j=i, i.e., j=k-i: h[k-i]*h[i] if 0<=k-i<N
            # These are the same term! So d(ac[k])/d(h[i]) = 2*h[k-i] if 0<=k-i<N
            lo = max(0, k - N + 1)
            hi = min(N - 1, k)
            # For position i, it contributes to ac[k] if i is in [lo, hi]
            # The contribution is h[i]*h[k-i]
            # When we flip h[i], the change in ac[k] is -2*h[i]*h[k-i]
            # So the "importance" of position i for lag k is |h[k-i]| if i in [lo,hi]
            for i in range(lo, hi + 1):
                j = k - i
                if 0 <= j < N:
                    priority[i] += abs_conv[k] * abs(h[j])
        
        # Sort positions by priority (highest first)
        order = np.argsort(-priority)
        
        for i in order:
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            h[i] = -old_val
            delta = -2 * old_val * _basis[i]
            H_new = H + delta
            conv_new = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
            new_integral = integral - 2 * old_val
            if abs(new_integral) < 1e-12:
                new_val = float('inf')
            else:
                new_val = np.max(np.abs(conv_new)) / (new_integral * new_integral)
            
            if new_val < best_val:
                best_val = new_val
                H = H_new
                integral = new_integral
                improved = True
            else:
                h[i] = old_val
        if not improved:
            break
    return h, best_val

def _local_search_block(h, deadline=None, block_sizes=None):
    """
    Block flip local search: flip contiguous blocks of signs.
    """
    if block_sizes is None:
        block_sizes = [2, 3, 5, 7, 11]
    
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    integral = np.sum(h)
    
    improved = True
    pass_count = 0
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        pass_count += 1
        step_base = max(1, (N - 15) // (100 - min(pass_count * 10, 60)))
        
        for bs in block_sizes:
            step = max(1, step_base)
            for start in range(0, N - bs + 1, step):
                if deadline is not None and time.time() > deadline:
                    break
                block = h[start:start+bs].copy()
                h[start:start+bs] = -h[start:start+bs]
                delta = np.zeros(_FSIZE//2+1, dtype=complex)
                for j in range(bs):
                    idx = start + j
                    delta += -2 * block[j] * _basis[idx]
                H_new = H + delta
                conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                new_integral = integral - 2 * np.sum(block)
                if abs(new_integral) < 1e-12:
                    new_val = float('inf')
                else:
                    new_val = np.max(np.abs(conv)) / (new_integral * new_integral)
                
                if new_val < best_val:
                    best_val = new_val
                    H = H_new
                    integral = new_integral
                    improved = True
                else:
                    h[start:start+bs] = block
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    # 1. Baseline: All ones
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
    
    # Barker-like structured
    x_barker = np.array([1 if i % 4 in [0, 2] else -1 for i in range(N)], dtype=float)
    candidates.append(x_barker)
    
    # Walsh-Hadamard-like patterns (using bit-reversal or simple XOR patterns)
    for period in [2, 4, 8, 16]:
        x_wh = np.array([1 if (i // period) % 2 == 0 else -1 for i in range(N)], dtype=float)
        candidates.append(x_wh)
    
    # Random sequences
    rng = np.random.default_rng(42)
    for _ in range(20):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)
    
    # Golden ratio based
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
    
    # 4. Smooth surrogate optimization from best candidates
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
        
        for beta in [20, 50, 100]:
            if time.time() > deadline - 20:
                break
            
            try:
                def obj_and_grad(x, b=beta):
                    return _soft_max_abs_and_grad(x, beta=b)
                
                res = minimize(obj_and_grad, h_norm, method='L-BFGS-B',
                               jac=True,
                               options={'maxiter': 300, 'disp': False, 'gtol': 1e-8})
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
    if time.time() < deadline - 20:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        
        # Peak-targeted local search first
        h_work, val_work = _local_search_peak_targeted(h_work, deadline=deadline - 15)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
        
        # Single-bit flip local search
        if time.time() < deadline - 10:
            h_work, val_work = _local_search_flip(h_work, max_passes=6, deadline=deadline - 8)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
        
        # Block flip local search
        if time.time() < deadline - 5:
            h_work, val_work = _local_search_block(h_work, deadline=deadline - 3,
                                                    block_sizes=[2, 3, 5, 7, 11])
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
    
    # 6. Random restarts with local search if time remains
    if time.time() < deadline - 30:
        for _ in range(10):
            if time.time() > deadline - 15:
                break
            x_rand = rng.choice([1.0, -1.0], N)
            h_work, val_work = _local_search_peak_targeted(x_rand, deadline=deadline - 10)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
            # Also try standard flip search on the result
            if time.time() < deadline - 12:
                h_work2, val_work2 = _local_search_flip(h_work, max_passes=3, deadline=deadline - 10)
                if val_work2 < best_val:
                    best_val = val_work2
                    best_x = h_work2.copy()
    
    # 7. Final high-precision polish: multiple passes
    if time.time() < deadline - 3:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        for polish_pass in range(4):
            if time.time() > deadline - 1:
                break
            h_work, val_work = _local_search_flip(h_work, max_passes=2, deadline=deadline - 1)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
            # Also try peak-targeted polish
            if time.time() < deadline - 1:
                h_work, val_work = _local_search_peak_targeted(h_work, deadline=deadline - 1, max_passes=2)
                if val_work < best_val:
                    best_val = val_work
                    best_x = h_work.copy()
    
    return best_x

if __name__ == "__main__":
    x = construct()
    val = compute_raw_value(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
