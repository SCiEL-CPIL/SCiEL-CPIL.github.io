"""Autoconvolution peak suppression via smooth log-sum-exp surrogate with FFT gradients."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0  # Safety margin before 150s kill

def _eval_value(h):
    """Compute the objective value (lower is better)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(~np.isfinite(h)):
        return np.inf
    sh = float(np.sum(h))
    if abs(sh) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (sh * sh))
    return val

def _autoconv(h):
    """Compute autoconvolution efficiently."""
    return np.convolve(h, h)

def _smooth_objective(h, beta=100.0):
    """
    Smooth surrogate: replace max(conv) with log-sum-exp approximation.
    """
    h = np.asarray(h, dtype=np.float64)
    s = float(np.sum(h))
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv(h)
    max_c = float(np.max(conv))
    shifted = conv - max_c
    max_shifted = float(np.max(shifted))
    if max_shifted > 50:
        shifted = shifted - max_shifted
        max_c_adj = max_c + max_shifted
    else:
        max_c_adj = max_c
    log_sum = max_c_adj + np.log(np.sum(np.exp(beta * shifted))) / beta
    val = abs(2.0 * N * log_sum / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val

def _smooth_grad(h, beta=100.0):
    """
    Gradient of the smooth surrogate, computed efficiently via FFT.
    """
    h = np.asarray(h, dtype=np.float64)
    s = float(np.sum(h))
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = _autoconv(h)
    max_c = float(np.max(conv))
    shifted = conv - max_c
    max_shifted = float(np.max(shifted))
    if max_shifted > 50:
        shifted = shifted - max_shifted
        max_c_adj = max_c + max_shifted
    else:
        max_c_adj = max_c
    
    # Soft-max weights
    exp_vals = np.exp(beta * shifted)
    weight_sum = float(np.sum(exp_vals))
    weights = exp_vals / weight_sum
    
    # lse = max_c_adj + log(weight_sum) / beta
    lse = max_c_adj + np.log(weight_sum) / beta
    
    # Gradient of lse w.r.t. h_m:
    # d lse / d h_m = sum_k weights_k * d c_k / d h_m
    # d c_k / d h_m = 2 * h_{k-m} (if 0 <= k-m < N), else 0
    # So d lse / d h_m = 2 * sum_{j=0}^{N-1} weights[m+j] * h[j] (where m+j < 2N-1)
    
    # This is a cross-correlation. Use FFT for efficiency.
    n_fft = 2 * (2 * N - 1)
    W = np.fft.rfft(weights, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    
    # cross_corr[m] = sum_j weights[m+j] * h[j]
    # This equals the cross-correlation of weights and h.
    # cross_corr = ifft(conj(fft(h)) * fft(weights)) but we need to be careful with indexing.
    # Actually: sum_j weights[m+j] h[j] is the cross-correlation (weights ⊛ h)[m]
    # = ifft(fft(weights) * conj(fft(h))) shifted... 
    # Let's just use direct FFT approach:
    # Define w_full = weights (length 2N-1), h_full = h (length N)
    # cross[m] = sum_j w_full[m+j] * h_full[j] for j where m+j < len(w_full)
    # This is (w_full ⊛ h_full)[m] where ⊛ is cross-correlation
    # In FFT: cross = ifft(conj(FFT(h_full)) * FFT(w_full)) ... need to handle lengths
    
    # Simpler: use scipy or direct
    from scipy.signal import fftconvolve
    # cross[m] = sum_j weights[m+j] h[j]
    # Let's reverse weights: w_rev[k] = weights[2N-2-k]
    # Then sum_j weights[m+j] h[j] = sum_j w_rev[2N-2-(m+j)] h[j]
    # = sum_j w_rev[(2N-2-m) - j] h[j] = (w_rev ⊛ h)[2N-2-m] ... this is getting complex.
    
    # Let's just do it directly since N=400 is small enough
    d_lse_dh = np.zeros(N)
    for m in range(N):
        j_vals = np.arange(N)
        k_vals = m + j_vals
        mask = k_vals < (2 * N - 1)
        d_lse_dh[m] = 2.0 * np.sum(weights[k_vals[mask]] * h[j_vals[mask]])
    
    # f = 2N * lse / s^2
    # df/dh_m = 2N * (d_lse_dh[m] * s^2 - lse * 2 * s) / s^4
    #         = 2N * (d_lse_dh[m] * s - 2 * lse) / s^3
    
    f_inner = 2.0 * N * lse / (s * s)
    df_dh = 2.0 * N * (d_lse_dh * s - 2.0 * lse) / (s * s * s)
    
    # The objective is |f_inner|
    if f_inner >= 0:
        grad = df_dh
    else:
        grad = -df_dh
    
    return grad

def _generate_initializations():
    """Generate diverse initializations."""
    inits = []
    i_arr = np.arange(N)
    t = i_arr / N
    
    # Multi-frequency cosine combinations
    freq_combos = [
        [1, 3, 5, 7],
        [2, 5, 8, 11],
        [1, 4, 7, 10, 13],
        [3, 6, 9, 12],
        [1, 2, 3, 4, 5, 6, 7],
        [2, 4, 6, 8, 10],
        [1, 3, 5, 7, 9, 11],
        [5, 10, 15, 20],
        [1, 5, 10, 15, 25],
        [2, 7, 12, 17],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [3, 7, 11, 15, 19],
    ]
    
    for freqs in freq_combos:
        h = np.zeros(N)
        for f in freqs:
            h += np.cos(2 * np.pi * f * t)
            h += 0.3 * np.sin(2 * np.pi * f * t)
        s = np.sum(h)
        if abs(s) < 1e-10:
            h += 0.05
        inits.append(h.copy())
    
    # Alternating sign versions
    for freqs in freq_combos[:6]:
        h = np.zeros(N)
        for i, f in enumerate(freqs):
            sign = 1 if i % 2 == 0 else -1
            h += sign * np.cos(2 * np.pi * f * t)
        s = np.sum(h)
        if abs(s) < 1e-10:
            h += 0.05
        inits.append(h.copy())
    
    # Random smooth perturbations
    rng = np.random.RandomState(42)
    from scipy.ndimage import gaussian_filter1d
    for _ in range(10):
        h_r = rng.uniform(-1, 1, N)
        h_r = gaussian_filter1d(h_r, sigma=rng.uniform(5, 30))
        s = np.sum(h_r)
        if abs(s) < 1e-10:
            h_r += 0.05
        inits.append(h_r.copy())
    
    # Wigner-like functions
    for a in [0.5, 1.0, 2.0, 3.0]:
        for b in [0.5, 1.0, 2.0, 3.0]:
            h = np.exp(-a * (t - 0.5)**2) * np.cos(2 * np.pi * b * t)
            s = np.sum(h)
            if abs(s) < 1e-10:
                h += 0.05
            inits.append(h.copy())
    
    # Chirp signals
    for alpha in [2, 5, 10]:
        h = np.cos(2 * np.pi * (t + alpha * t**2))
        s = np.sum(h)
        if abs(s) < 1e-10:
            h += 0.05
        inits.append(h.copy())
        inits.append((-h).copy())
    
    return inits

def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # 2.0
    
    def elapsed():
        return time.time() - t0
    
    # --- Phase 1: Generate and evaluate diverse initializations ---
    candidates = _generate_initializations()
    
    cand_vals = []
    for h in candidates:
        v = _eval_value(h)
        if np.isfinite(v):
            cand_vals.append((v, h))
    
    cand_vals.sort(key=lambda x: x[0])
    
    # Keep top 12
    top_candidates = [h for _, h in cand_vals[:12]]
    
    if top_candidates:
        v_top = _eval_value(top_candidates[0])
        if v_top < best_val:
            best_val = v_top
            best_h = top_candidates[0].copy()
    
    # --- Phase 2: Smooth surrogate optimization from top initializations ---
    for h_init in top_candidates[:8]:
        if elapsed() > DEADLINE - 25:
            break
        
        s = np.sum(h_init)
        if abs(s) < 1e-10:
            h_init = h_init + 0.05
        
        for beta in [50.0, 100.0, 200.0]:
            if elapsed() > DEADLINE - 8:
                break
            
            try:
                res = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    h_init,
                    jac=lambda x, b=beta: _smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'maxfun': 600, 'gtol': 1e-10}
                )
                
                if elapsed() > DEADLINE - 5:
                    break
                
                h_candidate = res.x
                v = _eval_value(h_candidate)
                if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                    best_val = v
                    best_h = h_candidate.copy()
                    h_init = best_h.copy()
                    
            except Exception:
                pass
    
    # --- Phase 3: High-precision polish ---
    if elapsed() < DEADLINE - 15:
        h0 = best_h.copy()
        s0 = np.sum(h0)
        if abs(s0) < 1e-10:
            h0 = h0 + 0.01
        
        for beta in [200.0, 400.0, 800.0]:
            if elapsed() > DEADLINE - 5:
                break
            
            try:
                res = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    h0,
                    jac=lambda x, b=beta: _smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 600, 'maxfun': 900, 'gtol': 1e-12}
                )
                
                if elapsed() > DEADLINE - 3:
                    break
                
                h_candidate = res.x
                v = _eval_value(h_candidate)
                if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                    best_val = v
                    best_h = h_candidate.copy()
                    h0 = best_h.copy()
                    
            except Exception:
                pass
    
    # --- Phase 4: Try a few more diverse starts if time permits ---
    if elapsed() < DEADLINE - 20:
        extra_inits = []
        i_arr = np.arange(N)
        t = i_arr / N
        
        # Additional structured patterns
        for freq in [1, 2, 3, 5, 8, 13]:
            h = np.sin(2 * np.pi * freq * t)
            s = np.sum(h)
            if abs(s) < 1e-10:
                h += 0.05
            extra_inits.append(h.copy())
        
        # Piecewise constant with random breakpoints
        rng = np.random.RandomState(123)
        for _ in range(5):
            breaks = sorted(rng.choice(np.arange(1, N), size=rng.randint(5, 15), replace=False))
            h = np.zeros(N)
            for i in range(len(breaks) + 1):
                start = breaks[i-1] if i > 0 else 0
                end = breaks[i] if i < len(breaks) else N
                h[start:end] = rng.uniform(-1, 1)
            s = np.sum(h)
            if abs(s) < 1e-10:
                h += 0.05
            extra_inits.append(h.copy())
        
        for h_init in extra_inits:
            if elapsed() > DEADLINE - 10:
                break
            
            s = np.sum(h_init)
            if abs(s) < 1e-10:
                h_init = h_init + 0.05
            
            v_init = _eval_value(h_init)
            if v_init < best_val + 0.3:
                try:
                    res = minimize(
                        lambda x: _smooth_objective(x, 100.0),
                        h_init,
                        jac=lambda x: _smooth_grad(x, 100.0),
                        method='L-BFGS-B',
                        options={'maxiter': 250, 'maxfun': 400, 'gtol': 1e-10}
                    )
                    
                    if elapsed() > DEADLINE - 5:
                        break
                    
                    h_candidate = res.x
                    v = _eval_value(h_candidate)
                    if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                        best_val = v
                        best_h = h_candidate.copy()
                        
                except Exception:
                    pass
    
    # Final safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
