import numpy as np
from scipy.optimize import minimize
import time

N = 400

def _autoconv(h):
    """Compute full autoconvolution of h using FFT."""
    n = len(h)
    # Use linear convolution size to avoid circular wrap-around
    # Size 2*n is sufficient for full convolution of two length n vectors
    f = np.fft.rfft(h, n=2 * n)
    c = np.fft.irfft(f * f, n=2 * n)
    return c

def _soft_max_conv(h, beta):
    """
    Compute soft-max of autoconvolution and its gradient.
    Returns (value, gradient).
    """
    n = len(h)
    conv = _autoconv(h)
    
    # Stable log-sum-exp calculation for soft-max
    m = np.max(conv)
    shifted = beta * (conv - m)
    # Clip to prevent overflow/underflow issues in exp, though beta*diff is usually small
    # If beta is large, only the max matters.
    ez = np.exp(shifted)
    z = ez.sum()
    sm = m + np.log(z) / beta
    w = ez / z
    
    # Gradient calculation:
    # grad_i = sum_k w[k] * (h[k-i] + h[i-k])
    # This is the cross-correlation of w and h, plus the reverse cross-correlation.
    # Let c[d] = sum_k w[k] * h[k-d]
    # Then grad_i = c[i] + c[-i]
    
    # Compute cross-correlation using FFT
    # We need c[d] for d in [-(n-1), n-1] effectively, but w has support [0, 2n-2].
    # Actually, w is defined on the support of conv, which is [0, 2n-2].
    # h is defined on [0, n-1].
    # c[d] = sum_{k=0}^{2n-2} w[k] h[k-d]
    # This is a cross-correlation.
    # Using FFT:
    # ifft(fft(w, size) * conj(fft(h, size))) gives circular cross-correlation.
    # We need linear cross-correlation.
    # Size must be at least (2n-1) + n - 1 = 3n - 2. Let's use 4n for safety and power of 2 efficiency.
    size = 4 * n
    Wf = np.fft.rfft(w, n=size)
    Hf = np.fft.rfft(h, n=size)
    
    # Cross-correlation r_w_h[d] = sum_k w[k] h[k-d]
    # In frequency domain: ifft(Wf * conj(Hf))
    # Note: numpy fft uses e^{-2pi i k n / N}.
    # Standard cross-correlation x corr y is ifft(fft(x) * conj(fft(y))).
    # Let's verify indices.
    # cc[p] = sum_k w[k] h[(k-p) mod size]
    # For linear correlation, if 0 <= k-p < n, h[k-p] is the value. Otherwise 0.
    # We want c[d] = sum_k w[k] h[k-d].
    # So c[d] = cc[d] for d >= 0.
    # And c[d] = cc[size+d] for d < 0.
    
    cc = np.fft.irfft(Wf * np.conj(Hf), n=size)
    
    grad = np.zeros(n)
    # We need c[i] and c[-i] for i in [0, n-1]
    # c[i] is at index i in cc
    # c[-i] is at index size - i in cc (for i>0), and size (which is 0) for i=0?
    # Wait, c[0] = sum w[k]h[k]. This is cc[0].
    # c[-i] = sum w[k]h[k+i].
    # cc[size-i] = sum_k w[k] h[k - (size-i)] = sum_k w[k] h[k+i - size].
    # Since h is zero outside [0, n-1], and k is in [0, 2n-2],
    # h[k+i-size] is non-zero only if 0 <= k+i-size < n.
    # This corresponds to the negative lag part.
    
    # Let's just extract the needed values.
    # c[i] for i in 0..n-1 is cc[0..n-1]
    # c[-i] for i in 0..n-1:
    # i=0: c[0] = cc[0]
    # i>0: c[-i] = cc[size-i]
    
    c_pos = cc[0:n]
    c_neg = cc[size-n:size] # This gives cc[size-(n-1)] ... cc[size-1]
    # Wait, c[-i] for i=1 is cc[size-1]. For i=n-1 is cc[size-(n-1)].
    # So c_neg[i-1] = c[-i].
    # We need grad[i] = c[i] + c[-i].
    # grad[0] = c[0] + c[0] = 2 * cc[0]
    # grad[i] = cc[i] + cc[size-i] for i > 0.
    
    grad[0] = 2.0 * cc[0]
    if n > 1:
        grad[1:] = cc[1:n] + cc[size-1:size-n-1:-1]
        
    return sm, grad

def _obj_and_grad(h, beta):
    """
    Objective: abs(2 * N * max(conv) / sum(h)^2)
    We optimize the soft-max surrogate.
    Note: The absolute value in the final objective makes it non-differentiable if max < 0.
    However, for the seed (ones), max > 0. We assume max stays positive or we optimize 
    the signed max and hope it stays positive, or we use a soft-max that pushes towards 
    minimizing the positive peak.
    Actually, the task says "maximum is taken over its signed entries".
    If the max is negative, the value is abs(2N * neg / pos) = 2N * |neg| / pos.
    Minimizing the signed max is a good proxy if the max is positive.
    If the max becomes negative, minimizing it (making it more negative) increases the absolute value.
    So we want to minimize the *positive* part of the max, or keep the max small.
    Using soft-max of the signed convolution is a standard approach.
    """
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12, np.zeros_like(h)
    
    sm, g_sm = _soft_max_conv(h, beta)
    
    # f(h) = 2 * N * sm / s^2
    # df/dh_i = 2N * ( g_sm_i * s^2 - sm * 2s ) / s^4
    #        = 2N * ( g_sm_i * s - 2 sm ) / s^3
    g = 2.0 * N * (g_sm * s - 2.0 * sm) / (s * s * s)
    
    return 2.0 * N * sm / (s * s), g

def _hard(h):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _param_to_h(p):
    """
    Map parameters to h.
    p = [a0, (a1, b1, f1), (a2, b2, f2), ...]
    DC + 8 sinusoidal modes.
    """
    i = np.arange(N)
    h = np.zeros(N)
    a0 = p[0]
    h += a0
    
    num_modes = 8
    for idx in range(num_modes):
        base = 1 + idx * 3
        a, b, f = p[base], p[base+1], p[base+2]
        # Ensure frequency is positive and reasonable
        # f is free, but cos/sin are periodic.
        h += a * np.cos(2 * np.pi * f * i / N) + b * np.sin(2 * np.pi * f * i / N)
        
    return h

def _param_obj(p):
    h = _param_to_h(p)
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    t0 = time.time()
    DEADLINE = 135.0 # Safety margin
    
    def time_left():
        return DEADLINE - (time.time() - t0)

    rng = np.random.default_rng(42)
    
    best_h = np.ones(N, dtype=float)
    best_val = _hard(best_h)
    
    # --- Phase 1: Parametric Optimization ---
    # Generate diverse initial candidates
    candidates = []
    num_modes = 8
    
    # 1. Low frequency dominant
    p0 = np.zeros(1 + 3*num_modes)
    p0[0] = 1.0
    for i in range(num_modes):
        base = 1 + i*3
        p0[base] = 0.1
        p0[base+1] = 0.0
        p0[base+2] = i+1.0
    candidates.append(p0)
    
    # 2. Random frequency mixes
    for _ in range(40):
        p = np.zeros(1 + 3*num_modes)
        p[0] = rng.uniform(-0.5, 1.5)
        for i in range(num_modes):
            base = 1 + i*3
            p[base] = rng.uniform(-1.0, 1.0)
            p[base+1] = rng.uniform(-1.0, 1.0)
            # Frequencies spread out
            p[base+2] = rng.uniform(0.5, 20.0)
        candidates.append(p)
    
    # 3. Specific "Barker-like" or structured patterns
    # Try to mimic low autocorrelation sequences
    for _ in range(20):
        p = np.zeros(1 + 3*num_modes)
        p[0] = 1.0
        for i in range(num_modes):
            base = 1 + i*3
            # Alternating signs
            p[base] = (-1)**i * 0.5
            p[base+1] = 0.0
            p[base+2] = (i+1)*2.0
        candidates.append(p)

    # Evaluate and sort candidates
    scored = []
    for p in candidates:
        if time_left() < 40.0:
            break
        v = _param_obj(p)
        scored.append((v, p))
    scored.sort(key=lambda x: x[0])
    
    # Optimize top candidates
    top_k = 10
    for i, (v, p0) in enumerate(scored[:top_k]):
        if time_left() < 20.0:
            break
        try:
            res = minimize(_param_obj, p0, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000})
            h = _param_to_h(res.x)
            v_hard = _hard(h)
            if v_hard < best_val:
                best_val = v_hard
                best_h = h.copy()
        except Exception:
            pass

    # --- Phase 2: Full 400-dim Polish ---
    # Normalize sum to 1
    s = best_h.sum()
    if abs(s) > 1e-9:
        h_curr = best_h / s
    else:
        h_curr = np.ones(N) / N
        
    # Use soft-max optimization with increasing beta
    beta_seq = [10.0, 20.0, 50.0, 100.0]
    
    for beta in beta_seq:
        if time_left() < 15.0:
            break
        try:
            def obj(x):
                val, _ = _obj_and_grad(x, beta)
                return val
            def gr(x):
                _, g = _obj_and_grad(x, beta)
                return g
                
            res = minimize(obj, h_curr, jac=gr, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 8000})
            h_new = res.x
            v = _hard(h_new)
            if v < best_val:
                best_val = v
                best_h = h_new.copy()
                h_curr = h_new.copy()
        except Exception:
            pass

    # --- Phase 3: Coordinate Descent on Hard Objective ---
    # This helps refine the "hard" max, which soft-max approximates.
    h_curr = best_h.copy()
    best_local = _hard(h_curr)
    
    # Precompute indices for speed
    i_range = np.arange(N)
    
    iters = 0
    while time_left() > 5.0 and iters < 100:
        iters += 1
        conv = _autoconv(h_curr)
        kmax = int(np.argmax(conv))
        lag = kmax - (N - 1)
        
        # Subgradient of max(conv) at lag
        # d c[lag] / d h_i = h[lag-i] + h[i-lag]
        sg = np.zeros(N)
        
        # Term 1: h[lag-i]
        idx1 = lag - i_range
        mask1 = (idx1 >= 0) & (idx1 < N)
        sg[mask1] += h_curr[idx1[mask1]]
        
        # Term 2: h[i-lag]
        idx2 = i_range - lag
        mask2 = (idx2 >= 0) & (idx2 < N)
        sg[mask2] += h_curr[idx2[mask2]]
        
        ss = h_curr.sum()
        if abs(ss) < 1e-9:
            break
            
        # We want to reduce 2N * maxc / ss^2.
        # Step in direction -sg to reduce maxc.
        # Center sg to keep sum constant (approx)
        sg_centered = sg - np.mean(sg)
        
        improved = False
        # Try multiple step sizes
        for alpha in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
            # Reduce maxc: h - alpha * sg
            # But we also need to consider the denominator ss^2.
            # If we keep sum constant, we just minimize maxc.
            cand = h_curr - alpha * sg_centered
            vv = _hard(cand)
            if vv < best_local:
                best_local = vv
                h_curr = cand
                if vv < best_val:
                    best_val = vv
                    best_h = cand.copy()
                improved = True
                break
            
            # Sometimes increasing maxc (if negative) or adjusting sum helps?
            # Stick to minimizing the positive peak for now.
            
        if not improved:
            # Try smaller steps or break
            if alpha < 2e-4:
                break

    # Final sanity check
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    if abs(best_h.sum()) < 1e-10:
        best_h = np.ones(N)
        
    return list(best_h)
