"""Combined program: Full-vector optimization with analytical gradients + local search."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import i0

N = 400

# EVOLVE-BLOCK-START

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """
    Smooth approximation of max(|ac|) using log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(ac)
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    log_sum = np.log(sum_exp)
    S = m + log_sum / beta
    w = exp_scaled / sum_exp
    return S, w

def _objective_smooth(h, beta=50.0):
    """
    Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    S, _ = _soft_max_abs(ac, beta)
    return S / (s * s)

def _grad_smooth(h, beta=50.0):
    """
    Analytical gradient of the smooth objective.
    f = S / s^2 where S = soft_max_abs(ac), s = sum(h)
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return np.zeros_like(h)
    
    ac = _autoconv_fft(h)
    S, w = _soft_max_abs(ac, beta)
    
    # Weights for the soft-max
    # v_k = w_k * sign(ac_k)
    v = w * np.sign(ac)
    
    # Gradient of S w.r.t h
    # dS/dh_i = 2 * sum_k v_k h_{k-i}
    # This is 2 * (v cross-correlation h)_i
    # Using FFT: ifft(fft(v) * conj(fft(h)))
    
    fsize = 1
    while fsize < 2 * (2 * N - 1):
        fsize *= 2
        
    V = np.fft.rfft(v, fsize)
    H = np.fft.rfft(h, fsize)
    C = np.fft.irfft(V * np.conj(H), fsize)
    
    # C[i] = sum_n v[n+i] h[n]
    dS_dh = 2 * C[:N]
    
    # f = S / s^2
    # df/dh_i = (dS/dh_i * s^2 - S * 2 * s * 1) / s^4
    # = (dS/dh_i * s - 2 * S) / s^3
    
    grad = (dS_dh * s - 2 * S) / (s ** 3)
    
    return grad

def _local_search_flip(h, deadline, max_passes=10):
    """Local search: try flipping signs of individual elements."""
    h = h.copy()
    best_val = _score(h)
    n = len(h)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    return h, best_val

def _local_search_block(h, deadline, block_size=4, max_passes=5):
    """Local search: try flipping signs of blocks of consecutive elements."""
    h = h.copy()
    best_val = _score(h)
    n = len(h)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for start in range(0, n - block_size + 1, block_size):
            if time.time() > deadline:
                break
            orig = h[start:start+block_size].copy()
            h[start:start+block_size] = -orig
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[start:start+block_size] = orig
        if not improved:
            break
    return h, best_val

def _init_flat_top(n, alpha=14.0):
    """Flat-top-like window (Kaiser with high alpha)."""
    x = np.arange(n)
    mid = (n - 1) / 2.0
    r = np.sqrt(1.0 - ((x - mid) / mid) ** 2)
    h = i0(alpha * r) / i0(alpha)
    s = np.sum(h)
    if abs(s) < 1e-9:
        h += 1e-6
        s = np.sum(h)
    return h / s

def _init_random_smooth(rng, n, smooth_len):
    """Random sequence, smoothed."""
    noise = rng.standard_normal(n)
    kernel = np.ones(smooth_len) / smooth_len
    h = np.convolve(noise, kernel, mode='same')
    s = np.sum(h)
    if abs(s) < 1e-6:
        h = h + 0.1
        s = np.sum(h)
    return h / s

def _init_chirp(n, freq_scale=1.0):
    """Chirp signal."""
    i = np.arange(n)
    h = np.cos(np.pi * freq_scale * i * i / n)
    s = np.sum(h)
    if abs(s) < 1e-6:
        h += 0.1
        s = np.sum(h)
    return h / s

def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # Safety margin
    
    best_score = np.inf
    best_h = None
    
    def update_best(h):
        nonlocal best_score, best_h
        s = np.sum(h)
        if abs(s) < 1e-12:
            return
        val = _score(h)
        if val < best_score:
            best_score = val
            best_h = h.copy()
    
    # Baseline: uniform
    h_uniform = np.ones(N) / N
    update_best(h_uniform)
    
    # Initializations
    rng = np.random.default_rng(42)
    inits = []
    
    # 1. Kaiser windows with different alphas
    for alpha in [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
        inits.append(_init_flat_top(N, alpha))
    
    # 2. Random smooth
    for smooth_len in [5, 10, 20, 50]:
        inits.append(_init_random_smooth(rng, N, smooth_len))
    
    # 3. Chirp variants
    for freq_scale in [0.5, 1.0, 2.0]:
        inits.append(_init_chirp(N, freq_scale))
        
    # 4. Alternating signs on smooth base
    base = _init_flat_top(N, 10.0)
    for shift in [0, 1, 2]:
        h_alt = base.copy()
        h_alt[shift::2] *= -1
        s = np.sum(h_alt)
        if abs(s) > 1e-6:
            inits.append(h_alt / s)

    # Optimize each initialization
    for h_init in inits:
        if time.time() > deadline - 15:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        # Multi-beta gradient optimization
        # We use a sequence of increasing beta to sharpen the max approximation
        for beta in [10, 25, 50, 100]:
            if time.time() > deadline - 10:
                break
            
            try:
                def obj(x, b=beta):
                    return _objective_smooth(x, b)
                
                def grad(x, b=beta):
                    return _grad_smooth(x, b)
                
                res = minimize(obj, h_norm, jac=grad, method='L-BFGS-B',
                               options={'maxiter': 100, 'disp': False, 'gtol': 1e-6, 'maxfun': 200})
                h_opt = res.x
                
                s_opt = np.sum(h_opt)
                if abs(s_opt) < 1e-6:
                    continue
                h_opt = h_opt / s_opt
                update_best(h_opt)
                h_norm = h_opt # Use optimized result for next beta
            except Exception:
                pass
    
    # Local refinement on best
    if best_h is not None and time.time() < deadline - 5:
        h_ref, val_ref = _local_search_flip(best_h, deadline - 2, max_passes=5)
        update_best(h_ref)
        
    # Block local search
    if best_h is not None and time.time() < deadline - 2:
        h_ref, val_ref = _local_search_block(best_h, deadline - 1, block_size=4, max_passes=3)
        update_best(h_ref)
        
    if best_h is None:
        return np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
