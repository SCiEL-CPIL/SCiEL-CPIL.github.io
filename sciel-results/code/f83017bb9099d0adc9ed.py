"""Combined program: Full-vector optimization with analytical gradients + Gumbel-Softmax sign relaxation + local search."""
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

def _gumbel_softmax_sample(logits, temperature, rng):
    """
    Gumbel-Softmax relaxation for binary sign selection.
    logits: shape (n,) - log-odds for each position being +1
    temperature: controls softness of the relaxation
    Returns: shape (n,) in [0, 1], to be mapped to signs via 2*p - 1
    """
    gumbels = -np.log(-np.log(rng.uniform(1e-10, 1.0 - 1e-10, size=logits.shape)))
    y = logits / temperature + gumbels
    # Softmax
    y_max = np.max(y)
    exp_y = np.exp(y - y_max)
    p = exp_y / np.sum(exp_y)
    return p

def _gumbel_sign_objective(logits, temperature, beta, rng):
    """
    Objective for Gumbel-Softmax sign optimization.
    logits: shape (n,) - log-odds for each position being +1
    Returns: (objective_value, gradient_wrt_logits)
    """
    # Sample relaxed signs
    p = _gumbel_softmax_sample(logits, temperature, rng)
    # Map to signed values: h_i = 2*p_i - 1, then normalize
    h = 2.0 * p - 1.0
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(logits)
    h_norm = h / s
    
    # Compute smooth objective and gradient w.r.t h
    ac = _autoconv_fft(h_norm)
    S, w = _soft_max_abs(ac, beta)
    obj_val = S / (s * s)
    
    # Gradient of S w.r.t h
    v = w * np.sign(ac)
    fsize = 1
    while fsize < 2 * (2 * N - 1):
        fsize *= 2
    V = np.fft.rfft(v, fsize)
    H = np.fft.rfft(h_norm, fsize)
    C = np.fft.irfft(V * np.conj(H), fsize)
    dS_dh = 2 * C[:N]
    
    # Gradient of obj w.r.t h_norm
    d_obj_dh_norm = (dS_dh * s - 2 * S) / (s ** 3)
    
    # Chain rule: h_norm = h / s, h = 2*p - 1
    # dh_norm/dh = (1/s) * (I - h/s * 1^T)
    # dh/dp = 2 * I
    # dp/dlogits from Gumbel-Softmax: p_i * (delta_ij - p_j) / temperature
    
    # Simplified: compute gradient w.r.t logits via finite difference or analytical
    # For efficiency, use the fact that d_obj/dlogits_j = sum_i d_obj/dp_i * dp_i/dlogits_j
    
    # dp_i/dlogits_j = p_i * (delta_ij - p_j) / temperature
    # This is expensive to compute fully, so we approximate
    
    # Approximate gradient: use the sensitivity of p to logits
    # dp_i/dlogits_j ≈ p_i * (delta_ij - p_j) / temperature
    # d_obj/dlogits_j = sum_i d_obj/dh_norm_i * (1/s - h_i/s^2) * 2 * p_i * (delta_ij - p_j) / temperature
    
    # Simplify: for the j-th logit
    # d_obj/dlogits_j = (2/temperature) * sum_i d_obj/dh_norm_i * (1/s - h_i/s^2) * p_i * (delta_ij - p_j)
    # = (2/temperature) * [p_j * sum_i d_obj/dh_norm_i * (1/s - h_i/s^2) * p_i - p_j * sum_i d_obj/dh_norm_i * (1/s - h_i/s^2) * p_i * p_j]
    
    # This is getting complex; let's use a simpler approximation
    # The key insight: changing logit_j changes p_j, which changes h_j
    # Approximate gradient as proportional to d_obj/dh_j * sensitivity
    
    # For practical purposes, use a simplified gradient
    grad_h = d_obj_dh_norm
    # Sensitivity of p to logits (approximate)
    sens = p * (1 - p) / temperature
    grad_logits = 2.0 * grad_h * sens
    
    return obj_val, grad_logits

def _optimize_gumbel_signs(h_init, deadline, rng):
    """
    Optimize sign pattern using Gumbel-Softmax relaxation.
    h_init: initial signed sequence (will be normalized to ±1 signs)
    Returns: optimized signed sequence
    """
    # Initialize logits from h_init
    # If h_init[i] > 0, logit should be positive (favor +1)
    h_signs = np.sign(h_init)
    h_abs = np.abs(h_init)
    # Initialize logits proportional to magnitude
    logits = np.log(h_abs + 1e-8) + 2.0 * h_signs
    
    # Anneal temperature from high to low
    temperatures = [5.0, 3.0, 2.0, 1.5, 1.0, 0.8, 0.6, 0.5]
    betas = [10, 25, 50, 100]
    
    best_val = _score(h_init)
    best_logits = logits.copy()
    
    for temp in temperatures:
        if time.time() > deadline:
            break
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            try:
                def obj(x, t=temp, b=beta):
                    val, _ = _gumbel_sign_objective(x, t, b, rng)
                    return val
                
                def grad(x, t=temp, b=beta):
                    _, g = _gumbel_sign_objective(x, t, b, rng)
                    return g
                
                res = minimize(obj, logits, jac=grad, method='L-BFGS-B',
                               options={'maxiter': 50, 'disp': False, 'gtol': 1e-5, 'maxfun': 100})
                new_logits = res.x
                
                # Convert to signs
                p = _gumbel_softmax_sample(new_logits, 0.1, rng)  # Low temp for nearly hard decision
                h_new = 2.0 * p - 1.0
                s = np.sum(h_new)
                if abs(s) < 1e-6:
                    continue
                h_new = h_new / s
                
                val = _score(h_new)
                if val < best_val:
                    best_val = val
                    best_logits = new_logits.copy()
                    logits = new_logits.copy()
            except Exception:
                pass
    
    # Final hard assignment
    p = _gumbel_softmax_sample(best_logits, 0.05, rng)
    h_final = 2.0 * p - 1.0
    s = np.sum(h_final)
    if abs(s) < 1e-6:
        return h_init
    h_final = h_final / s
    
    # Refine with local search
    h_ref, val_ref = _local_search_flip(h_final, deadline, max_passes=3)
    return h_ref

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

    # Phase 1: Standard smooth optimization
    for h_init in inits:
        if time.time() > deadline - 40:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        for beta in [10, 25, 50]:
            if time.time() > deadline - 35:
                break
            
            try:
                def obj(x, b=beta):
                    return _objective_smooth(x, b)
                
                def grad(x, b=beta):
                    return _grad_smooth(x, b)
                
                res = minimize(obj, h_norm, jac=grad, method='L-BFGS-B',
                               options={'maxiter': 80, 'disp': False, 'gtol': 1e-6, 'maxfun': 150})
                h_opt = res.x
                
                s_opt = np.sum(h_opt)
                if abs(s_opt) < 1e-6:
                    continue
                h_opt = h_opt / s_opt
                update_best(h_opt)
                h_norm = h_opt
            except Exception:
                pass
    
    # Phase 2: Gumbel-Softmax sign optimization on top candidates
    if best_h is not None and time.time() < deadline - 50:
        # Use the best result so far as seed for Gumbel optimization
        h_seed = best_h.copy()
        try:
            h_gumbel = _optimize_gumbel_signs(h_seed, deadline - 10, rng)
            update_best(h_gumbel)
        except Exception:
            pass
        
        # Also try on a few other diverse initializations
        for h_init in inits[:3]:
            if time.time() > deadline - 20:
                break
            try:
                h_gumbel = _optimize_gumbel_signs(h_init, deadline - 10, rng)
                update_best(h_gumbel)
            except Exception:
                pass
    
    # Phase 3: Local refinement
    if best_h is not None and time.time() < deadline - 10:
        h_ref, val_ref = _local_search_flip(best_h, deadline - 5, max_passes=8)
        update_best(h_ref)
        
    if best_h is not None and time.time() < deadline - 5:
        h_ref, val_ref = _local_search_block(best_h, deadline - 2, block_size=4, max_passes=5)
        update_best(h_ref)
        
    # Phase 4: Final high-precision polish
    if best_h is not None and time.time() < deadline - 3:
        # Tighter optimization on the best result
        h_polish = best_h.copy()
        s_polish = np.sum(h_polish)
        if abs(s_polish) > 1e-6:
            h_polish = h_polish / s_polish
            try:
                def obj_polish(x):
                    return _objective_smooth(x, 200.0)
                
                def grad_polish(x):
                    return _grad_smooth(x, 200.0)
                
                res = minimize(obj_polish, h_polish, jac=grad_polish, method='L-BFGS-B',
                               options={'maxiter': 200, 'disp': False, 'gtol': 1e-8, 'maxfun': 500})
                h_opt = res.x
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    update_best(h_opt)
            except Exception:
                pass
    
    if best_h is None:
        return np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
