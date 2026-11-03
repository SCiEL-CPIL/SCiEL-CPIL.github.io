"""Rewritten program: signed step heights minimizing max|autoconv|/(sum h)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(h):
    """Compute linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution."""
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _soft_max_abs(ac, beta=100.0):
    """
    Smooth approximation of max(|ac|) via log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=100.0):
    """
    Smooth objective for optimization: soft_max_abs(autoconv(h)) / (sum h)^2.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _gen_barker_like(n):
    """
    Generate a Barker-like sequence: +1/-1 with good autocorrelation properties.
    We use a pseudo-random sign sequence smoothed slightly.
    """
    rng = np.random.default_rng(123)
    h = rng.choice([-1.0, 1.0], size=n)
    # Light smoothing to help gradient methods
    kernel = np.array([0.2, 0.3, 0.3, 0.2])
    h = np.convolve(h, kernel, mode='same')
    return h

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_gaussian(n):
    """Generate a Gaussian-shaped sequence."""
    x = np.arange(n)
    mu = n / 2
    sigma = n / 4
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _gen_random_smooth(n, seed=42):
    """Generate a random smoothed sequence."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(8) / 8
    return np.convolve(noise, kernel, mode='same')

def _local_search(h, max_passes=12, deadline=None):
    """
    Local search with batched sign-flip evaluation.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(42)
    
    # Precompute FFT size
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Batch sign-flip evaluation
        # Flipping sign at index i: h' = h - 2*h[i]*e_i
        # autoconv(h') = autoconv(h) - 2*h[i]*(autoconv(e_i))
        # We can compute this efficiently
        ac_base = _autoconv_fft(h)
        
        # Try all indices in batch
        h_2d = np.tile(h, (N, 1))
        idx = np.arange(N)
        h_2d[np.arange(N), idx] = -h[idx]
        
        # Compute autoconv for each row via FFT
        H_2d = np.fft.rfft(h_2d, fsize, axis=1)
        ac_2d = np.fft.irfft(H_2d * H_2d, fsize, axis=1)[:, :size]
        
        s_2d = np.sum(h_2d, axis=1)
        scores = np.max(np.abs(ac_2d), axis=1) / (s_2d * s_2d)
        
        best_i = np.argmin(scores)
        if scores[best_i] < best_val * (1 - 1e-10):
            h = h_2d[best_i].copy()
            best_val = scores[best_i]
            improved = True
        
        # Try small multiplicative perturbations (individual, since batching is trickier)
        for _ in range(N // 2):
            if deadline is not None and time.time() > deadline:
                break
            i = rng.integers(N)
            orig = h[i]
            for factor in [1.03, 0.97, 1.06, 0.94]:
                if deadline is not None and time.time() > deadline:
                    break
                h[i] = orig * factor
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                    break
                else:
                    h[i] = orig
        
        if not improved:
            break
    
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_h = np.ones(N)
    
    base_score = _score(best_h)
    best_score = base_score
    
    inits = []
    
    # 1. Constant
    inits.append(np.ones(N))
    
    # 2. Gaussian
    inits.append(_gen_gaussian(N))
    
    # 3. Chirp
    inits.append(_gen_chirp(N))
    
    # 4. Barker-like
    inits.append(_gen_barker_like(N))
    
    # 5. Random smoothed
    inits.append(_gen_random_smooth(N, seed=1))
    
    # 6. Another random with different seed
    inits.append(_gen_random_smooth(N, seed=99))
    
    # 7. Alternating signs with envelope
    x = np.arange(N)
    envelope = np.exp(-0.5 * ((x - N/2) / (N/5))**2)
    inits.append(envelope * np.cos(np.pi * x / 5))
    
    for h_init in inits:
        if time.time() > deadline - 15:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        # Try multiple beta values for smooth objective
        for beta in [50, 100, 200]:
            if time.time() > deadline - 8:
                break
            
            try:
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B',
                               options={'maxiter': 300, 'disp': False, 'gtol': 1e-6, 'ftol': 1e-8})
                h_opt = res.x
                
                if abs(np.sum(h_opt)) < 1e-6:
                    continue
                    
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
            except Exception:
                pass
        
        # Also try without normalization (the objective handles it)
        if time.time() > deadline - 8:
            continue
        try:
            def obj_raw(x):
                return _objective_smooth(x, beta=100.0)
            
            res = minimize(obj_raw, h_init, method='L-BFGS-B',
                           options={'maxiter': 200, 'disp': False, 'gtol': 1e-5})
            h_opt = res.x
            if abs(np.sum(h_opt)) < 1e-6:
                continue
            val = _score(h_opt)
            if val < best_score:
                best_score = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final local search on best found
    if time.time() < deadline - 5:
        h_refined, val_refined = _local_search(best_h, max_passes=10, deadline=deadline - 2)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # High-precision polish: run L-BFGS-B with tighter tolerance on the best result
    if time.time() < deadline - 3:
        try:
            s0 = np.sum(best_h)
            if abs(s0) > 1e-9:
                h_norm = best_h / s0
                res = minimize(lambda x: _objective_smooth(x, beta=200.0), h_norm,
                               method='L-BFGS-B',
                               options={'maxiter': 500, 'disp': False, 'gtol': 1e-8, 'ftol': 1e-10})
                h_opt = res.x
                if abs(np.sum(h_opt)) > 1e-9:
                    val = _score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Safety: if somehow we lost track, fall back
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
