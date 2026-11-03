"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _get_fft_size(n):
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize, size

def _autoconv_fft(h):
    n = len(h)
    fsize, size = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _score(h):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _gen_chirp(n):
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_sinc(n, beta=5.0):
    i = np.arange(n)
    x = (i - n/2) / (n * beta)
    h = np.sinc(x)
    return h

def _gen_gaussian(n, sigma_frac=0.25):
    i = np.arange(n)
    mu = n / 2
    sigma = n * sigma_frac
    return np.exp(-(i - mu)**2 / (2 * sigma**2))

def _gen_barker_like(n):
    h = np.zeros(n)
    for i in range(n):
        if i % 4 in [0, 2]:
            h[i] = 1.0
        else:
            h[i] = -1.0
    return h

def _gen_msequence_like(n, period=7):
    h = np.zeros(n)
    for i in range(n):
        val = ((i // period) % 2) ^ ((i % period) % 2)
        h[i] = 1.0 if val == 0 else -1.0
    return h

def _gen_random_smoothed(n, seed=42, kernel_size=10):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(noise, kernel, mode='same')

def _gen_binary_random(n, seed=42):
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < 0.5, 1.0, -1.0)

def _compute_ac_delta(h, i, delta):
    n = len(h)
    size = 2 * n - 1
    ac_change = np.zeros(size)
    
    # The change in autoconvolution when h[i] changes by delta
    # ac_new[k] = ac_old[k] + 2*delta*h[k-i] + delta^2 * (1 if k==2i else 0)
    ac_change[i:i+n] += 2 * delta * h
    
    if 2 * i < size:
        ac_change[2 * i] += delta * delta
        
    return ac_change

def _local_search_flip(h, max_passes=5, deadline=None):
    h = h.copy()
    ac = _autoconv_fft(h)
    s = np.sum(h)
    best_val = np.max(np.abs(ac)) / (s * s)
    cur_val = best_val
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            delta = -2 * h[i]
            ac_change = _compute_ac_delta(h, i, delta)
            ac_new = ac + ac_change
            s_new = s + delta
            
            if abs(s_new) < 1e-9:
                continue
            
            new_val = np.max(np.abs(ac_new)) / (s_new * s_new)
            
            if new_val < cur_val:
                ac = ac_new
                s = s_new
                h[i] = -h[i]
                cur_val = new_val
                improved = True
                if cur_val < best_val:
                    best_val = cur_val
            # else: reject
            
        if not improved:
            break
    
    return h, best_val

def _simulated_annealing(h, deadline, max_iter=50000):
    h = h.copy()
    ac = _autoconv_fft(h)
    s = np.sum(h)
    best_h = h.copy()
    best_val = np.max(np.abs(ac)) / (s * s)
    cur_val = best_val
    
    T = 1.0
    T_min = 1e-4
    cooling = 0.9995
    rng = np.random.default_rng(999)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        i = rng.integers(0, N)
        delta = -2 * h[i]
        ac_change = _compute_ac_delta(h, i, delta)
        ac_new = ac + ac_change
        s_new = s + delta
        
        if abs(s_new) < 1e-9:
            T *= cooling
            continue
        
        new_val = np.max(np.abs(ac_new)) / (s_new * s_new)
        
        delta_val = new_val - cur_val
        if delta_val < 0 or rng.random() < np.exp(-delta_val / max(T, 1e-15)):
            ac = ac_new
            s = s_new
            h[i] = -h[i]
            cur_val = new_val
            if cur_val < best_val:
                best_val = cur_val
                best_h = h.copy()
        else:
            pass
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_h, best_val

def _refine_continuous(h_init, deadline):
    """
    Refine a discrete or mixed solution using continuous gradient descent
    on the smooth surrogate.
    """
    s0 = np.sum(h_init)
    if abs(s0) < 1e-6:
        return h_init, _score(h_init)
    
    current_h = h_init / s0
    best_h = current_h.copy()
    best_val = _score(best_h)
    
    # Progressive refinement with increasing beta
    betas = [20, 50, 100, 200]
    for beta in betas:
        if time.time() > deadline:
            break
        
        def obj(x):
            return _objective_smooth(x, beta=beta)
        
        try:
            res = minimize(obj, current_h, method='L-BFGS-B', 
                           options={'maxiter': 50, 'disp': False, 'gtol': 1e-6})
            h_opt = res.x
            
            s_opt = np.sum(h_opt)
            if abs(s_opt) > 1e-6:
                h_opt = h_opt / s_opt
                val = _score(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                    current_h = h_opt.copy()
        except Exception:
            pass
            
    return best_h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # 1. Initial Diverse Set
    inits = []
    x = np.arange(N)
    
    inits.append(np.ones(N))
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(np.pi * x / (2*N))**2)
    inits.append(_gen_gaussian(N, sigma_frac=0.25))
    inits.append(_gen_gaussian(N, sigma_frac=0.15))
    inits.append(_gen_chirp(N))
    inits.append(_gen_sinc(N, beta=3.0))
    inits.append(_gen_barker_like(N))
    inits.append(_gen_msequence_like(N, period=7))
    
    # Add random smoothed
    for seed in [42, 123, 456]:
        inits.append(_gen_random_smoothed(N, seed=seed))
    
    # Add random binary
    for seed in [7, 17, 27]:
        inits.append(_gen_binary_random(N, seed=seed))

    # 2. Phase 1: Quick Continuous Optimization to find basins
    phase1_deadline = t_start + 40.0
    
    for h_init in inits:
        if time.time() > phase1_deadline:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        h_opt, val_opt = _refine_continuous(h_norm, phase1_deadline)
        
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()

    # 3. Phase 2: Discrete Search (SA + Local Search)
    phase2_deadline = t_start + 135.0
    
    # Discretize best continuous result for SA? 
    # Or just run SA on the continuous result (flipping signs of continuous values).
    # The task allows signed step heights. The "flip" operation h -> -h is valid for any h.
    
    # Run SA starting from the best continuous solution
    if time.time() < phase2_deadline - 5:
        h_sa, val_sa = _simulated_annealing(best_h, phase2_deadline - 5, max_iter=100000)
        if val_sa < best_score:
            best_score = val_sa
            best_h = h_sa.copy()
    
    # Polish with local search
    if time.time() < phase2_deadline:
        h_ls, val_ls = _local_search_flip(best_h, max_passes=5, deadline=phase2_deadline)
        if val_ls < best_score:
            best_score = val_ls
            best_h = h_ls.copy()
            
    # 4. Phase 3: Final Continuous Polish
    # The discrete search might have found a better "shape" (signs) but with suboptimal magnitudes.
    # Re-apply continuous refinement to the best discrete result.
    if time.time() < deadline - 2:
        h_final, val_final = _refine_continuous(best_h, deadline - 1)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()
            
    # Safety check
    if not np.all(np.isfinite(best_h)):
        return np.ones(N) / N
        
    return best_h

# EVOLVE-BLOCK-END
