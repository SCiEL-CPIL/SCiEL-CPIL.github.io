"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_fft_size(n):
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize

def _autoconv_fft(h):
    n = len(h)
    fsize = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:2*n-1]

def _score(h):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _soft_max_abs(ac, beta=100.0):
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=100.0):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _gen_gaussian(n, sigma_ratio=0.25):
    x = np.arange(n)
    mu = n / 2
    sigma = n * sigma_ratio
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _gen_chirp(n):
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_barker_like(n):
    h = np.ones(n)
    for i in range(1, n):
        h[i] = 1.0 if ((i * 7 + 3) % 11) < 5 else -1.0
    return h

def _gen_random_smoothed(n, seed=42, kernel_size=10):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(noise, kernel, mode='same')

def _gen_sparse_support(n, k, seed=42, spacing='uniform'):
    """Generate a sparse support pattern with k active positions."""
    rng = np.random.default_rng(seed)
    if spacing == 'uniform':
        # Uniformly spaced with jitter
        base = np.linspace(0, n-1, k, endpoint=False).astype(int)
        jitter = rng.integers(-n//k//2, n//k//2 + 1, size=k)
        positions = np.sort((base + jitter) % n)
    elif spacing == 'clustered':
        # Clustered in a few regions
        n_clusters = max(2, k // 10)
        centers = rng.choice(n, size=n_clusters, replace=False)
        positions = np.sort(rng.choice(n, size=k, replace=False))
        # Bias towards clusters
        for c in centers:
            idx = np.argmin(np.abs(positions - c))
            positions[idx] = c + rng.integers(-5, 5) % n
        positions = np.sort(positions % n)
    else:  # random
        positions = np.sort(rng.choice(n, size=k, replace=False))
    return positions

def _local_search_adaptive(h, max_passes=5, deadline=None):
    """
    Adaptive local search: sign flips + multiplicative perturbations.
    Uses decreasing step sizes.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(42)
    
    step_sizes = [0.3, 0.15, 0.07, 0.03, 0.01]
    
    for pass_num in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        step = step_sizes[min(pass_num, len(step_sizes)-1)]
        
        # 1. Try sign flips
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        
        # 2. Try multiplicative perturbations on active elements
        active = np.where(np.abs(h) > 1e-8)[0]
        if len(active) > 0:
            for i in active:
                if deadline is not None and time.time() > deadline:
                    break
                orig = h[i]
                for factor in [1 + step, 1 - step]:
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

def _optimize_sparse_amplitudes(support, init_amplitudes, beta=100.0, deadline=None):
    """
    Optimize amplitudes for a fixed sparse support pattern.
    Uses gradient descent on the smooth surrogate.
    """
    h = np.zeros(N)
    h[support] = init_amplitudes
    
    s0 = np.sum(h)
    if abs(s0) < 1e-9:
        return h, 1e10
    h = h / s0  # Normalize sum to 1
    
    def obj(x):
        h_x = np.zeros(N)
        h_x[support] = x
        sx = np.sum(h_x)
        if abs(sx) < 1e-9:
            return 1e10
        h_norm = h_x / sx
        ac = _autoconv_fft(h_norm)
        sm = _soft_max_abs(ac, beta)
        return sm
    
    x0 = h[support].copy()
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', 
                       options={'maxiter': 100, 'disp': False, 'gtol': 1e-5})
        x_opt = res.x
        h_opt = np.zeros(N)
        h_opt[support] = x_opt
        s_opt = np.sum(h_opt)
        if abs(s_opt) > 1e-9:
            h_opt = h_opt / s_opt
            val = _score(h_opt)
            return h_opt, val
    except Exception:
        pass
    
    return h, _score(h)

def _simulated_annealing_sparse(h, deadline, max_iter=50000):
    """
    Simulated annealing on sign flips for sparse sequences.
    """
    h = h.copy()
    best_h = h.copy()
    best_val = _score(h)
    cur_val = best_val
    
    T = 1.0
    T_min = 1e-4
    cooling = 0.9997
    
    rng = np.random.default_rng(999)
    active = np.where(np.abs(h) > 1e-8)[0]
    
    if len(active) == 0:
        return best_h, best_val
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick random active index to flip
        i = active[rng.integers(0, len(active))]
        h[i] = -h[i]
        new_val = _score(h)
        
        delta = new_val - cur_val
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-15)):
            cur_val = new_val
            if cur_val < best_val:
                best_val = cur_val
                best_h = h.copy()
        else:
            h[i] = -h[i]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Baseline
    base_score = _score(best_h)
    best_score = base_score
    
    # Phase 1: Sparse combinatorial search with amplitude optimization
    phase1_deadline = t_start + 80.0
    
    # Try various sparsity levels and support patterns
    sparsity_configs = [
        (N // 4, 'uniform', 42),
        (N // 4, 'uniform', 123),
        (N // 4, 'clustered', 42),
        (N // 3, 'uniform', 77),
        (N // 3, 'clustered', 156),
        (N // 5, 'uniform', 42),
        (N // 5, 'random', 42),
        (N // 2, 'uniform', 42),
    ]
    
    for k, spacing, seed in sparsity_configs:
        if time.time() > phase1_deadline - 10:
            break
        
        support = _gen_sparse_support(N, k, seed=seed, spacing=spacing)
        
        # Initialize amplitudes with random signs
        rng = np.random.default_rng(seed + 1000)
        amplitudes = rng.choice([-1.0, 1.0], size=k)
        
        h_opt, val_opt = _optimize_sparse_amplitudes(support, amplitudes, beta=100.0, deadline=phase1_deadline - 5)
        
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()
    
    # Phase 2: Dense gradient optimization with diverse inits
    phase2_deadline = t_start + 120.0
    
    inits = []
    x = np.arange(N)
    inits.append(np.ones(N))
    inits.append(_gen_gaussian(N))
    inits.append(_gen_chirp(N))
    inits.append(_gen_barker_like(N))
    inits.append(_gen_random_smoothed(N, seed=1))
    inits.append(_gen_random_smoothed(N, seed=2))
    inits.append(np.cos(np.pi * x / N))
    
    gauss_env = _gen_gaussian(N, sigma_ratio=0.3)
    alt_signs = np.where(x % 2 == 0, 1.0, -1.0)
    inits.append(gauss_env * alt_signs)
    
    for h_init in inits:
        if time.time() > phase2_deadline - 15:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        try:
            betas = [50, 100, 200]
            current_h = h_norm.copy()
            
            for beta in betas:
                if time.time() > phase2_deadline - 10:
                    break
                
                def obj(x, beta=beta):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, current_h, method='L-BFGS-B', 
                               options={'maxiter': 80, 'disp': False, 'gtol': 1e-5})
                
                h_opt = res.x
                if abs(np.sum(h_opt)) < 1e-6:
                    continue
                
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
                
                current_h = h_opt.copy()
                
        except Exception:
            pass
    
    # Phase 3: Local search refinement
    if time.time() < deadline - 15:
        h_refined, val_refined = _local_search_adaptive(best_h, max_passes=8, deadline=deadline - 10)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # Phase 4: Simulated annealing on the refined solution
    if time.time() < deadline - 5:
        h_sa, val_sa = _simulated_annealing_sparse(best_h, deadline - 3, max_iter=30000)
        if val_sa < best_score:
            best_score = val_sa
            best_h = h_sa.copy()
    
    # Final polish
    if time.time() < deadline - 2:
        h_final, val_final = _local_search_adaptive(best_h, max_passes=3, deadline=deadline - 1)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()
    
    # Ensure valid output
    if not np.all(np.isfinite(best_h)):
        return np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
