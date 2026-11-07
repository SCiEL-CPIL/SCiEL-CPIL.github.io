"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def compute_value(h):
    """Compute the value metric for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h, mode='full')
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    
    w = 1.0 / (m + 1)
    
    y_i = y[:m+1]
    y_ip1 = y[1:m+2]
    l2 = (w / 3.0) * np.sum(y_i**2 + y_i * y_ip1 + y_ip1**2)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value


def compute_value_fft(h):
    """FFT-based compute_value for speed."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft <<= 1
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:2*n-1]
    g = np.real(g)
    g = np.maximum(g, 0.0)
    m = len(g)
    w = 1.0 / (m + 1)
    g_sq = np.sum(g**2)
    cross = np.sum(g[:-1] * g[1:])
    l2 = (w / 3.0) * (2.0 * g_sq + cross)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def neg_value_smooth(log_h, beta=200.0):
    """Smooth objective using log-sum-exp for the max term, in log-space."""
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft <<= 1
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:2*n-1]
    g = np.real(g)
    g = np.maximum(g, 0.0)
    
    m = len(g)
    w = 1.0 / (m + 1)
    g_sq = np.sum(g**2)
    cross = np.sum(g[:-1] * g[1:])
    l2 = (w / 3.0) * (2.0 * g_sq + cross)
    
    sum_g = np.sum(g)
    norm_1 = sum_g / (m + 1)
    
    g_max_raw = np.max(g)
    if g_max_raw > 0:
        log_sum_exp = g_max_raw + np.log(np.sum(np.exp(beta * (g - g_max_raw))))
        norm_inf_smooth = (1.0 / beta) * log_sum_exp
    else:
        norm_inf_smooth = 0.0
    
    if norm_inf_smooth == 0 or norm_1 == 0:
        return 0.0
        
    return -l2 / (norm_1 * norm_inf_smooth)


def neg_value_direct(log_h):
    """Direct objective (exact max) in log-space."""
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    return -compute_value_fft(h)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Seed: 50 equal heights
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial guesses in log space
    starts = []
    x = np.arange(N)
    center = (N - 1) / 2.0
    
    # 1. Uniform
    starts.append(np.zeros(N))
    
    # 2. Gaussians with various widths
    for sigma_sq in [1, 2, 4, 8, 16, 32, 64]:
        g_init = np.exp(-((x - center)**2) / (2 * sigma_sq))
        starts.append(np.log(g_init + 1e-9))
    
    # 3. Raised cosine
    g_init = 0.5 * (1 + np.cos(np.pi * x / (N - 1)))
    starts.append(np.log(g_init + 1e-9))
    
    # 4. Triangle
    g_init = 1 - np.abs(x - center) / center
    g_init[g_init < 0] = 1e-9
    starts.append(np.log(g_init))
    
    # 5. Sine bump
    g_init = np.sin(np.pi * x / (N - 1))
    g_init[g_init < 0] = 1e-9
    starts.append(np.log(g_init))
    
    # 6. Flat top with smooth edges
    for width in [0.3, 0.5, 0.7]:
        g_init = np.ones(N)
        edge = max(1, int(N * (1 - width) / 2))
        for i in range(edge):
            t = i / max(edge, 1)
            g_init[i] = 0.5 * (1 + np.cos(np.pi * t))
            g_init[N-1-i] = g_init[i]
        starts.append(np.log(g_init + 1e-9))
    
    # 7. Random smooth perturbations
    for _ in range(10):
        x_rand = rng.randn(N)
        kernel_size = rng.choice([3, 5, 7, 9])
        kernel = np.ones(kernel_size) / kernel_size
        x_rand = np.convolve(x_rand, kernel, mode='same')
        x_rand = x_rand - np.mean(x_rand)
        x_rand = x_rand * 0.5
        starts.append(x_rand)
    
    # 8. Two bumps
    for sep in [5, 10, 15]:
        g_init = np.exp(-((x - center + sep/2)**2) / 8) + np.exp(-((x - center - sep/2)**2) / 8)
        starts.append(np.log(g_init + 1e-9))
    
    # 9. Plateau-like profiles
    for pw in [10, 15, 20, 25, 30]:
        h_init = np.zeros(N)
        c = N // 2
        half = pw // 2
        h_init[c - half:c + half + 1] = 1.0
        if half > 1:
            h_init[c - half] = 0.5
            h_init[c + half] = 0.5
        starts.append(np.log(h_init + 1e-9))
    
    def try_optimize(start_log, use_smooth=True, maxiter=3000, maxfun=15000):
        """Try to optimize from a starting point, return (value, h)."""
        if use_smooth:
            obj = neg_value_smooth
        else:
            obj = neg_value_direct
        
        try:
            res = minimize(
                obj,
                start_log,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'maxfun': maxfun, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            h_cand = np.exp(res.x - np.mean(res.x))
            h_cand = np.maximum(h_cand, 0.0)
            v_cand = compute_value(h_cand)
            return v_cand, h_cand
        except Exception:
            return 0.0, None
    
    # Phase 1: Optimize from all starts using smooth objective
    for i, start_log in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
        
        v_cand, h_cand = try_optimize(start_log, use_smooth=True, maxiter=3000, maxfun=15000)
        
        if h_cand is not None and v_cand > best_val:
            best_val = v_cand
            best_h = h_cand.copy()
    
    # Phase 2: Refine best solution with tighter tolerance
    if time.time() < deadline - 15.0:
        best_log = np.log(np.maximum(best_h, 1e-9))
        
        for use_smooth in [True, False]:
            if time.time() > deadline - 10.0:
                break
            v_cand, h_cand = try_optimize(best_log, use_smooth=use_smooth, maxiter=5000, maxfun=25000)
            if h_cand is not None and v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
    
    # Phase 3: Random perturbations around best
    if time.time() < deadline - 10.0:
        best_log = np.log(np.maximum(best_h, 1e-9))
        
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            
            perturb = rng.randn(N)
            kernel_size = rng.choice([3, 5, 7])
            kernel = np.ones(kernel_size) / kernel_size
            perturb = np.convolve(perturb, kernel, mode='same')
            perturb = perturb * 0.2
            
            start_log = best_log + perturb
            v_cand, h_cand = try_optimize(start_log, use_smooth=True, maxiter=2000, maxfun=10000)
            
            if h_cand is not None and v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
    
    # Phase 4: Final high-precision polish with direct objective
    if time.time() < deadline - 5.0:
        best_log = np.log(np.maximum(best_h, 1e-9))
        v_cand, h_cand = try_optimize(best_log, use_smooth=False, maxiter=10000, maxfun=50000)
        if h_cand is not None and v_cand > best_val:
            best_val = v_cand
            best_h = h_cand.copy()
    
    # Ensure output is valid
    best_h = np.asarray(best_h, dtype=float)
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 1.0)
    
    return list(best_h)
# EVOLVE-BLOCK-END
