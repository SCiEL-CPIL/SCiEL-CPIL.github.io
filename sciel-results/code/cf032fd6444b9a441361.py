import numpy as np
import time
from scipy.optimize import minimize

N = 50
w = 1.0 / 100.0

def compute_value(h):
    """Compute the objective value for given step heights h."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0
    
    # FFT based convolution
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft <<= 1
    
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:2*n-1]
    
    # Ensure g is real
    g = np.real(g)
    # Small negative values from numerical noise
    g = np.maximum(g, 0.0)
    
    m = len(g)
    
    # l2 calculation: sum (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # where y = [0, g[0], ..., g[m-1], 0]
    # = (w/3) * (2*sum(g^2) + sum(g[:-1]*g[1:]))
    g_sq = np.sum(g**2)
    cross = np.sum(g[:-1] * g[1:])
    l2 = (w / 3.0) * (2.0 * g_sq + cross)
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def neg_value_smooth(log_h, beta=100.0):
    """Smooth objective using log-sum-exp for the max term."""
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
    
    g_sq = np.sum(g**2)
    cross = np.sum(g[:-1] * g[1:])
    l2 = (w / 3.0) * (2.0 * g_sq + cross)
    
    sum_g = np.sum(g)
    norm_1 = sum_g / (m + 1)
    
    # Smooth max using log-sum-exp
    # max(g) ≈ (1/beta) * log(sum(exp(beta*g)))
    # But we need to be careful about overflow
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
    """Direct objective (exact max)."""
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    return -compute_value(h)

def construct():
    deadline = time.time() + 140.0
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial guesses in log space
    starts = []
    x = np.arange(N)
    center = (N - 1) / 2.0
    
    # 1. Uniform (all ones in log space = zeros)
    starts.append(np.zeros(N))
    
    # 2. Gaussians with various widths
    for sigma_sq in [1, 2, 4, 8, 16, 32, 64]:
        g = np.exp(-((x - center)**2) / (2 * sigma_sq))
        starts.append(np.log(g + 1e-9))
    
    # 3. Raised cosine
    g = 0.5 * (1 + np.cos(np.pi * x / (N - 1)))
    starts.append(np.log(g + 1e-9))
    
    # 4. Triangle
    g = 1 - np.abs(x - center) / center
    g[g < 0] = 1e-9
    starts.append(np.log(g))
    
    # 5. Sine bump
    g = np.sin(np.pi * x / (N - 1))
    g[g < 0] = 1e-9
    starts.append(np.log(g))
    
    # 6. Flat top with smooth edges
    for width in [0.3, 0.5, 0.7]:
        g = np.ones(N)
        edge = int(N * (1 - width) / 2)
        for i in range(edge):
            t = i / max(edge, 1)
            g[i] = 0.5 * (1 + np.cos(np.pi * t))
            g[N-1-i] = g[i]
        starts.append(np.log(g + 1e-9))
    
    # 7. Random smooth perturbations
    for _ in range(15):
        x_rand = rng.randn(N)
        # Smooth with various kernels
        kernel_size = rng.choice([3, 5, 7, 9])
        kernel = np.ones(kernel_size) / kernel_size
        x_rand = np.convolve(x_rand, kernel, mode='same')
        x_rand = x_rand - np.mean(x_rand)
        # Scale to reasonable range
        x_rand = x_rand * 0.5
        starts.append(x_rand)
    
    # 8. Two bumps
    for sep in [5, 10, 15]:
        g = np.exp(-((x - center + sep/2)**2) / 8) + np.exp(-((x - center - sep/2)**2) / 8)
        starts.append(np.log(g + 1e-9))
    
    # Optimization loop
    def try_optimize(start_log, use_smooth=True, maxiter=3000, maxfun=15000):
        """Try to optimize from a starting point, return (value, h)."""
        if use_smooth:
            obj = neg_value_smooth
            # Use higher beta for smoother approximation
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
            # Clip small negative values
            h_cand = np.maximum(h_cand, 0.0)
            v_cand = compute_value(h_cand)
            return v_cand, h_cand
        except Exception:
            return 0.0, None
    
    # Phase 1: Optimize from all starts
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
        
        # Try both smooth and direct objective for refinement
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
        
        for _ in range(30):
            if time.time() > deadline - 5.0:
                break
            
            # Random smooth perturbation
            perturb = rng.randn(N)
            kernel_size = rng.choice([3, 5, 7])
            kernel = np.ones(kernel_size) / kernel_size
            perturb = np.convolve(perturb, kernel, mode='same')
            perturb = perturb * 0.2  # Small perturbation
            
            start_log = best_log + perturb
            v_cand, h_cand = try_optimize(start_log, use_smooth=True, maxiter=2000, maxfun=10000)
            
            if h_cand is not None and v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
    
    # Phase 4: Final high-precision polish
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
