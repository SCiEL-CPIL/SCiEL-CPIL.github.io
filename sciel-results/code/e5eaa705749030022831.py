import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        return 0.0
    h = np.clip(h, 0.0, None)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution: (h*h)[k] = sum_i h[i] * h[i+k] for all lags k
    # Use FFT for efficiency
    n = len(h)
    # Pad to avoid circular convolution issues
    size = 2 * n
    h_fft = np.fft.fft(h, size=size)
    h_conv = np.real(np.fft.ifft(h_fft * h_fft))
    # h_conv[k] = sum_i h[i] * h[(i-k) mod size] ... need to be careful
    # Actually, ifft(fft(h)^2) gives circular convolution. For linear:
    # Let's just compute directly for correctness since N=50 is small
    h_conv_linear = np.correlate(h, h, mode='full')
    # h_conv_linear[k] = sum_i h[i] * h[i + k - (n-1)] for k=0..2n-2
    # The max over all lags
    inf_norm = np.max(h_conv_linear)
    if inf_norm < 1e-15:
        return 0.0
    return l2_sq / (l1 * inf_norm)

def neg_ratio(params):
    """Negative ratio for minimization, with L1 normalization."""
    h = np.abs(params)  # Ensure nonnegative
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h = h / l1  # Normalize to L1 = 1
    return -compute_ratio(h)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_h = np.ones(N) / N  # Start with uniform, L1=1
    best_val = compute_ratio(best_h)
    
    # Try some heuristic initial points
    heuristics = []
    
    # 2-point cluster
    h2 = np.zeros(N)
    h2[0] = 2.0
    h2[1] = 1.0
    h2 /= np.sum(h2)
    heuristics.append(h2)
    
    # 3-point cluster
    h3 = np.zeros(N)
    h3[0] = 1.0
    h3[1] = 2.0
    h3[2] = 1.0
    h3 /= np.sum(h3)
    heuristics.append(h3)
    
    # Concentrated single point
    h1 = np.zeros(N)
    h1[0] = 1.0
    heuristics.append(h1)
    
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    hg = np.exp(-x**2 / 0.1)
    hg /= np.sum(hg)
    heuristics.append(hg)
    
    # Broader Gaussian
    hg2 = np.exp(-x**2 / 0.5)
    hg2 /= np.sum(hg2)
    heuristics.append(hg2)
    
    # Two separated peaks
    hsep = np.zeros(N)
    hsep[5] = 1.0
    hsep[40] = 1.0
    hsep /= np.sum(hsep)
    heuristics.append(hsep)
    
    for h_init in heuristics:
        val = compute_ratio(h_init)
        if val > best_val:
            best_val = val
            best_h = h_init.copy()
    
    # Local optimization from best heuristic
    for h_start in [best_h] + heuristics[:3]:
        if time.time() > deadline - 30:
            break
        
        # Use Nelder-Mead for local optimization
        try:
            res = minimize(neg_ratio, h_start, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
            h_opt = np.abs(res.x)
            l1 = np.sum(h_opt)
            if l1 > 1e-15:
                h_opt = h_opt / l1
                val = compute_ratio(h_opt)
                if val > best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Try random restarts if time permits
    rng = np.random.default_rng(42)
    for i in range(50):
        if time.time() > deadline - 10:
            break
        
        # Random sparse initialization
        h_rand = np.zeros(N)
        n_nonzero = rng.integers(1, 15)
        indices = rng.choice(N, size=n_nonzero, replace=False)
        h_rand[indices] = rng.random(n_nonzero) + 0.1
        l1 = np.sum(h_rand)
        h_rand /= l1
        
        val = compute_ratio(h_rand)
        if val > best_val:
            best_val = val
            best_h = h_rand.copy()
        
        # Quick local opt from random start
        try:
            res = minimize(neg_ratio, h_rand, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9,
                                  'adaptive': True})
            h_opt = np.abs(res.x)
            l1 = np.sum(h_opt)
            if l1 > 1e-15:
                h_opt = h_opt / l1
                val = compute_ratio(h_opt)
                if val > best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Return as list of floats
    return list(best_h)
