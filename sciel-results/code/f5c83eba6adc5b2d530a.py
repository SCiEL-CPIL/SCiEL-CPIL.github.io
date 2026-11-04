import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute max(autoconv) / ||h||_2^2 using FFT on fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def smooth_max(values, beta=100.0):
    """Soft-max via log-sum-exp for smooth surrogate of max."""
    m = np.max(values)
    if m < -1000:
        return m
    return m + np.log(np.sum(np.exp(beta * (values - m)))) / beta

def smooth_objective(params, beta=50.0):
    """
    Smooth objective: soft-max of autoconv / ||h||_2^2.
    params: [mu1, sigma1, amp1, mu2, sigma2, amp2, ..., base]
    """
    n_gauss = (len(params) - 1) // 3
    base = params[-1]
    h = np.full(N, max(base, 0.0), dtype=np.float64)
    for i in range(n_gauss):
        mu = params[3*i]
        sigma = abs(params[3*i + 1]) + 1e-6
        amp = params[3*i + 2]
        h += amp * np.exp(-0.5 * ((X - mu) / sigma) ** 2)
    
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return 1e6
    
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return 1e6
    
    # Use soft-max over the convolution to get smooth objective
    # Scale conv so that the soft-max is well-conditioned
    conv_scaled = conv / l2
    smax = smooth_max(conv_scaled, beta)
    return smax

def gaussians_to_h(params):
    """
    Convert Gaussian superposition parameters to step heights.
    params = [mu1, sigma1, amp1, mu2, sigma2, amp2, ..., base]
    h(x) = base + sum_i amp_i * exp(-0.5 * ((x - mu_i)/sigma_i)^2)
    """
    n_gauss = (len(params) - 1) // 3
    base = params[-1]
    h = np.full(N, base, dtype=np.float64)
    for i in range(n_gauss):
        mu = params[3*i]
        sigma = params[3*i + 1]
        amp = params[3*i + 2]
        sigma = abs(sigma) + 1e-6
        h += amp * np.exp(-0.5 * ((X - mu) / sigma) ** 2)
    return np.clip(h, 0.0, None)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with uniform
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Optimize Gaussian superposition with smooth objective
    for n_gauss in [3, 4, 5, 6, 7]:
        if time.time() > deadline - 20.0:
            break
        
        n_params = 3 * n_gauss + 1  # +1 for base
        
        # Bounds
        bounds = []
        for _ in range(n_gauss):
            bounds.append((-0.25, 0.25))   # mu
            bounds.append((0.01, 0.15))     # sigma
            bounds.append((-1.0, 5.0))      # amp
        bounds.append((0.0, 2.0))           # base
        
        # Try multiple restarts with smooth objective first, then exact
        n_restarts = 15
        for i in range(n_restarts):
            if time.time() > deadline - 15.0:
                break
            
            # Random initial point
            x0 = np.array([rng.uniform(b[0], b[1]) for b in bounds])
            
            try:
                # First optimize smooth objective with lower beta
                result1 = opt.minimize(
                    lambda p: smooth_objective(p, beta=20.0),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                # Then polish with higher beta
                result2 = opt.minimize(
                    lambda p: smooth_objective(p, beta=100.0),
                    result1.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                h_opt = gaussians_to_h(result2.x)
                true_val = ratio_from_h(h_opt)
                update_best(h_opt, true_val)
                
            except Exception:
                pass
    
    # Phase 2: Local spatial optimization on best_h
    if time.time() < deadline - 5.0:
        current_h = best_h.copy()
        current_val = best_val
        step_size = 0.05
        
        for _ in range(15):
            if time.time() > deadline - 5.0:
                break
            improved = False
            # Use a subset of indices to speed up
            indices = rng.choice(N, size=min(150, N), replace=False)
            for idx in indices:
                if time.time() > deadline - 5.0:
                    break
                original = current_h[idx]
                current_h[idx] = max(0, original + step_size)
                val_plus = ratio_from_h(current_h)
                if val_plus < current_val:
                    current_val = val_plus
                    improved = True
                else:
                    current_h[idx] = max(0, original - step_size)
                    val_minus = ratio_from_h(current_h)
                    if val_minus < current_val:
                        current_val = val_minus
                        improved = True
                    else:
                        current_h[idx] = original
            if not improved:
                step_size *= 0.5
                if step_size < 1e-4:
                    break
            update_best(current_h, current_val)
    
    return list(best_h)
