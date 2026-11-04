import numpy as np
import time
from scipy.optimize import minimize

N = 50
dx = 0.5 / N
_DEADLINE = 112.0  # seconds, safety margin before 120s kill

def _compute_ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        return -np.inf
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h * h)
    if l2_sq <= 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv <= 0:
        return 0.0
    return l2_sq / (dx * l1 * max_conv)

def _smooth_max(x, sigma=0.005):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    x_shifted = x - x_max
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_shifted / sigma))) * sigma
    return log_sum_exp

def _objective_complex(g_flat, sigma=0.005):
    """
    Objective to minimize: negative of the ratio.
    g_flat is a real vector of length 2*N representing complex g.
    h = |g|^2.
    """
    g = g_flat.reshape(N, 2)
    h = g[:, 0]**2 + g[:, 1]**2
    
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h = h / l1
    
    l2sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    peak = _smooth_max(conv, sigma=sigma)
    
    if peak == 0:
        return 0.0
    
    return -l2sq / (dx * peak)

def _optimize_complex(g0, time_limit, sigma=0.005, maxiter=500, ftol=1e-12, gtol=1e-8, maxfun=2000):
    """Run L-BFGS-B optimization on complex parameters."""
    start_time = time.time()
    
    try:
        result = minimize(
            lambda x: _objective_complex(x, sigma=sigma),
            g0,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxfun': maxfun}
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            g = g0.reshape(N, 2)
            h = g[:, 0]**2 + g[:, 1]**2
            return h
        
        g_opt = result.x.reshape(N, 2)
        h_opt = g_opt[:, 0]**2 + g_opt[:, 1]**2
        return h_opt
    except Exception:
        g = g0.reshape(N, 2)
        h = g[:, 0]**2 + g[:, 1]**2
        return h

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _compute_ratio(best_h)
    
    # Generate diverse initializations for h
    initializations_h = []
    
    x = np.linspace(-1, 1, N)
    
    # 1. Uniform
    initializations_h.append(np.ones(N))
    
    # 2. Gaussian-like
    initializations_h.append(np.exp(-x**2 / 0.1))
    initializations_h.append(np.exp(-x**2 / 0.5))
    initializations_h.append(np.exp(-x**2 / 2.0))
    
    # 3. Bimodal
    initializations_h.append(np.exp(-((x - 0.4)**2) / 0.1) + np.exp(-((x + 0.4)**2) / 0.1))
    initializations_h.append(np.exp(-((x - 0.6)**2) / 0.05) + np.exp(-((x + 0.6)**2) / 0.05))
    
    # 4. Triangle / tent
    initializations_h.append(1 - np.abs(x))
    
    # 5. Cosine window
    initializations_h.append(0.5 + 0.5 * np.cos(np.pi * x))
    
    # 6. Random smooth
    for seed in [1, 2, 3, 4, 5]:
        rng_local = np.random.RandomState(seed)
        h_rand = np.abs(rng_local.randn(N))
        kernel = np.ones(5) / 5
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        initializations_h.append(h_smooth)
    
    # 7. Peak in center
    h_center = np.zeros(N)
    h_center[N//2 - 2:N//2 + 3] = 1.0
    initializations_h.append(h_center)
    
    # 8. Peak at edges
    h_edge = np.zeros(N)
    h_edge[0:3] = 1.0
    h_edge[-3:] = 1.0
    initializations_h.append(h_edge)
    
    # 9. Alternating
    h_alt = np.ones(N)
    h_alt[::2] = 0.5
    initializations_h.append(h_alt)
    
    # 10. Linear ramp
    initializations_h.append(np.linspace(0.1, 1.0, N))
    initializations_h.append(np.linspace(1.0, 0.1, N))
    
    # 11. Structured patterns
    initializations_h.append(np.sin(np.pi * (x + 1) / 2)**2)
    initializations_h.append(np.cos(np.pi * x / 2)**2)
    
    # 12. DFT-based (binary mask approach)
    for seed in [10, 20, 30, 40, 50]:
        rng_local = np.random.RandomState(seed)
        z = np.zeros(N, dtype=complex)
        K = rng_local.randint(5, 15)
        active_bins = rng_local.choice(N, size=K, replace=False)
        z[active_bins] = rng_local.randn(K) + 1j * rng_local.randn(K)
        H = np.fft.fft(z)
        h_dft = np.abs(H) ** 2
        h_dft = h_dft / np.max(h_dft)
        initializations_h.append(h_dft)
    
    # 13. DST-based sparse spectral
    for seed in [60, 70, 80]:
        rng_local = np.random.RandomState(seed)
        K = rng_local.randint(3, 7)
        active_bins = rng_local.choice(N, size=K, replace=False)
        amps = rng_local.rand(K) + 0.5
        phases = rng_local.rand(K) * 2 * np.pi
        spec = np.zeros(N)
        spec[active_bins] = amps * np.cos(phases)
        h_dst = np.sin(np.pi * (np.arange(N) + 1) * (np.arange(N) + 1) / (2 * (N + 1))) @ spec
        h_dst = np.abs(h_dst)
        if np.max(h_dst) > 0:
            h_dst = h_dst / np.max(h_dst)
        initializations_h.append(h_dst)
    
    time_per_opt = 5.0  # seconds per optimization attempt
    
    for i, h0 in enumerate(initializations_h):
        elapsed = time.time() - start_time
        if elapsed + time_per_opt > _DEADLINE:
            break
        
        # Convert h0 to g0
        g0_real = np.sqrt(h0)
        g0_imag = np.zeros(N)
        g0_flat = np.concatenate([g0_real, g0_imag])
        
        h_opt = _optimize_complex(g0_flat, time_per_opt)
        
        score = _compute_ratio(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt
    
    # Final polish: run a few more iterations from the best solution with tighter tolerances
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 10:
        # Try small perturbations around the best
        for pert_seed in [100, 200, 300, 400, 500]:
            elapsed = time.time() - start_time
            if elapsed + 5 > _DEADLINE:
                break
            rng_local = np.random.RandomState(pert_seed)
            g_best_real = np.sqrt(best_h)
            g_best_imag = np.zeros(N)
            
            pert_real = g_best_real + rng_local.randn(N) * 0.05
            pert_imag = rng_local.randn(N) * 0.05
            
            g_pert_flat = np.concatenate([pert_real, pert_imag])
            
            h_opt = _optimize_complex(g_pert_flat, 5.0)
            score = _compute_ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt
    
    # High-precision polish: tighter tolerances and more iterations
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 5:
        g_best_real = np.sqrt(best_h)
        g_best_imag = np.zeros(N)
        g_best_flat = np.concatenate([g_best_real, g_best_imag])
        
        # Run with tighter tolerances
        h_polished = _optimize_complex(g_best_flat, _DEADLINE - elapsed - 2, sigma=0.002, maxiter=1000, ftol=1e-14, gtol=1e-10, maxfun=5000)
        score = _compute_ratio(h_polished)
        if score > best_score:
            best_score = score
            best_h = h_polished
    
    return [float(x) for x in best_h]
