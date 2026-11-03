import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 8000
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

def quick_ratio(h):
    """Fast ratio computation for coordinate descent."""
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    rng = np.random.default_rng(42)
    
    # === Phase 1: Differential Evolution over Gaussian Mixture Model ===
    # Structural change: Use DE for global search over a smooth parametric space.
    # This avoids the local minima traps of Nelder-Mead on non-smooth objectives.
    
    def model_to_h(params):
        # params: [base, amp1, mu1, sig1, amp2, mu2, sig2, ...]
        base = params[0]
        h = np.full(N, base)
        for k in range(1, len(params), 3):
            amp = params[k]
            mu = params[k+1]
            sig = params[k+2]
            if sig < 1e-6:
                sig = 1e-6
            h += amp * np.exp(-0.5 * ((X - mu) / sig) ** 2)
        return np.clip(h, 0.0, None)
    
    def de_objective(params):
        h = model_to_h(params)
        return ratio_from_h(h)
    
    # We will test two model complexities: 2 Gaussians and 3 Gaussians
    # DE is stochastic and robust, so we can afford fewer iterations if the population is larger.
    
    configs = [
        {'n_gauss': 2, 'popsize': 20, 'maxiter': 150, 'bounds_scale': 1.0},
        {'n_gauss': 3, 'popsize': 20, 'maxiter': 100, 'bounds_scale': 1.0},
    ]
    
    for cfg in configs:
        if time.time() > deadline - 30.0:
            break
            
        n_g = cfg['n_gauss']
        popsize = cfg['popsize']
        maxiter = cfg['maxiter']
        
        # Define bounds for parameters
        # base: [0, 1]
        # amp: [-1, 1] (can be negative to create dips, clipped to 0 later)
        # mu: [-0.2, 0.2]
        # sig: [0.01, 0.2]
        
        bounds = [(0.0, 1.0)] # base
        for _ in range(n_g):
            bounds.append((-1.0, 1.0)) # amp
            bounds.append((-0.2, 0.2)) # mu
            bounds.append((0.01, 0.2)) # sig
            
        # Differential Evolution
        try:
            result = opt.differential_evolution(
                de_objective,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-5,
                seed=42,
                polish=False, # We will do our own polishing
                init='sobol'
            )
            
            if result.fun < best_val:
                h_opt = model_to_h(result.x)
                update_best(h_opt, result.fun)
        except Exception:
            pass

    # === Phase 2: Coordinate Descent Polishing ===
    # DE finds a good neighborhood, but coordinate descent can squeeze out more performance.
    if time.time() < deadline - 5.0:
        rng2 = np.random.default_rng(777)
        cur_h = best_h.copy()
        cur_val = best_val
        
        n_iter = 5000
        
        for it in range(n_iter):
            if time.time() > deadline - 2.0:
                break
            
            # Pick random coordinates
            n_coords = int(rng2.integers(1, 6))
            idxs = rng2.choice(N, size=n_coords, replace=False)
            
            h_try = cur_h.copy()
            for idx in idxs:
                # Multiplicative change
                factor = float(rng2.uniform(0.8, 1.2))
                h_try[idx] = max(0.0, cur_h[idx] * factor)
            
            v = quick_ratio(h_try)
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)
    
    # === Phase 3: Hand-crafted shapes ===
    if time.time() < deadline - 2.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        crafted = [
            np.exp(-2.0 * x_norm**2),
            np.exp(-0.5 * x_norm**2),
            0.5 + 0.5 * np.exp(-8.0 * x_norm**2),
            np.exp(-20.0 * (x_norm - 0.3)**2) + np.exp(-20.0 * (x_norm + 0.3)**2),
            0.1 + np.exp(-30.0 * (x_norm - 0.4)**2) + np.exp(-30.0 * (x_norm + 0.4)**2),
            1.0 - 0.3 * np.exp(-5.0 * x_norm**2),
            0.5 + 0.5 * x_norm**2,
            0.5 + 0.5 * np.cos(np.pi * x_norm),
        ]
        
        for h in crafted:
            if time.time() > deadline - 1.0:
                break
            h = np.clip(h, 0, None)
            if h.sum() < 1e-12:
                continue
            v = ratio_from_h(h)
            update_best(h, v)

    return list(best_h)
