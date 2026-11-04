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

def boxcar_to_h(params):
    """Convert boxcar parameters to step function.
    params: [amp1, mu1, w1, amp2, mu2, w2, ...]
    Each boxcar: amp * (1 if |x - mu| <= w/2 else 0)
    """
    h = np.zeros(N)
    n_boxes = len(params) // 3
    for k in range(n_boxes):
        amp = params[3*k]
        mu = params[3*k+1]
        w = abs(params[3*k+2])
        if w < 1e-8:
            w = 1e-8
        # Boxcar from mu - w/2 to mu + w/2
        left = mu - w/2
        right = mu + w/2
        mask = (X >= left) & (X < right)
        h[mask] += amp
    return np.clip(h, 0.0, None)

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
    
    # === Phase 1: Differential Evolution over Boxcar Superposition ===
    # Represent f as sum of shifted, variable-width rectangular pulses.
    # Sparse high-contrast geometric structures can minimize peak autoconvolution.
    
    def de_objective_boxcar(params):
        h = boxcar_to_h(params)
        if h.sum() < 1e-15:
            return 1e10
        return ratio_from_h(h)
    
    # Try different numbers of boxcars
    boxcar_configs = [
        {'n_boxes': 2, 'popsize': 25, 'maxiter': 200},
        {'n_boxes': 3, 'popsize': 25, 'maxiter': 150},
        {'n_boxes': 4, 'popsize': 20, 'maxiter': 120},
    ]
    
    for cfg in boxcar_configs:
        if time.time() > deadline - 30.0:
            break
            
        n_b = cfg['n_boxes']
        popsize = cfg['popsize']
        maxiter = cfg['maxiter']
        
        # Bounds: amp in [0, 2], mu in [-0.2, 0.2], w in [0.01, 0.3]
        bounds = []
        for _ in range(n_b):
            bounds.append((0.0, 2.0))   # amp
            bounds.append((-0.2, 0.2))  # mu
            bounds.append((0.01, 0.3))  # w
            
        try:
            result = opt.differential_evolution(
                de_objective_boxcar,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-6,
                seed=42,
                polish=False,
                init='sobol'
            )
            
            if result.fun < best_val:
                h_opt = boxcar_to_h(result.x)
                update_best(h_opt, result.fun)
        except Exception:
            pass

    # === Phase 2: Coordinate Descent Polishing ===
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
    
    # === Phase 3: Hand-crafted boxcar shapes ===
    if time.time() < deadline - 2.0:
        crafted = [
            # Two symmetric boxcars
            np.where((X >= -0.2) & (X < -0.05), 1.0, 0.0) + np.where((X >= 0.05) & (X < 0.2), 1.0, 0.0),
            # Three boxcars
            np.where((X >= -0.2) & (X < -0.12), 1.0, 0.0) + np.where((X >= -0.02) & (X < 0.02), 1.0, 0.0) + np.where((X >= 0.12) & (X < 0.2), 1.0, 0.0),
            # Wide central boxcar
            np.where((X >= -0.1) & (X < 0.1), 1.0, 0.0),
            # Two offset boxcars
            np.where((X >= -0.18) & (X < -0.08), 1.0, 0.0) + np.where((X >= 0.08) & (X < 0.18), 1.0, 0.0),
            # Alternating pattern
            np.where((X >= -0.2) & (X < -0.15), 1.0, 0.0) + np.where((X >= -0.05) & (X < 0.05), 1.0, 0.0) + np.where((X >= 0.15) & (X < 0.2), 1.0, 0.0),
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
