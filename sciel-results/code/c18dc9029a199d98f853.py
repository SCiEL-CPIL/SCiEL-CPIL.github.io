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
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # === Phase 1: Envelope-modulated square wave optimization ===
    # Represent h as envelope(x) * (base + sum of harmonics with adjustable phases)
    # This ensures a periodic modulation structure that can reduce autoconvolution peak
    
    def envelope_modulated_h(params):
        """
        params: [env_width, env_center, base_level, 
                 amp1, phase1, freq1, amp2, phase2, freq2, ...]
        """
        env_width = max(0.05, params[0])
        env_center = params[1]
        base_level = max(0.0, params[2])
        
        # Smooth raised cosine envelope
        env = np.ones(N)
        mask = np.abs(X - env_center) > env_width
        if np.any(mask):
            t = (np.abs(X[mask] - env_center) - env_width) / (0.25 - env_width)
            t = np.clip(t, 0, 1)
            env[mask] = 0.5 * (1 + np.cos(np.pi * t))
        
        # Base + harmonic modulation
        h = np.full(N, base_level)
        n_harmonics = (len(params) - 3) // 3
        for i in range(n_harmonics):
            amp = params[3 + 3*i]
            phase = params[3 + 3*i + 1]
            freq = params[3 + 3*i + 2]
            h += amp * np.cos(2 * np.pi * freq * X + phase)
        
        h = env * h
        return np.clip(h, 0.0, None)
    
    def env_mod_ratio(params):
        h = envelope_modulated_h(params)
        if h.sum() < 1e-15:
            return 1e10
        return ratio_from_h(h)
    
    # Try different numbers of harmonics
    for n_harm in [1, 2, 3]:
        if time.time() > deadline - 60.0:
            break
        
        n_params = 3 + 3 * n_harm
        bounds = [(0.05, 0.25), (-0.25, 0.25), (0.0, 2.0)]
        for _ in range(n_harm):
            bounds.extend([(-0.5, 0.5), (0.0, 2.0 * np.pi), (1.0, 10.0)])
        
        n_starts = 8
        for start in range(n_starts):
            if time.time() > deadline - 55.0:
                break
            
            rng = np.random.default_rng(start * 100 + n_harm)
            x0 = np.zeros(n_params)
            x0[0] = rng.uniform(0.1, 0.2)
            x0[1] = 0.0
            x0[2] = rng.uniform(0.5, 1.5)
            for i in range(n_harm):
                x0[3 + 3*i] = rng.uniform(-0.3, 0.3)
                x0[3 + 3*i + 1] = rng.uniform(0, 2 * np.pi)
                x0[3 + 3*i + 2] = rng.uniform(2.0, 8.0)
            
            try:
                result = opt.differential_evolution(
                    env_mod_ratio,
                    bounds,
                    x0=x0,
                    maxiter=30,
                    tol=1e-5,
                    seed=start * 100 + n_harm,
                    polish=False,
                    workers=1
                )
                if result.fun < best_val:
                    h_opt = envelope_modulated_h(result.x)
                    update_best(h_opt, result.fun)
            except Exception:
                pass
    
    # === Phase 2: Coordinate descent refinement ===
    if time.time() < deadline - 10.0:
        rng2 = np.random.default_rng(777)
        cur_h = best_h.copy()
        cur_val = best_val
        
        step_scales = [1.0, 0.5, 0.25, 0.1]
        n_iter = 5000
        
        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break
            
            scale_idx = min(it // 1500, len(step_scales) - 1)
            scale = step_scales[scale_idx]
            
            n_coords = int(rng2.integers(1, 8))
            idxs = rng2.choice(N, size=n_coords, replace=False)
            
            h_try = cur_h.copy()
            for idx in idxs:
                factor_range = (1.0 - 0.12 * scale, 1.0 + 0.12 * scale)
                factor = float(rng2.uniform(*factor_range))
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
            np.exp(-4.0 * x_norm**2) + 0.2 * np.exp(-10.0 * (x_norm - 0.5)**2),
            np.exp(-4.0 * x_norm**2) + 0.2 * np.exp(-10.0 * (x_norm + 0.5)**2),
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
