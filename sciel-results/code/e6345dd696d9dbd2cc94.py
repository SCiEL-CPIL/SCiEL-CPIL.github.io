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
    
    # === Phase 1: B-spline basis optimization ===
    # Represent h as nonnegative superposition of B-splines
    # B-splines of order 3 (quadratic) with varying widths
    
    n_bsplines = 8
    
    def bspline_basis(x, center, width, order=3):
        """Quadratic B-spline basis centered at 'center' with 'width'."""
        t = (x - center) / width
        # Quadratic B-spline: (1 - |t|)^2 for |t| <= 1, 0 otherwise
        result = np.zeros_like(x)
        mask = np.abs(t) <= 1.0
        result[mask] = (1.0 - np.abs(t[mask])) ** 2
        return result
    
    def bspline_to_h(coeffs):
        """Convert B-spline coefficients to step function heights."""
        h = np.zeros(N)
        n = len(coeffs) // 3
        for i in range(n):
            amp = max(0.0, coeffs[3*i])
            center = coeffs[3*i + 1]
            width = max(0.01, coeffs[3*i + 2])
            h += amp * bspline_basis(X, center, width)
        return h
    
    def bspline_ratio(coeffs):
        h = bspline_to_h(coeffs)
        return ratio_from_h(h)
    
    rng = np.random.default_rng(123)
    
    # Try multiple B-spline configurations
    n_bs_starts = 20
    
    for start in range(n_bs_starts):
        if time.time() > deadline - 50.0:
            break
        
        # Random initialization of B-spline parameters
        coeffs = np.zeros(3 * n_bsplines)
        for i in range(n_bsplines):
            coeffs[3*i] = float(rng.uniform(0.5, 3.0))  # amplitude
            coeffs[3*i + 1] = float(rng.uniform(-0.2, 0.2))  # center
            coeffs[3*i + 2] = float(rng.uniform(0.03, 0.12))  # width
        
        try:
            result = opt.minimize(
                bspline_ratio,
                coeffs,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if result.fun < best_val:
                h_opt = bspline_to_h(result.x)
                update_best(h_opt, result.fun)
        except Exception:
            pass
    
    # === Phase 2: Fourier-domain optimization ===
    N_HARM = 18
    
    def fourier_to_h(coeffs):
        h = np.full(N, max(0.0, coeffs[0]))
        for k in range(1, N_HARM + 1):
            a_k = coeffs[2*k - 1]
            b_k = coeffs[2*k]
            h += a_k * np.cos(2 * np.pi * k * np.arange(N) / N)
            h += b_k * np.sin(2 * np.pi * k * np.arange(N) / N)
        return np.clip(h, 0.0, None)
    
    def fourier_ratio(coeffs):
        h = fourier_to_h(coeffs)
        return ratio_from_h(h)
    
    n_fourier_starts = 15
    
    for start in range(n_fourier_starts):
        if time.time() > deadline - 30.0:
            break
        
        coeffs = np.zeros(2 * N_HARM + 1)
        coeffs[0] = float(rng.uniform(0.5, 2.0))
        for k in range(1, N_HARM + 1):
            coeffs[2*k - 1] = float(rng.uniform(-0.4, 0.4))
            coeffs[2*k] = float(rng.uniform(-0.4, 0.4))
        
        try:
            result = opt.minimize(
                fourier_ratio,
                coeffs,
                method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            if result.fun < best_val:
                h_opt = fourier_to_h(result.x)
                update_best(h_opt, result.fun)
        except Exception:
            pass
    
    # === Phase 3: Gaussian superposition ===
    def model_to_h(params):
        h = np.full(N, max(0.0, params[0]))
        for k in range(1, len(params), 3):
            amp = max(0.0, params[k])
            mu = params[k+1]
            sig = max(0.01, params[k+2])
            h += amp * np.exp(-0.5 * ((X - mu) / sig) ** 2)
        return np.clip(h, 0.0, None)
    
    def nll(params):
        h = model_to_h(params)
        return ratio_from_h(h)
    
    for n_g in [1, 2]:
        n_starts = 10
        for start in range(n_starts):
            if time.time() > deadline - 15.0:
                break
            
            base = float(rng.uniform(0.0, 0.2))
            params = np.array([base])
            for _ in range(n_g):
                amp = float(rng.uniform(0.1, 2.5))
                mu = float(rng.uniform(-0.12, 0.12))
                sig = float(rng.uniform(0.02, 0.08))
                params = np.append(params, [amp, mu, sig])
            
            try:
                result = opt.minimize(
                    nll,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 120, 'xatol': 1e-4, 'fatol': 1e-6}
                )
                if result.fun < best_val:
                    h_opt = model_to_h(result.x)
                    update_best(h_opt, result.fun)
            except Exception:
                pass
    
    # === Phase 4: Coordinate descent refinement ===
    if time.time() < deadline - 5.0:
        rng2 = np.random.default_rng(777)
        cur_h = best_h.copy()
        cur_val = best_val
        
        step_scales = [1.0, 0.5, 0.25, 0.1]
        n_iter = 8000
        
        for it in range(n_iter):
            if time.time() > deadline - 2.0:
                break
            
            scale_idx = min(it // 2000, len(step_scales) - 1)
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
    
    # === Phase 5: Hand-crafted shapes ===
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
