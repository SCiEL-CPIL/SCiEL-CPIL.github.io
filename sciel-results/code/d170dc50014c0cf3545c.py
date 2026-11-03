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
    
    # === Phase 1: Fourier-domain optimization with L-BFGS-B ===
    # Structural change: use gradient-based optimizer on Fourier coefficients
    
    N_HARM = 30  # Increase harmonics slightly, L-BFGS-B handles higher dims better
    
    # Precompute basis functions for speed
    # h[n] = a0 + sum_k (a_k cos_k[n] + b_k sin_k[n])
    # coeffs = [a0, a1, b1, ..., aN_HARM, bN_HARM]
    n_basis = 2 * N_HARM + 1
    basis = np.zeros((N, n_basis))
    basis[:, 0] = 1.0
    for k in range(1, N_HARM + 1):
        basis[:, 2*k - 1] = np.cos(2 * np.pi * k * np.arange(N) / N)
        basis[:, 2*k] = np.sin(2 * np.pi * k * np.arange(N) / N)
    
    def fourier_to_h(coeffs):
        """Convert Fourier coefficients to step heights using precomputed basis."""
        h = basis @ coeffs
        return np.clip(h, 0.0, None)
    
    def fourier_ratio(coeffs):
        h = fourier_to_h(coeffs)
        return ratio_from_h(h)
    
    def fourier_grad(coeffs):
        """Numerical gradient for L-BFGS-B."""
        eps = 1e-7
        grad = np.zeros_like(coeffs)
        f0 = fourier_ratio(coeffs)
        for i in range(len(coeffs)):
            coeffs_p = coeffs.copy()
            coeffs_p[i] += eps
            f_p = fourier_ratio(coeffs_p)
            grad[i] = (f_p - f0) / eps
            coeffs_p[i] = coeffs[i]
        return grad
    
    # Multi-start Fourier optimization with L-BFGS-B
    rng = np.random.default_rng(42)
    n_fourier_starts = 25
    
    for start in range(n_fourier_starts):
        if time.time() > deadline - 50.0:
            break
        
        # Initialize with low-amplitude random coefficients
        coeffs = np.zeros(n_basis)
        coeffs[0] = float(rng.uniform(0.5, 2.0))  # DC term positive
        for k in range(1, N_HARM + 1):
            coeffs[2*k - 1] = float(rng.uniform(-0.3, 0.3))
            coeffs[2*k] = float(rng.uniform(-0.3, 0.3))
        
        # Use L-BFGS-B to optimize Fourier coefficients
        try:
            result = opt.minimize(
                fourier_ratio,
                coeffs,
                jac=fourier_grad,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            if result.fun < best_val:
                h_opt = fourier_to_h(result.x)
                update_best(h_opt, result.fun)
        except Exception:
            pass
    
    # === Phase 2: Gaussian superposition model with scipy optimization ===
    
    def model_to_h(params):
        """params = [base, amp1, mu1, sig1, amp2, mu2, sig2, ...]"""
        h = np.full(N, params[0])
        for k in range(1, len(params), 3):
            amp = params[k]
            mu = params[k+1]
            sig = params[k+2]
            if sig < 1e-6:
                sig = 1e-6
            h += amp * np.exp(-0.5 * ((X - mu) / sig) ** 2)
        return np.clip(h, 0.0, None)
    
    def nll(params):
        h = model_to_h(params)
        return ratio_from_h(h)
    
    n_gaussians_options = [1, 2, 3]
    
    for n_g in n_gaussians_options:
        n_starts = 15
        for start in range(n_starts):
            if time.time() > deadline - 35.0:
                break
            
            # Initialize parameters
            base = float(rng.uniform(0.0, 0.3))
            params = np.array([base])
            for _ in range(n_g):
                amp = float(rng.uniform(0.1, 3.0))
                mu = float(rng.uniform(-0.15, 0.15))
                sig = float(rng.uniform(0.02, 0.10))
                params = np.append(params, [amp, mu, sig])
            
            # Use Nelder-Mead
            try:
                result = opt.minimize(
                    nll,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-6}
                )
                if result.fun < best_val:
                    h_opt = model_to_h(result.x)
                    update_best(h_opt, result.fun)
            except Exception:
                pass
    
    # === Phase 3: Spatial-domain coordinate descent refinement ===
    if time.time() < deadline - 20.0:
        rng2 = np.random.default_rng(777)
        cur_h = best_h.copy()
        cur_val = best_val
        
        def quick_ratio(h):
            f = h[CELL]
            F = np.fft.rfft(f, N_FFT)
            conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
            l2 = np.sum(f * f) * DX
            if l2 < 1e-15:
                return float('inf')
            return float(np.max(conv) / l2)
        
        # Adaptive coordinate descent with multiple step sizes
        step_scales = [1.0, 0.5, 0.25, 0.1]
        n_iter = 8000
        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Adaptive step scale based on iteration
            scale_idx = min(it // 2000, len(step_scales) - 1)
            scale = step_scales[scale_idx]
            
            # Pick a random subset of coordinates to update
            n_coords = int(rng2.integers(1, 8))
            idxs = rng2.choice(N, size=n_coords, replace=False)
            
            h_try = cur_h.copy()
            for idx in idxs:
                # Try small multiplicative change with adaptive range
                factor_range = (1.0 - 0.15 * scale, 1.0 + 0.15 * scale)
                factor = float(rng2.uniform(*factor_range))
                h_try[idx] = max(0.0, cur_h[idx] * factor)
            
            v = quick_ratio(h_try)
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)
    
    # === Phase 4: Try hand-crafted shapes ===
    if time.time() < deadline - 5.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        crafted = []
        crafted.append(np.exp(-2.0 * x_norm**2))
        crafted.append(np.exp(-0.5 * x_norm**2))
        crafted.append(0.5 + 0.5 * np.exp(-8.0 * x_norm**2))
        crafted.append(np.exp(-20.0 * (x_norm - 0.3)**2) + np.exp(-20.0 * (x_norm + 0.3)**2))
        crafted.append(0.1 + np.exp(-30.0 * (x_norm - 0.4)**2) + np.exp(-30.0 * (x_norm + 0.4)**2))
        crafted.append(1.0 - 0.3 * np.exp(-5.0 * x_norm**2))
        # U-shaped (low in middle)
        crafted.append(0.5 + 0.5 * x_norm**2)
        # Cosine window
        crafted.append(0.5 + 0.5 * np.cos(np.pi * x_norm))
        
        for h in crafted:
            if time.time() > deadline - 3.0:
                break
            h = np.clip(h, 0, None)
            if h.sum() < 1e-12:
                continue
            v = ratio_from_h(h)
            update_best(h, v)

    return list(best_h)
