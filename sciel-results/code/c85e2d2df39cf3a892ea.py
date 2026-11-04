import numpy as np
import time
import scipy.optimize as opt
from scipy.special import logsumexp

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 6000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Precompute FFT of the cell mapping indicator for fast convolution via FFT
# We can compute conv = ifft(fft(f) * fft(f)) where f = h[CELL]
# But h[CELL] is a many-to-one mapping. Let's stick to the current fast method.

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
    """Fast ratio computation."""
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    rng = np.random.default_rng(12345)

    # === Phase 1: Low-dimensional smooth parametric models (Gaussians) ===
    def smooth_model_to_h(params):
        base = params[0]
        h = np.full(N, base)
        for k in range(1, len(params), 3):
            amp = params[k]
            mu = params[k+1]
            sig = max(params[k+2], 1e-4)
            h += amp * np.exp(-0.5 * ((X - mu) / sig) ** 2)
        return np.clip(h, 0.0, None)
    
    def smooth_ratio(params):
        h = smooth_model_to_h(params)
        return ratio_from_h(h)
    
    for n_comp in [1, 2, 3]:
        n_starts = 20
        for start in range(n_starts):
            if time.time() > deadline - 50.0:
                break
            
            base = float(rng.uniform(0.0, 0.2))
            params = np.array([base])
            for _ in range(n_comp):
                amp = float(rng.uniform(0.5, 4.0))
                mu = float(rng.uniform(-0.2, 0.2))
                sig = float(rng.uniform(0.03, 0.12))
                params = np.append(params, [amp, mu, sig])
            
            try:
                result = opt.minimize(
                    smooth_ratio,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
                )
                if result.fun < best_val:
                    h_opt = smooth_model_to_h(result.x)
                    update_best(h_opt, result.fun)
            except Exception:
                pass

    # === Phase 2: Cosine-based smooth models ===
    def cosine_model_to_h(coeffs):
        h = np.zeros(N)
        for k in range(len(coeffs)):
            if k == 0:
                h += coeffs[0]
            else:
                h += coeffs[k] * np.cos(2 * np.pi * k * X / 0.5)
        return np.clip(h, 0.0, None)
    
    def cosine_ratio(coeffs):
        h = cosine_model_to_h(coeffs)
        return ratio_from_h(h)
    
    n_cos_starts = 25
    max_k = 8
    
    for start in range(n_cos_starts):
        if time.time() > deadline - 35.0:
            break
        
        coeffs = np.zeros(max_k + 1)
        coeffs[0] = float(rng.uniform(0.5, 2.0))
        for k in range(1, max_k + 1):
            coeffs[k] = float(rng.uniform(-1.0, 1.0))
        
        try:
            result = opt.minimize(
                cosine_ratio,
                coeffs,
                method='Nelder-Mead',
                options={'maxiter': 250, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
            )
            if result.fun < best_val:
                h_opt = cosine_model_to_h(result.x)
                update_best(h_opt, result.fun)
        except Exception:
                pass

    # === Phase 2b: Non-negative finite Fourier sum (squared magnitude) ===
    L = 0.5
    def fourier_sq_model_to_h(params):
        K = (len(params) - 1) // 2
        c_re = params[1::2]
        c_im = params[2::2]
        c = c_re + 1j * c_im
        
        k_vals = np.arange(K + 1)
        phase = np.exp(1j * 2 * np.pi * np.outer(X, k_vals) / L)
        s_vals = phase @ c
        h = params[0] * np.real(s_vals * np.conj(s_vals))
        return np.clip(h, 0.0, None)
    
    def fourier_sq_ratio(params):
        h = fourier_sq_model_to_h(params)
        return ratio_from_h(h)
    
    n_fourier_starts = 15
    max_K = 6
    
    for start in range(n_fourier_starts):
        if time.time() > deadline - 25.0:
            break
        
        K = int(rng.integers(2, max_K + 1))
        dim = 1 + 2 * (K + 1)
        params = np.zeros(dim)
        params[0] = float(rng.uniform(0.1, 1.0))
        for k in range(K + 1):
            params[1 + 2*k] = float(rng.uniform(-1.0, 1.0))
            params[2 + 2*k] = float(rng.uniform(-1.0, 1.0))
        
        try:
            result = opt.minimize(
                fourier_sq_ratio,
                params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
            )
            if result.fun < best_val:
                h_opt = fourier_sq_model_to_h(result.x)
                update_best(h_opt, result.fun)
        except Exception:
            pass
    
    # === Phase 3: Differential Evolution for moderate-dimensional smooth models ===
    def de_smooth_ratio(params):
        h = smooth_model_to_h(params)
        val = ratio_from_h(h)
        if not np.isfinite(val):
            return 1e10
        return val
    
    for n_comp in [2, 3]:
        if time.time() > deadline - 20.0:
            break
        dim = 1 + 3 * n_comp
        bounds = [(0.0, 0.3)] + [(-4.0, 4.0), (-0.25, 0.25), (0.02, 0.15)] * n_comp
        
        x0 = np.array([0.1])
        for _ in range(n_comp):
            x0 = np.append(x0, [2.0, 0.0, 0.05])
        
        try:
            result = opt.differential_evolution(
                de_smooth_ratio,
                bounds=bounds,
                x0=x0,
                seed=42,
                maxiter=50,
                tol=1e-6,
                polish=True,
                update_interval=1
            )
            if result.fun < best_val:
                h_opt = smooth_model_to_h(result.x)
                update_best(h_opt, result.fun)
        except Exception:
            pass
    
    # === Phase 4: Differentiable soft-max optimization on full 600-dim space ===
    def soft_max_ratio(h, beta=1.0):
        h = np.clip(h, 1e-10, None)
        f = h[CELL]
        F = np.fft.rfft(f, N_FFT)
        conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
        l2 = np.sum(f * f) * DX
        if l2 < 1e-15:
            return 1e10
        max_conv = np.max(conv)
        soft_max_val = max_conv + beta * np.log(np.sum(np.exp((conv - max_conv) / beta)))
        return float(soft_max_val / l2)
    
    def soft_max_grad(h, beta=1.0):
        h = np.clip(h, 1e-10, None)
        f = h[CELL]
        F = np.fft.rfft(f, N_FFT)
        conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
        l2 = np.sum(f * f) * DX
        if l2 < 1e-15:
            return np.zeros(N)
        
        max_conv = np.max(conv)
        weights = np.exp((conv - max_conv) / beta)
        weights /= np.sum(weights)
        
        W_fft = np.fft.rfft(weights, N_FFT)
        cross_corr = np.fft.irfft(W_fft * np.conj(F), N_FFT)[:FINE_N] * DX
        
        soft_max_val = max_conv + beta * np.log(np.sum(weights))
        ratio_val = soft_max_val / l2
        
        grad_f = (1.0 / (beta * l2)) * cross_corr - (ratio_val / l2) * 2.0 * f * DX
        
        grad_h = np.zeros(N)
        np.add.at(grad_h, CELL, grad_f)
        
        return grad_h
    
    if time.time() < deadline - 15.0:
        beta = 0.01
        cur_h = best_h.copy()
        cur_val = best_val
        
        def soft_obj_and_grad(h):
            h = np.clip(h, 1e-10, None)
            val = soft_max_ratio(h, beta)
            if not np.isfinite(val):
                return 1e10, np.zeros(N)
            grad = soft_max_grad(h, beta)
            return val, grad
        
        try:
            result = opt.minimize(
                lambda h: soft_obj_and_grad(h)[0],
                cur_h,
                jac=lambda h: soft_obj_and_grad(h)[1],
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * N,
                options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-8}
            )
            if result.fun < best_val:
                h_opt = np.clip(result.x, 0, None)
                v = ratio_from_h(h_opt)
                update_best(h_opt, v)
                cur_h = h_opt.copy()
                cur_val = v
        except Exception:
            pass
        
        if time.time() < deadline - 10.0:
            for beta_try in [0.05, 0.1, 0.2]:
                if time.time() > deadline - 8.0:
                    break
                try:
                    def soft_obj_beta(h):
                        return soft_max_ratio(h, beta_try)
                    
                    def soft_grad_beta(h):
                        return soft_max_grad(h, beta_try)
                    
                    result = opt.minimize(
                        soft_obj_beta,
                        cur_h,
                        jac=soft_grad_beta,
                        method='L-BFGS-B',
                        bounds=[(1e-10, None)] * N,
                        options={'maxiter': 80, 'gtol': 1e-5, 'ftol': 1e-7}
                    )
                    if result.fun < best_val:
                        h_opt = np.clip(result.x, 0, None)
                        v = ratio_from_h(h_opt)
                        update_best(h_opt, v)
                        cur_h = h_opt.copy()
                        cur_val = v
                except Exception:
                    pass
    
    # === Phase 5: Fine-grained coordinate refinement on best found ===
    if time.time() < deadline - 10.0:
        cur_h = best_h.copy()
        cur_val = best_val
        rng2 = np.random.default_rng(999)
        
        n_iter = 20000
        for it in range(n_iter):
            if time.time() > deadline - 3.0:
                break
            
            progress = min(it / n_iter, 1.0)
            step_frac = 0.1 * (1.0 - 0.7 * progress) + 0.02
            
            n_coords = int(rng2.integers(1, 8))
            idxs = rng2.choice(N, size=n_coords, replace=False)
            
            h_try = cur_h.copy()
            for idx in idxs:
                factor = float(rng2.uniform(1.0 - step_frac, 1.0 + step_frac))
                h_try[idx] = max(0.0, cur_h[idx] * factor)
            
            v = quick_ratio(h_try)
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)
    
    # === Phase 6: Hand-crafted smooth shapes ===
    if time.time() < deadline - 3.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        crafted = [
            np.exp(-3.0 * x_norm**2),
            np.exp(-1.5 * x_norm**2),
            0.3 + 0.7 * np.exp(-5.0 * x_norm**2),
            np.exp(-15.0 * (x_norm - 0.35)**2) + np.exp(-15.0 * (x_norm + 0.35)**2),
            0.1 + np.exp(-20.0 * (x_norm - 0.45)**2) + np.exp(-20.0 * (x_norm + 0.45)**2),
            1.0 - 0.4 * np.exp(-6.0 * x_norm**2),
            0.5 + 0.5 * np.cos(np.pi * x_norm),
            0.5 + 0.3 * np.cos(2 * np.pi * x_norm) + 0.2 * np.cos(4 * np.pi * x_norm),
            np.exp(-4.0 * x_norm**2) + 0.3 * np.exp(-12.0 * (x_norm - 0.5)**2),
            np.exp(-4.0 * x_norm**2) + 0.3 * np.exp(-12.0 * (x_norm + 0.5)**2),
            1.0 / (1.0 + 10.0 * x_norm**2),
            np.sinc(3.0 * x_norm),
        ]
        
        for h in crafted:
            if time.time() > deadline - 2.0:
                break
            h = np.clip(h, 0, None)
            if h.sum() < 1e-12:
                continue
            v = ratio_from_h(h)
            update_best(h, v)
    
    return list(best_h)
