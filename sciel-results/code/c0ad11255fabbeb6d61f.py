import numpy as np
import time
from scipy.optimize import minimize
import scipy.fft as fft

N = 400

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT."""
    n = 2 * N - 1
    H = fft.rfft(h, n=n)
    c = fft.irfft(H * H, n=n)
    return c

def _soft_max(x, beta=100.0):
    """Differentiable approximation to max."""
    m = np.max(x)
    x_shifted = x - m
    sum_exp = np.sum(np.exp(beta * x_shifted))
    if sum_exp == 0:
        return m
    return m + (1.0 / beta) * np.log(sum_exp)

def _evaluate_exact(h):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _objective(params, beta):
    h = params
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    sm = _soft_max(c, beta)
    val = 2.0 * N * sm / (s * s)
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_val = 2.0
    
    def try_h(h):
        nonlocal best_h, best_val
        s = np.sum(h)
        if abs(s) < 1e-10:
            return
        c = _autoconv_fft(h)
        val = 2.0 * N * np.max(c) / (s * s)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        return val

    # Phase 1: Initial Grid Search with FFT
    # Try DC + 1 sinusoidal mode
    dc_values = [1.0, 2.0, 5.0, 10.0]
    freqs = [1, 2, 3, 5, 7, 10, 15, 20]
    amp_fracs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    
    t = np.arange(N)
    
    for dc in dc_values:
        for freq in freqs:
            for amp_frac in amp_fracs:
                if time.time() > deadline - 30.0:
                    break
                amp = dc * amp_frac
                # Antisymmetric mode
                a = amp * np.sin(2 * np.pi * freq * (t - (N - 1) / 2.0) / N)
                a = (a - a[::-1]) / 2.0
                h = dc * np.ones(N) + a
                try_h(h)
                
                # Symmetric mode
                a_sym = amp * np.cos(2 * np.pi * freq * (t - (N - 1) / 2.0) / N)
                a_sym = (a_sym + a_sym[::-1]) / 2.0
                h_sym = dc * np.ones(N) + a_sym
                try_h(h_sym)
    
    # Phase 1b: Random initial points
    for _ in range(50):
        if time.time() > deadline - 30.0:
            break
        h_rand = rng.standard_normal(N)
        # Ensure sum is not too small
        s = np.sum(h_rand)
        if abs(s) < 1e-5:
            continue
        try_h(h_rand)
        
        # Also try with DC offset
        for dc_off in [0.5, 1.0, 2.0]:
            h_rand_dc = dc_off * np.ones(N) + 0.5 * h_rand
            try_h(h_rand_dc)

    # Phase 2: Local Optimization with L-BFGS
    # Use soft-max objective for differentiability
    # We use a two-stage approach: first with low beta, then high beta
    
    h0 = best_h.copy()
    s0 = np.sum(h0)
    if abs(s0) > 1e-10:
        h0 = h0 / s0
        
    try:
        # Stage 1: Low beta for broader exploration
        for restart in range(8):
            if time.time() > deadline - 45.0:
                break
            
            if restart == 0:
                x0 = h0.flatten()
            else:
                noise_scale = 0.1 * np.std(h0) * (1 + 0.5 * restart)
                x0 = h0.flatten() + rng.standard_normal(N) * noise_scale
            
            try:
                result = minimize(
                    lambda x: _objective(x, beta=30.0),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                h_opt = result.x
                val_opt = _evaluate_exact(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

        # Stage 2: Medium beta
        h0 = best_h.copy()
        s0 = np.sum(h0)
        if abs(s0) > 1e-10:
            h0 = h0 / s0
            
        for restart in range(5):
            if time.time() > deadline - 30.0:
                break
            
            if restart == 0:
                x0 = h0.flatten()
            else:
                noise_scale = 0.05 * np.std(h0) * (1 + 0.5 * restart)
                x0 = h0.flatten() + rng.standard_normal(N) * noise_scale
            
            try:
                result = minimize(
                    lambda x: _objective(x, beta=100.0),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_opt = result.x
                val_opt = _evaluate_exact(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

        # Stage 3: High beta for precise refinement
        h0 = best_h.copy()
        s0 = np.sum(h0)
        if abs(s0) > 1e-10:
            h0 = h0 / s0
            
        for restart in range(5):
            if time.time() > deadline - 20.0:
                break
            
            if restart == 0:
                x0 = h0.flatten()
            else:
                noise_scale = 0.02 * np.std(h0) * (1 + 0.5 * restart)
                x0 = h0.flatten() + rng.standard_normal(N) * noise_scale
            
            try:
                result = minimize(
                    lambda x: _objective(x, beta=500.0),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                h_opt = result.x
                val_opt = _evaluate_exact(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    except Exception:
        pass

    # Phase 3: Final Polish with exact max
    # Do Nelder-Mead on the exact objective
    
    try:
        from scipy.optimize import minimize as sp_minimize
        nm_result = sp_minimize(
            lambda x: _evaluate_exact(x),
            best_h.copy(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if nm_result.fun < best_val:
            best_val = nm_result.fun
            best_h = nm_result.x.copy()
    except Exception:
        pass
    
    # Phase 4: Coordinate descent polish
    for _ in range(2000):
        if time.time() > deadline - 3.0:
            break
        
        # Try perturbing a few random coordinates
        idx = rng.integers(0, N, size=3)
        h_new = best_h.copy()
        for i in idx:
            h_new[i] += rng.standard_normal() * 0.005 * np.std(best_h)
        val_new = _evaluate_exact(h_new)
        if val_new < best_val:
            best_val = val_new
            best_h = h_new

    return best_h
