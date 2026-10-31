"""Optimized program for uncertainty inequality Hermite test function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so
from scipy.special import eval_hermite

DEADLINE = 175.0  # safety margin under 210s kill

def _P(x, a0, a4, a8, a12):
    return (a0 * eval_hermite(0, x) +
            a4 * eval_hermite(4, x) +
            a8 * eval_hermite(8, x) +
            a12 * eval_hermite(12, x))

def _g(x, a0, a4, a8, a12):
    return np.exp(-x**2 / 2.0) * _P(x, a0, a4, a8, a12)

def _compute_bound(coeffs):
    a0, a4, a8 = coeffs
    H0_0 = eval_hermite(0, 0.0)
    H4_0 = eval_hermite(4, 0.0)
    H8_0 = eval_hermite(8, 0.0)
    H12_0 = eval_hermite(12, 0.0)
    
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    x = np.linspace(-15, 15, 200001)
    dx = x[1] - x[0]
    
    g = _g(x, a0, a4, a8, a12)
    g2 = g**2
    g4 = g**4
    
    L2_sq = np.trapz(g2, x)
    if L2_sq <= 1e-15:
        return 1e10
    
    L4_4 = np.trapz(g4, x)
    
    # Fourier transform using FFT
    N = 262144  # power of 2 for FFT efficiency
    x_f = np.linspace(-15, 15, N)
    dx_f = x_f[1] - x_f[0]
    g_f = _g(x_f, a0, a4, a8, a12)
    
    G = np.fft.fftshift(np.fft.fft(g_f)) * dx_f
    xi = 2 * np.pi * np.fft.fftfreq(N, d=dx_f)
    xi = np.fft.fftshift(xi)
    
    G2 = np.abs(G)**2
    G4 = np.abs(G)**4
    
    L2_G = np.trapz(G2, xi)
    L4_G = np.trapz(G4, xi)
    
    if L2_G <= 1e-15 or L4_G <= 1e-15:
        return 1e10
    
    ratio_g = L4_4 / (L2_sq**2)
    ratio_G = L4_G / (L2_G**2)
    
    C4 = np.sqrt(ratio_g * ratio_G)
    
    return C4

def construct():
    start_time = time.time()
    
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_coeffs = x0.copy()
    best_bound = _compute_bound(x0)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 40:
        return best_coeffs
    
    # Nelder-Mead
    try:
        result = so.minimize(_compute_bound, x0, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.fun < best_bound:
            best_bound = result.fun
            best_coeffs = result.x.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 40:
        return best_coeffs
    
    # Powell
    try:
        result = so.minimize(_compute_bound, best_coeffs, method='Powell',
                            options={'maxiter': 1000, 'xtol': 1e-10, 'ftol': 1e-10})
        if result.fun < best_bound:
            best_bound = result.fun
            best_coeffs = result.x.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 20:
        return best_coeffs
    
    # Random restarts
    rng = np.random.default_rng(123)
    for _ in range(30):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
        x_trial = best_coeffs + rng.normal(0, 0.2, size=3)
        try:
            b = _compute_bound(x_trial)
            if b < best_bound:
                best_bound = b
                best_coeffs = x_trial.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
