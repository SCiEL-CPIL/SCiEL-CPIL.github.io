"""Optimize Hermite-Gaussian test function coefficients for uncertainty inequality."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_herm
from scipy.optimize import minimize, differential_evolution

def _eval_g(x, c0, c4, c8):
    """Evaluate g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)"""
    H0 = np.ones_like(x)
    H4 = eval_herm(4, x)
    H8 = eval_herm(8, x)
    H12 = eval_herm(12, x)
    
    # H_n(0) for physicist's Hermite: H_0(0)=1, H_4(0)=-12, H_8(0)=1008, H_12(0)=-103968
    H0_0, H4_0, H8_0, H12_0 = 1.0, -12.0, 1008.0, -103968.0
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    return (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * np.exp(-0.5 * x**2)

def _uncertainty_product(c0, c4, c8):
    """Compute A(x) * A(k) via numerical integration and FFT."""
    N = 4096
    x = np.linspace(-8, 8, N)
    dx = x[1] - x[0]
    
    g = _eval_g(x, c0, c4, c8)
    
    norm_sq = np.sum(g**2) * dx
    if norm_sq < 1e-20:
        return 1e10
    
    # x-space moments
    x2 = x**2
    x_mean = np.sum(x * g**2) * dx / norm_sq
    x_var = np.sum((x - x_mean)**2 * g**2) * dx / norm_sq
    A_x = np.sqrt(max(x_var, 1e-30))
    
    # Fourier transform
    # Use FFT: F[k] = sum g[x] * exp(-2*pi*i*k*x) * dx
    g_hat = np.fft.fftshift(np.fft.fft(g * dx))
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    norm_k = np.sum(g_hat**2) * (2*np.pi/N)  # Parseval with dx
    # Actually for FFT: sum|g_hat|^2 * dk = sum|g|^2 * dx, so dk = 2*pi/N for this setup
    # Let me just compute variance directly
    k_mean = np.sum(k * g_hat**2) / np.sum(g_hat**2)
    k_var = np.sum((k - k_mean)**2 * g_hat**2) / np.sum(g_hat**2)
    A_k = np.sqrt(max(k_var, 1e-30))
    
    return A_x * A_k

def _objective(coeffs):
    try:
        val = _uncertainty_product(coeffs[0], coeffs[1], coeffs[2])
        return val if np.isfinite(val) else 1e10
    except:
        return 1e10

def construct():
    start_time = time.time()
    deadline = 170.0  # safety margin
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best = seed.copy()
    best_val = _objective(best)
    
    # Local refinement around seed
    try:
        res = minimize(_objective, seed, method='Nelder-Mead', 
                     options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best = res.x.copy()
    except:
        pass
    
    # Check time
    if time.time() - start_time > deadline - 20:
        return best
    
    # Try differential evolution in a bounded region
    try:
        bounds = [(-5, 5), (-1, 1), (-0.1, 0.1)]
        de_res = differential_evolution(_objective, bounds, seed=42, maxiter=50, 
                                        popsize=15, tol=1e-10, polish=True)
        if de_res.fun < best_val:
            best_val = de_res.fun
            best = de_res.x.copy()
    except:
        pass
    
    # Final polish
    if time.time() - start_time < deadline - 5:
        try:
            res2 = minimize(_objective, best, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-15})
            if res2.fun < best_val:
                best_val = res2.fun
                best = res2.x.copy()
        except:
            pass
    
    return best
# EVOLVE-BLOCK-END
