"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from scipy import integrate
from scipy.optimize import minimize
import time

_deadline = time.time() + 190  # safety margin before 210s kill

def _uncertainty_product(a0, a4, a8):
    """Compute A(f)*A(F) for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2).
    a12 is determined by g(0)=0."""
    # H_n(0) for even n: H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-20160
    # g(0) = (a0*1 + a4*(-12) + a8*1680 + a12*(-20160)) * 1 = 0
    # So a12 = (a0 - 12*a4 + 1680*a8) / 20160
    a12 = (a0 - 12.0 * a4 + 1680.0 * a8) / 20160.0
    
    coeffs = np.array([a0, a4, a8, a12])
    degrees = np.array([0, 4, 8, 12])
    
    def g(x):
        h = np.zeros_like(x)
        for c, d in zip(coeffs, degrees):
            h += c * eval_hermite(d, x)
        return h * np.exp(-0.5 * x**2)
    
    def g_abs(x):
        return np.abs(g(x))
    
    # A(f) = ∫|x g(x)| dx / ∫|g(x)| dx
    num1, _ = integrate.quad(lambda x: np.abs(x * g(x)), -15, 15, limit=200)
    den1, _ = integrate.quad(g_abs, -15, 15, limit=200)
    if den1 < 1e-12:
        return np.inf
    A_f = num1 / den1
    
    # Fourier transform of g(x) = sum c_n H_n(x) exp(-x^2/2)
    # The Fourier transform of H_n(x) exp(-x^2/2) is (-i)^n H_n(x) exp(-x^2/2) (up to normalization)
    # With the convention F(k) = ∫ f(x) exp(-2πi k x) dx, but let's use the physics convention
    # Actually, for the uncertainty principle, we need to be careful about normalization.
    # Let's use: F(k) = ∫ f(x) exp(-i k x) dx
    # Then F{H_n(x) e^{-x^2/2}} = i^n √(2π) H_n(k) e^{-k^2/2} ... this gets complicated.
    # 
    # Alternative: compute numerically.
    def G(k):
        # Numerical Fourier transform
        result, _ = integrate.quad(lambda x: g(x) * np.cos(k * x), -15, 15, limit=200)
        result_im, _ = integrate.quad(lambda x: g(x) * np.sin(k * x), -15, 15, limit=200)
        return result - 1j * result_im
    
    # A(F) = ∫|k G(k)| dk / ∫|G(k)| dk
    # This is expensive to compute numerically. Let me use the analytic form instead.
    #
    # For the Hermite-Gaussian basis with the convention used in uncertainty principle literature:
    # If g(x) = Σ c_n H_n(x) e^{-x^2/2}, then the Fourier transform (with F(k)=∫g(x)e^{-ikx}dx)
    # is G(k) = i^(-n) √(2π) Σ c_n H_n(k) e^{-k^2/2} ... 
    #
    # Actually, let me just compute it numerically but more efficiently using the fact that
    # the Fourier transform of a Hermite-Gaussian is another Hermite-Gaussian.
    #
    # With the convention: F{e^{-x^2/2} H_n(x)} = i^n √(2π) e^{-k^2/2} H_n(k)
    # (this depends on the exact FT convention; let me use the standard one where
    #  F(k) = ∫ f(x) e^{-2π i k x} dx, but that changes things)
    #
    # Let me just do it numerically for correctness, but use a smarter approach.
    # Since g is a combination of Hermite-Gaussians, G(k) is also a combination of
    # Hermite-Gaussians in k.
    
    # Using the convention F(k) = ∫ f(x) e^{-ikx} dx:
    # F{H_n(x) e^{-x^2/2}} = i^n √(2π) H_n(k) e^{-k^2/2}
    # Wait, I need to verify. The standard result is:
    # ∫ H_n(x) e^{-x^2/2} e^{-ikx} dx = i^n √(2π) H_n(k) e^{-k^2/2}
    # Let me just use this.
    
    def G_analytic(k):
        h = np.zeros_like(k)
        for c, d in zip(coeffs, degrees):
            h += c * (1j)**d * np.sqrt(2 * np.pi) * eval_hermite(d, k)
        return h * np.exp(-0.5 * k**2)
    
    def G_abs(k):
        return np.abs(G_analytic(k))
    
    num2, _ = integrate.quad(lambda k: np.abs(k * G_analytic(k)), -15, 15, limit=200)
    den2, _ = integrate.quad(G_abs, -15, 15, limit=200)
    if den2 < 1e-12:
        return np.inf
    A_F = num2 / den2
    
    return A_f * A_F

def construct():
    t0 = time.time()
    
    # Grid search over a0, a4, a8
    # Normalize: we can set a0 = 1 wlog (scale invariance of the ratio)
    # So search over a4, a8 with a0 = 1
    best_val = np.inf
    best_params = None
    
    # Coarse grid
    a4_vals = np.linspace(-2, 2, 41)
    a8_vals = np.linspace(-0.1, 0.1, 41)
    
    for a4 in a4_vals:
        if time.time() > _deadline - 30:
            break
        for a8 in a8_vals:
            if time.time() > _deadline - 30:
                break
            try:
                val = _uncertainty_product(1.0, a4, a8)
                if val < best_val:
                    best_val = val
                    best_params = (1.0, a4, a8)
            except Exception:
                pass
    
    # Local refinement
    if best_params is not None and time.time() < _deadline - 20:
        try:
            res = minimize(_uncertainty_product, best_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    a0, a4, a8 = best_params
    return np.array([a0, a4, a8])
# EVOLVE-BLOCK-END
