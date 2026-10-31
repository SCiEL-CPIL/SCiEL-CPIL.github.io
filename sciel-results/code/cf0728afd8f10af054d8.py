import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

_DEADLINE = time.time() + 200  # 200s deadline, 10s margin before 210s kill

def _hermite_polys(x):
    """Compute H0, H4, H8, H12 (physicists' Hermite polynomials)."""
    H0 = np.ones_like(x)
    H2 = x**2 - 1
    H4 = x**4 - 6*x**2 + 3
    H6 = x**6 - 15*x**4 + 45*x**2 - 15
    H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    H10 = x**10 - 45*x**8 + 630*x**6 - 3150*x**4 + 4725*x**2 - 945
    H12 = x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945
    return H0, H4, H8, H12

def _compute_uncertainty(a0, a4, a8):
    """Compute A(f)*A(Fourier(f)) for given coefficients."""
    if time.time() > _DEADLINE:
        return None
    
    # Hermite values at 0
    H0_0, H4_0, H8_0, H12_0 = 1.0, 3.0, 105.0, 945.0
    
    # Constraint: g(0) = 0 => a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    def g(x):
        H0, H4, H8, H12 = _hermite_polys(x)
        return (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * np.exp(-x**2 / 2)
    
    # Compute |g(x)|^2 for normalization and moments
    def g_sq(x):
        H0, H4, H8, H12 = _hermite_polys(x)
        val = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * np.exp(-x**2 / 2)
        return val**2
    
    # Normalize: <1> = integral of |g|^2
    norm, _ = quad(g_sq, -20, 20, limit=200)
    if norm <= 0 or not np.isfinite(norm):
        return None
    
    # <x> = integral x*|g|^2 / norm
    def x_g_sq(x):
        return x * g_sq(x)
    x_mean, _ = quad(x_g_sq, -20, 20, limit=200)
    x_mean /= norm
    
    # <x^2> = integral x^2*|g|^2 / norm
    def x2_g_sq(x):
        return x**2 * g_sq(x)
    x2_mean, _ = quad(x2_g_sq, -20, 20, limit=200)
    x2_mean /= norm
    
    A_f_sq = x2_mean - x_mean**2
    if A_f_sq <= 0:
        return None
    
    # For Fourier transform: F(ξ) of g(x) = p(x)exp(-x²/2)
    # The Fourier transform of x^n * exp(-x²/2) can be computed, but it's complex.
    # Alternative: use the fact that for the Fourier transform, the moments relate to derivatives.
    # Actually, let's use the property that if g(x) = sum c_n H_n(x) exp(-x²/2), then
    # the Fourier transform involves the same Hermite structure with scaled coefficients.
    # 
    # For physicists' Hermite: F[H_n(x) exp(-x²/2)] = (-i)^n H_n(ξ) exp(-ξ²/2) * sqrt(2π)
    # Wait, this is for the specific normalization. Let me use numerical Fourier transform instead.
    
    # Numerical approach: compute Fourier transform on a grid
    x_grid = np.linspace(-20, 20, 4096)
    g_vals = g(x_grid)
    dx = x_grid[1] - x_grid[0]
    
    # FFT
    G = np.fft.fftshift(np.fft.fft(g_vals * dx)) * dx
    xi_grid = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(len(x_grid), d=dx))
    
    G_sq = np.abs(G)**2
    G_norm = np.sum(G_sq) * dx
    if G_norm <= 0 or not np.isfinite(G_norm):
        return None
    
    xi_mean = np.sum(xi_grid * G_sq) * dx / G_norm
    xi2_mean = np.sum(xi_grid**2 * G_sq) * dx / G_norm
    
    A_F_sq = xi2_mean - xi_mean**2
    if A_F_sq <= 0:
        return None
    
    return np.sqrt(A_f_sq * A_F_sq)

def construct():
    # Seed coefficients
    seed = np.array([-113/100, 1/25, 1/3240])
    best_val = _compute_uncertainty(*seed)
    if best_val is None:
        return seed
    best_coeffs = seed.copy()
    
    # Try to optimize
    def objective(params):
        val = _compute_uncertainty(*params)
        if val is None:
            return 1e10
        return val
    
    # Try several random restarts
    rng = np.random.default_rng(42)
    start_time = time.time()
    max_time = 180  # leave 20s margin
    
    for restart in range(50):
        if time.time() - start_time > max_time - 10:
            break
        
        # Random perturbation around best
        scale = 0.5 / (restart + 1)
        x0 = best_coeffs + rng.normal(0, scale, 3)
        
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x.copy()
    
    return best_coeffs
