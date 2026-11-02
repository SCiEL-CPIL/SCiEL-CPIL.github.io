"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import hermite
import time

def _hermite(n, x):
    """Physicist's Hermite polynomial H_n(x)."""
    Hn = hermite(n)
    return Hn(x)

def _build_g(c0, c4, c8):
    """Build g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2) with g(0)=0."""
    # H0(0) = 1, H4(0) = -12, H8(0) = 13440, H12(0) = -10321920
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 13440.0
    H12_0 = -10321920.0
    
    # g(0) = (c0*H0_0 + c4*H4_0 + c8*H8_0 + c12*H12_0) * exp(0) = 0
    # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    return c0, c4, c8, c12

def _compute_bound(c0, c4, c8):
    """Compute the upper bound on A(f)*A(Fourier(f)) for g(x) = P(x)*exp(-x^2/2)."""
    c0, c4, c8, c12 = _build_g(c0, c4, c8)
    
    # Numerical integration on a fine grid
    x = np.linspace(-15, 15, 200001)
    dx = x[1] - x[0]
    
    # Compute g(x)
    H0 = np.ones_like(x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = P * np.exp(-x**2 / 2)
    
    # Normalize
    norm_sq = np.sum(g**2) * dx
    if norm_sq < 1e-15:
        return 1e10  # invalid
    
    g_norm = g / np.sqrt(norm_sq)
    
    # Position spread: (∫ x^2 |g|^2 dx)^{1/2}
    x2_weighted = np.sum(x**2 * g_norm**2) * dx
    delta_x = np.sqrt(x2_weighted)
    
    # Fourier transform via FFT or numerical
    # Use the fact that FT of H_n(x)*exp(-x^2/2) is (i)^n * H_n(x) * exp(-x^2/2) (up to constants)
    # With unitary FT: F[f](ξ) = (1/√(2π)) ∫ f(x) e^{-ixξ} dx
    # For f(x) = H_n(x) e^{-x^2/2}, F[f](ξ) = (i)^n H_n(ξ) e^{-ξ^2/2} / √(2π) * √(2π) ... 
    # Actually with the right normalization, FT of H_n(x)e^{-x^2/2} = (i)^n H_n(x) e^{-x^2/2}
    
    # Let me use numerical FFT to be safe
    # Pad and use FFT
    N = 2**18
    x_fft = np.linspace(-15, 15, N, endpoint=False)
    dx_fft = x_fft[1] - x_fft[0]
    
    H0_f = np.ones_like(x_fft)
    H4_f = _hermite(4, x_fft)
    H8_f = _hermite(8, x_fft)
    H12_f = _hermite(12, x_fft)
    
    P_f = c0 * H0_f + c4 * H4_f + c8 * H8_f + c12 * H12_f
    g_f = P_f * np.exp(-x_fft**2 / 2)
    
    # Unitary FFT
    G = np.fft.fftshift(np.fft.fft(g_f)) * dx_fft / np.sqrt(2 * np.pi)
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=dx_fft))
    
    # Normalize G
    norm_G = np.sum(G**2) * dx_fft
    if norm_G < 1e-15:
        return 1e10
    
    G_norm = G / np.sqrt(norm_G)
    
    # Momentum spread
    xi2_weighted = np.sum(freq**2 * G_norm**2) * dx_fft
    delta_p = np.sqrt(xi2_weighted)
    
    return delta_x * delta_p

def construct():
    """Return three coefficients (c0, c4, c8) that minimize the upper bound."""
    start_time = time.time()
    deadline = start_time + 180  # safety margin before 210s kill
    
    # Seed values
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    # Compute seed value
    seed_value = _compute_bound(seed[0], seed[1], seed[2])
    
    # Try multiple starting points and optimize
    best_val = seed_value
    best_coeffs = seed.copy()
    
    # Generate diverse starting points
    rng = np.random.RandomState(42)
    starts = [seed.copy()]
    
    # Add perturbations around seed
    for _ in range(20):
        pert = seed + rng.randn(3) * 0.1
        starts.append(pert)
    
    # Add some random starts
    for _ in range(10):
        starts.append(rng.randn(3) * 0.5)
    
    # Also try some structured starts
    starts.append(np.array([1.0, 0.0, 0.0]))
    starts.append(np.array([0.0, 1.0, 0.0]))
    starts.append(np.array([0.0, 0.0, 1.0]))
    starts.append(np.array([1.0, 1.0, 1.0]))
    starts.append(np.array([-1.0, -1.0, -1.0]))
    
    for s in starts:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                lambda c: _compute_bound(c[0], c[1], c[2]),
                s,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_val:
                val = result.fun
                if val < best_val:
                    best_val = val
                    best_coeffs = result.x.copy()
        except Exception:
            continue
    
    # Final polish with high precision
    if time.time() < deadline - 10:
        try:
            result = minimize(
                lambda c: _compute_bound(c[0], c[1], c[2]),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
