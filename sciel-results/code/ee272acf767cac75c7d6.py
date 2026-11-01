"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_h(n, x):
    """Compute Hermite polynomial H_n(x) for n = 0, 4, 8, 12."""
    if n == 0:
        return np.ones_like(x)
    elif n == 4:
        return x**4 - 6*x**2 + 3
    elif n == 8:
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    elif n == 12:
        return x**12 - 60*x**10 + 1260*x**8 - 12600*x**6 + 45360*x**4 - 55440*x**2 + 10395
    else:
        raise ValueError(f"Unsupported n={n}")


def _compute_a12(a0, a4, a8):
    """Compute a12 such that g(0) = 0."""
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    return -(a0 + 3*a4 + 105*a8) / 10395.0


def _uncertainty_bound_fast(a0, a4, a8):
    """Compute the upper bound on A(f)*A(Fourier(f)) efficiently."""
    a12 = _compute_a12(a0, a4, a8)
    
    # Use a coarser grid for speed during optimization
    x = np.linspace(-8, 8, 4000)
    dx = x[1] - x[0]
    
    # Compute g(x)
    H0 = _hermite_h(0, x)
    H4 = _hermite_h(4, x)
    H8 = _hermite_h(8, x)
    H12 = _hermite_h(12, x)
    
    poly = a0*H0 + a4*H4 + a8*H8 + a12*H12
    g = poly * np.exp(-x**2/2)
    
    # Compute integral |x| |g(x)| dx
    abs_x_g = np.abs(x) * np.abs(g)
    integral_x = np.trapz(abs_x_g, x)
    
    # Compute G(xi) using FFT
    N = len(x)
    G_fft = np.fft.fftshift(np.fft.fft(g)) * dx
    xi_fft = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    # For the uncertainty product, we need integral |xi| |G(xi)| dxi
    # Use the FFT grid directly
    abs_xi_G = np.abs(xi_fft) * np.abs(G_fft)
    integral_xi = np.trapz(abs_xi_G, xi_fft)
    
    # The uncertainty product bound
    bound = integral_x * integral_xi
    
    return bound


def construct():
    """Find optimal coefficients for the Hermite combination."""
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    # Initial guess from the seed program
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    best_bound = _uncertainty_bound_fast(*best_coeffs)
    
    # Try multiple starting points
    start_points = [
        best_coeffs,
        np.array([-1.0, 0.01, 0.0001]),
        np.array([-0.5, 0.05, 0.001]),
        np.array([-1.5, 0.02, 0.0005]),
        np.array([-0.8, 0.03, 0.0008]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 15:  # Stop if close to deadline
            break
        
        try:
            result = minimize(
                lambda c: _uncertainty_bound_fast(*c),
                sp,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.success and result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
        except Exception:
            continue
    
    # Try a final polish with tighter tolerances
    if time.time() < deadline - 5:
        try:
            result = minimize(
                lambda c: _uncertainty_bound_fast(*c),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result.success and result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
