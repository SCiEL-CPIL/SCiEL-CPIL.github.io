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


def _uncertainty_bound(a0, a4, a8):
    """Compute the upper bound on A(f)*A(Fourier(f))."""
    a12 = _compute_a12(a0, a4, a8)
    
    # Use a grid for integration
    # The functions decay like exp(-x^2/2), so [-10, 10] is sufficient.
    # 2000 points is usually enough for smooth functions with trapezoidal rule.
    x = np.linspace(-10, 10, 2000)
    
    # Compute g(x)
    H0 = _hermite_h(0, x)
    H4 = _hermite_h(4, x)
    H8 = _hermite_h(8, x)
    H12 = _hermite_h(12, x)
    
    poly = a0*H0 + a4*H4 + a8*H8 + a12*H12
    g = poly * np.exp(-x**2/2)
    
    # Compute integral |x| |g(x)| dx
    abs_x_g = np.abs(x) * np.abs(g)
    integral_x = np.trapezoid(abs_x_g, x)
    
    # Compute Fourier transform analytically
    # g(x) = sum a_n H_n(x) exp(-x^2/2)
    # F[g](xi) = sum a_n (-i)^n H_n(xi) exp(-xi^2/2)
    # Note: This assumes the unitary FT convention or consistent scaling.
    # Let's verify scaling.
    # If g(x) = exp(-x^2/2), F[g](xi) = sqrt(2*pi) exp(-xi^2/2).
    # Using H0=1, a0=1, a4=a8=a12=0.
    # Formula gives: 1 * (-i)^0 * 1 * exp(-xi^2/2) = exp(-xi^2/2).
    # Missing factor of sqrt(2*pi).
    # So F[g](xi) = sqrt(2*pi) * sum a_n (-i)^n H_n(xi) exp(-xi^2/2)
    
    # For the uncertainty product, constant factors in FT might matter or cancel depending on definition.
    # The task asks for "upper bound on A(f)*A(Fourier(f))".
    # Usually A(f) = int |f| or similar.
    # Let's assume standard non-unitary FT: F(omega) = int f(x) exp(-i x omega) dx.
    
    xi = x  # Symmetric grid
    
    H0_xi = _hermite_h(0, xi)
    H4_xi = _hermite_h(4, xi)
    H8_xi = _hermite_h(8, xi)
    H12_xi = _hermite_h(12, xi)
    
    # (-i)^0 = 1, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1
    # All powers of 4 are 1. So no phase factors for even n divisible by 4.
    # Wait, (-i)^4 = 1. (-i)^8 = 1. (-i)^12 = 1.
    # So the coefficients are just a0, a4, a8, a12.
    
    G_poly = a0*H0_xi + a4*H4_xi + a8*H8_xi + a12*H12_xi
    G = np.sqrt(2 * np.pi) * G_poly * np.exp(-xi**2/2)
    
    # Compute integral |xi| |G(xi)| dxi
    abs_xi_G = np.abs(xi) * np.abs(G)
    integral_xi = np.trapezoid(abs_xi_G, xi)
    
    # The uncertainty product bound
    bound = integral_x * integral_xi
    
    return bound


def construct():
    """Find optimal coefficients for the Hermite combination."""
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin for 180s limit
    
    # Initial guess from the seed program
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    best_bound = _uncertainty_bound(*best_coeffs)
    
    # Try multiple starting points and local optimizations
    start_points = [
        best_coeffs,
        np.array([-1.0, 0.01, 0.0001]),
        np.array([-0.5, 0.05, 0.001]),
        np.array([-1.5, 0.02, 0.0005]),
        np.array([-0.8, 0.03, 0.0008]),
        np.array([-1.2, 0.015, 0.0003]),
        np.array([-0.9, 0.04, 0.0012]),
        np.array([-1.1, 0.025, 0.0007]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 10:  # Stop if close to deadline
            break
        
        try:
            result = minimize(
                lambda c: _uncertainty_bound(*c),
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
                lambda c: _uncertainty_bound(*c),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result.success and result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
