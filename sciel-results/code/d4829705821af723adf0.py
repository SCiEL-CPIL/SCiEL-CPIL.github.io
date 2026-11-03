"""Optimized search for Hermite coefficients to minimize uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import sys

def _hermite_powers(x):
    """Compute H_0, H_4, H_8, H_12 in a stable way using recurrence."""
    H0 = np.ones_like(x)
    H1 = 2 * x
    H2 = 4 * x**2 - 2
    H3 = 8 * x**3 - 12 * x
    H4 = 16 * x**4 - 48 * x**2 + 12
    H5 = 32 * x**5 - 160 * x**3 + 120 * x
    H6 = 64 * x**6 - 480 * x**4 + 720 * x**2 - 120
    H7 = 128 * x**7 - 1344 * x**5 + 3360 * x**3 - 1680 * x
    H8 = 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    H9 = 512 * x**9 - 10752 * x**7 + 60480 * x**5 - 120960 * x**3 + 60480 * x
    H10 = 1024 * x**10 - 32256 * x**8 + 302400 * x**6 - 705600 * x**4 + 453600 * x**2 - 30240
    H11 = 2048 * x**11 - 90112 * x**9 + 1209600 * x**7 - 3528000 * x**5 + 3024000 * x**3 - 604800 * x
    H12 = 4096 * x**12 - 239616 * x**10 + 5702400 * x**8 - 24192000 * x**6 + 37440000 * x**4 - 18144000 * x**2 + 1036800
    return H0, H4, H8, H12

def _compute_bound(coeffs):
    """
    Given three free coefficients (c0, c4, c8), compute the upper bound on A(f)*A(F).
    The H12 coefficient is derived to impose g(0) = 0.
    
    We use a grid-based numerical approach to estimate the L1 norms.
    """
    c0, c4, c8 = coeffs
    
    # Derive c12 from g(0) = 0
    # g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    # At x=0: H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=1036800
    # c0 + 12*c4 + 1680*c8 + 1036800*c12 = 0
    c12 = -(c0 + 12*c4 + 1680*c8) / 1036800.0
    
    # Use a fine grid to compute the function and its Fourier transform
    # For efficiency, we use a moderate grid size
    N = 4096
    x = np.linspace(-15, 15, N)
    dx = x[1] - x[0]
    
    H0, H4, H8, H12 = _hermite_powers(x)
    
    # The test function is g(x) * exp(-x^2/2)
    gauss = np.exp(-x**2 / 2.0)
    g = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    f = g * gauss
    
    # Compute A(f) = integral |f(x)| dx
    A_f = np.sum(np.abs(f)) * dx
    
    # Compute Fourier transform using FFT
    # The Fourier transform convention matters. We use the standard physics convention:
    # F(k) = integral f(x) exp(-i k x) dx
    # For the uncertainty principle, we need the same normalization
    
    # Pad to next power of 2 for FFT efficiency
    N_fft = 1
    while N_fft < N * 2:
        N_fft *= 2
    
    f_padded = np.zeros(N_fft)
    f_padded[:N] = f
    
    # FFT
    F = np.fft.fft(f_padded) * dx
    
    # The frequency axis
    k = np.fft.fftfreq(N_fft, d=dx)
    
    # We need to be careful with the ordering. For real functions, the FFT is symmetric.
    # A(F) = integral |F(k)| dk
    # Since F is complex in general, we take the absolute value
    
    A_F = np.sum(np.abs(F)) * (2 * np.pi / N_fft)  # dk = 2*pi/N_fft
    
    return A_f * A_F

def construct():
    """
    Find three coefficients (c0, c4, c8) that minimize the upper bound.
    Returns np.array([c0, c4, c8]).
    """
    start_time = time.time()
    deadline = 180.0  # Must finish within 180s
    
    # Initial best from prior work
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    best_bound = _compute_bound(best_coeffs)
    
    # Use Differential Evolution in spherical coordinates for robustness
    # We parameterize as:
    # c0 = r * cos(theta) * cos(phi)
    # c4 = r * cos(theta) * sin(phi)  
    # c8 = r * sin(theta)
    
    def spherical_to_cartesian(params):
        r, theta, phi = params
        c0 = r * np.cos(theta) * np.cos(phi)
        c4 = r * np.cos(theta) * np.sin(phi)
        c8 = r * np.sin(theta)
        return np.array([c0, c4, c8])
    
    def objective_spherical(params):
        if time.time() - start_time > deadline - 10:
            return best_bound
        coeffs = spherical_to_cartesian(params)
        return _compute_bound(coeffs)
    
    # Bounds for spherical coordinates
    # r: scale, theta: [0, pi], phi: [0, 2*pi]
    bounds = [(0.01, 5.0), (0.0, np.pi), (0.0, 2*np.pi)]
    
    try:
        # First run DE with moderate settings
        result = differential_evolution(
            objective_spherical,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            polish=False,
            init='sobol'
        )
        
        if time.time() - start_time < deadline - 30:
            # Polish with local optimization
            best_spherical = result.x
            res_polish = minimize(
                objective_spherical,
                best_spherical,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000, 'maxfev': 5000}
            )
            
            if res_polish.fun < best_bound:
                best_spherical = res_polish.x
                best_bound = res_polish.fun
        
        if time.time() - start_time < deadline - 15:
            # Convert back and potentially do another round
            best_coeffs_from_spherical = spherical_to_cartesian(best_spherical)
            final_bound = _compute_bound(best_coeffs_from_spherical)
            
            if final_bound < best_bound:
                best_coeffs = best_coeffs_from_spherical
                best_bound = final_bound
        
        # Final check
        if time.time() - start_time < deadline - 5:
            # One more polish on Cartesian coordinates
            res_final = minimize(
                _compute_bound,
                best_coeffs,
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000, 'maxfev': 10000}
            )
            
            if res_final.fun < best_bound:
                best_coeffs = res_final.x
                best_bound = res_final.fun
                
    except Exception as e:
        # Fall back to initial best
        pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
