"""Uncertainty inequality: tune Fourier-invariant Hermite-Gaussian test function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import factorial, gamma
from scipy.optimize import minimize, differential_evolution

# Precompute constants
# Hermite polynomials (physicists') H_n(x)
# We need H0, H4, H8, H12
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
# H12(x) = 4096x^12 - 73728x^10 + 516096x^8 - 1651584x^6 + 2540160x^4 - 1590720x^2 + 103950

def _H0(x):
    return np.ones_like(x)

def _H4(x):
    return 16*x**4 - 48*x**2 + 12

def _H8(x):
    return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680

def _H12(x):
    return (4096*x**12 - 73728*x**10 + 516096*x**8 
            - 1651584*x**6 + 2540160*x**4 - 1590720*x**2 + 103950)

def _g(x, c0, c4, c8):
    """
    g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    g(0) = (c0*1 + c4*12 + c8*1680 + c12*103950) * 1 = 0
    => c12 = -(c0 + 12*c4 + 1680*c8) / 103950
    """
    c12 = -(c0 + 12*c4 + 1680*c8) / 103950
    poly = c0*_H0(x) + c4*_H4(x) + c8*_H8(x) + c12*_H12(x)
    return poly * np.exp(-x**2 / 2)

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on A(f)*A(F(f)).
    
    For a Fourier-invariant test function g, the uncertainty product bound is:
    Δx * Δk = (∫x^2|g|^2 dx / ∫|g|^2 dx) * (∫k^2|ĝ|^2 dk / ∫|ĝ|^2 dk)
    
    Since g is Fourier-invariant (up to normalization), we can compute this numerically.
    """
    # Use a fine grid
    N = 100000
    x = np.linspace(-30, 30, N)
    dx = x[1] - x[0]
    
    g = _g(x, c0, c4, c8)
    
    norm_sq = np.sum(np.abs(g)**2) * dx
    if norm_sq < 1e-15:
        return 1e10
    
    # Δx^2 = ∫x^2|g|^2 dx / ∫|g|^2 dx
    moment_sq = np.sum(x**2 * np.abs(g)**2) * dx
    var_x = moment_sq / norm_sq
    
    # Compute Fourier transform numerically
    # Use FFT with appropriate scaling
    # The Fourier transform convention: ĝ(k) = ∫ g(x) e^{-ikx} dx
    # Using FFT: we need to be careful with scaling
    
    # Use scipy's fft or numpy's fft
    # For a function on x in [-L, L] with N points, dx = 2L/N
    # The FFT gives: FFT[g]_j ≈ dx * sum_m g_m * exp(-2πi m j / N)
    # We want: ĝ(k) = ∫ g(x) e^{-ikx} dx ≈ dx * sum g_m e^{-ik_m x_m}
    # where k_m = 2π m / (N dx) for m = -N/2, ..., N/2-1
    
    L = x[-1] - x[0]  # total length
    dk = 2 * np.pi / L
    
    # Use FFT
    g_fft = np.fft.fftshift(np.fft.fft(g)) * dx
    
    # k values
    k = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
    
    fft_norm_sq = np.sum(np.abs(g_fft)**2) * dk
    if fft_norm_sq < 1e-15:
        return 1e10
    
    # Δk^2 = ∫k^2|ĝ|^2 dk / ∫|ĝ|^2 dk
    moment_k_sq = np.sum(k**2 * np.abs(g_fft)**2) * dk
    var_k = moment_k_sq / fft_norm_sq
    
    # Uncertainty product
    return np.sqrt(var_x * var_k)

def construct():
    """
    Return three coefficients of the Hermite combination (degrees 0, 4, 8).
    The degree-12 weight is fixed so g(0) = 0.
    Score is the upper bound on C4 (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 170  # Safety margin before 180s limit
    
    # Initial guess from prior state of the art
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    best_params = x0.copy()
    best_score = _compute_bound(*x0)
    
    # Try differential evolution for global optimization
    bounds = [(-2, 2), (-0.5, 0.5), (-0.01, 0.01)]
    
    def objective(params):
        return _compute_bound(params[0], params[1], params[2])
    
    # Use differential evolution with limited iterations
    try:
        result = differential_evolution(
            objective, bounds, maxiter=50, popsize=15, 
            seed=42, tol=1e-6, polish=True,
            init='latinhypercube'
        )
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        return best_params
    
    # Refine with local optimization
    try:
        result2 = minimize(
            objective, best_params, method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        if result2.fun < best_score:
            best_score = result2.fun
            best_params = result2.x
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        return best_params
    
    # Try a few random restarts if time permits
    rng = np.random.RandomState(123)
    for i in range(5):
        if time.time() > deadline - 20:
            break
        x_rand = rng.uniform([-2, -0.5, -0.01], [2, 0.5, 0.01])
        try:
            result3 = minimize(
                objective, x_rand, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result3.fun < best_score:
                best_score = result3.fun
                best_params = result3.x
        except Exception:
            pass
    
    return best_params

# EVOLVE-BLOCK-END
