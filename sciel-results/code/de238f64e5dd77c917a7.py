"""Optimized program for uncertainty inequality - Hermite-Gaussian test function tuning."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_polynomials(x):
    """Compute H0, H4, H8, H12 at points x."""
    H0 = np.ones_like(x)
    H4 = x**4 - 6*x**2 + 3
    H8 = x**8 - 24*x**6 + 72*x**4 - 32*x**2 + 15
    H12 = x**12 - 60*x**10 + 300*x**8 - 1200*x**6 + 1200*x**4 - 168*x**2 + 105
    return H0, H4, H8, H12


def _compute_objective(c):
    """Compute the upper bound on C4 for given coefficients c = [c0, c4, c8]."""
    c0, c4, c8 = c
    # c12 determined by P(0) = 0: c0 + 3*c4 + 15*c8 + 105*c12 = 0
    c12 = -(c0 + 3*c4 + 15*c8) / 105.0
    
    # Use a fine grid for numerical integration
    x = np.linspace(-8, 8, 20001)
    dx = x[1] - x[0]
    
    H0, H4, H8, H12 = _hermite_polynomials(x)
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    
    # Gaussian weight: exp(-pi*x^2)
    gauss = np.exp(-np.pi * x**2)
    f = P * gauss
    
    # A(f) = integral of |f(x)| dx
    Af = np.sum(np.abs(f)) * dx
    
    # For the Fourier transform of x*exp(-pi*x^2)*P(x), we need to be careful
    # The Fourier transform convention: F(f)(xi) = integral f(x) exp(-2*pi*i*x*xi) dx
    # For f(x) = P(x)*exp(-pi*x^2), the Fourier transform can be computed
    # using the fact that F[x^n * exp(-pi*x^2)] involves derivatives of F[exp(-pi*x^2)]
    # F[exp(-pi*x^2)] = exp(-pi*xi^2)
    # 
    # Actually, for this problem, the standard approach is:
    # f(x) = g(x) * exp(-pi*x^2) where g is a polynomial
    # F(f)(xi) = (1/sqrt(pi)) * (some polynomial in xi) * exp(-pi*xi^2)
    # 
    # Let's use the property that for f(x) = P(x)*exp(-pi*x^2):
    # F(f)(xi) = i^k * (d^k/dxi^k) [exp(-pi*xi^2)] / (2*pi)^k ... this gets complex
    # 
    # Simpler: use numerical FFT or direct computation
    # Since the function is smooth and decays fast, we can compute F numerically
    
    # Use a larger grid for FFT to avoid aliasing
    N = 2**15  # 32768
    x_fft = np.linspace(-8, 8, N, endpoint=False)
    dx_fft = x_fft[1] - x_fft[0]
    
    H0_f, H4_f, H8_f, H12_f = _hermite_polynomials(x_fft)
    P_f = c0 * H0_f + c4 * H4_f + c8 * H8_f + c12 * H12_f
    f_fft = P_f * np.exp(-np.pi * x_fft**2)
    
    # FFT: F(f)(xi) = sum f(x_n) exp(-2*pi*i*x_n*xi) dx
    # Using numpy FFT: ifft gives us the inverse, so we need fft
    # The relationship: F(f)(xi) ≈ dx * FFT(f) evaluated at appropriate frequencies
    F_f = np.fft.fft(f_fft) * dx_fft
    # The frequency axis for FFT of length N with spacing dx: xi_k = k/(N*dx) for k=0..N/2, then negative
    # Actually, np.fft.fftshift helps, but let's just compute |F_f| and sum
    
    # The frequencies are: xi_k = k / (N * dx_fft) for k = 0, 1, ..., N-1
    # But we need to account for the shift since x starts at -8
    # xi_k = (k - N/2) / (N * dx_fft) for centered
    # Actually for the integral approximation, we just need sum |F_f| * (1/(N*dx))... 
    # Let me reconsider: the DFT approximates the integral
    # F(f)(xi_k) ≈ dx_fft * sum_{n=0}^{N-1} f(x_n) exp(-2*pi*i*n*k/N)
    # which is what np.fft.fft computes (times dx_fft)
    # The frequency spacing in xi is 1/(N*dx_fft)
    # So A(F(f)) ≈ sum |F_f| * (1/(N*dx_fft))
    
    AFf = np.sum(np.abs(F_f)) / (N * dx_fft)
    
    return Af * AFf


def construct():
    """Return three coefficients [c0, c4, c8] that minimize the upper bound on C4."""
    start_time = time.time()
    deadline = start_time + 180.0  # 180 second limit
    
    # Seed values from Goncalves et al. 2017
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_c = seed.copy()
    best_val = _compute_objective(best_c)
    
    # Try multiple starting points and optimization methods
    # Use Nelder-Mead and BFGS from different initial points
    
    # Generate diverse starting points
    rng = np.random.default_rng(42)
    start_points = [seed]
    
    # Add perturbed versions
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for _ in range(3):
            pert = seed + rng.normal(0, scale, size=3)
            start_points.append(pert)
    
    # Also try some hand-crafted points
    start_points.append(np.array([0, 0, 0]))
    start_points.append(np.array([-1, 0.04, 0.0003]))
    start_points.append(np.array([-1.1, 0.04, 0.0003]))
    start_points.append(np.array([-1.2, 0.04, 0.0003]))
    
    for i, x0 in enumerate(start_points):
        if time.time() > deadline - 30:
            break
        
        try:
            # Try Nelder-Mead
            res = minimize(_compute_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 30:
            break
        
        try:
            # Try BFGS
            res = minimize(_compute_objective, x0, method='BFGS',
                          options={'maxiter': 200, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Final polish with L-BFGS-B if time permits
    if time.time() < deadline - 20:
        try:
            res = minimize(_compute_objective, best_c, method='L-BFGS-B',
                          options={'maxiter': 500, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    return best_c
# EVOLVE-BLOCK-END
