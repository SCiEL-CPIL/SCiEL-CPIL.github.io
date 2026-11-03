import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite
import time

# Precompute Hermite polynomials as explicit power coefficients for speed
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680
# H12(x) = 4096x^12 - 367008x^10 + 6291456x^8 - 39813120x^6 + 118464000x^4 - 153676800x^2 + 39916800

# Let's verify H12(0) using the recurrence or known value.
# H_{2k}(0) = (-1)^k (2k)! / k!
# k=6: (-1)^6 * 12! / 6! = 479001600 / 720 = 665280.
# Wait, the previous code used 332640. Let's re-calculate.
# 12! = 479001600
# 6! = 720
# 479001600 / 720 = 665280.
# Let's check H8(0): k=4. (-1)^4 * 8! / 4! = 40320 / 24 = 1680. Correct.
# Let's check H4(0): k=2. (-1)^2 * 4! / 2! = 24 / 2 = 12. Correct.
# So H12(0) = 665280.
# The previous code had a12 = (a0 - 12*a4 + 1680*a8) / 332640.0
# This implies H12(0) was thought to be -332640? Or the sign was flipped?
# P(0) = a0*1 + a4*(-12) + a8*(1680) + a12*(665280) = 0
# a12 = -(a0 - 12*a4 + 1680*a8) / 665280.
# Let's stick to the standard definition.
# Physicist's Hermite polynomials H_n(x).
# H_0 = 1
# H_1 = 2x
# H_2 = 4x^2 - 2
# H_3 = 8x^3 - 12x
# H_4 = 16x^4 - 48x^2 + 12
# H_5 = 32x^5 - 160x^3 + 120x
# H_6 = 64x^6 - 480x^4 + 720x^2 - 120
# H_7 = 128x^7 - 1344x^5 + 3360x^3 - 1680x
# H_8 = 256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680
# H_9 = 512x^9 - 10752x^7 + 60480x^5 - 120960x^3 + 60480x
# H_10 = 1024x^10 - 35840x^8 + 322560x^6 - 1075200x^4 + 1075200x^2 - 120960
# H_11 = 2048x^11 - 114688x^9 + 1433600x^7 - 6839040x^5 + 13516800x^3 - 6839040x
# H_12 = 4096x^12 - 367008x^10 + 6291456x^8 - 39813120x^6 + 118464000x^4 - 153676800x^2 + 665280

# Coefficients for powers x^0, x^2, ..., x^12
H0_coeffs = np.array([1.0])
H4_coeffs = np.array([12.0, -48.0, 16.0]) # x^0, x^2, x^4
H8_coeffs = np.array([1680.0, -13440.0, 13440.0, -3584.0, 256.0]) # x^0..x^8
H12_coeffs = np.array([665280.0, -153676800.0, 118464000.0, -39813120.0, 6291456.0, -367008.0, 4096.0]) # x^0..x^12

# Precompute integration grid
# Use a finer grid for accuracy but keep it manageable
x_grid = np.linspace(-10, 10, 2000)
dx = x_grid[1] - x_grid[0]
x2 = x_grid**2
x4 = x2**2
x6 = x4*x2
x8 = x4**2
x10 = x8*x2
x12 = x8*x4

# Precompute Gaussian weights
exp_2pi = np.exp(-2 * np.pi * x2)
exp_4pi = np.exp(-4 * np.pi * x2)

def _compute_objective_fast(coeffs):
    """
    Fast objective function using precomputed grid and polynomial evaluation.
    coeffs = [a0, a4, a8]
    """
    a0, a4, a8 = coeffs
    
    # Calculate a12 such that P(0) = 0
    # P(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0)=1, H4(0)=-12 (wait, H4(x)=16x^4-48x^2+12, so H4(0)=12)
    # H8(0)=1680
    # H12(0)=665280
    # a0*1 + a4*12 + a8*1680 + a12*665280 = 0
    a12 = -(a0 + 12*a4 + 1680*a8) / 665280.0
    
    # Evaluate P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    # Using the precomputed powers
    P = (a0 * 1.0 + 
         a4 * (12.0 - 48.0*x2 + 16.0*x4) + 
         a8 * (1680.0 - 13440.0*x2 + 13440.0*x4 - 3584.0*x6 + 256.0*x8) +
         a12 * (665280.0 - 153676800.0*x2 + 118464000.0*x4 - 39813120.0*x6 + 6291456.0*x8 - 367008.0*x10 + 4096.0*x12))
    
    # P_minus(x) for Fourier side (i^12 = -1, so H12 term flips sign in the polynomial part if we consider the analytic continuation? 
    # Actually, the Fourier transform of H_n(x)e^{-pi x^2} is i^n H_n(x) e^{-pi x^2}.
    # So the Fourier side polynomial is Q(x) = a0*H0(x) + a4*i^4*H4(x) + a8*i^8*H8(x) + a12*i^12*H12(x)
    # i^4 = 1, i^8 = 1, i^12 = -1.
    # So Q(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) - a12*H12(x)
    P_minus = (a0 * 1.0 + 
               a4 * (12.0 - 48.0*x2 + 16.0*x4) + 
               a8 * (1680.0 - 13440.0*x2 + 13440.0*x4 - 3584.0*x6 + 256.0*x8) -
               a12 * (665280.0 - 153676800.0*x2 + 118464000.0*x4 - 39813120.0*x6 + 6291456.0*x8 - 367008.0*x10 + 4096.0*x12))
    
    # Compute norms
    # ||g||_2^2 = integral |P(x)|^2 e^{-2pi x^2} dx
    # ||g||_4^2 = integral |P(x)|^4 e^{-4pi x^2} dx
    # Same for Fourier side with P_minus
    
    norm2_sq = np.sum(P**2 * exp_2pi) * dx
    norm4_sq = np.sum(P**4 * exp_4pi) * dx
    
    norm2_sq_hat = np.sum(P_minus**2 * exp_2pi) * dx
    norm4_sq_hat = np.sum(P_minus**4 * exp_4pi) * dx
    
    if norm2_sq < 1e-10 or norm2_sq_hat < 1e-10:
        return 1e10
        
    obj = (norm4_sq / norm2_sq**2) * (norm4_sq_hat / norm2_sq_hat**2)
    
    return obj

def construct():
    start_time = time.time()
    deadline = 170.0 # Safety margin
    
    # Seed
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    seed_score = _compute_objective_fast(x0)
    
    best_result = x0
    best_score = seed_score
    
    # Strategy:
    # 1. Differential Evolution for global exploration (robust to local minima)
    # 2. Local refinement with L-BFGS-B from the best DE result and seed
    
    # Differential Evolution
    # Bounds: a0 in [-5, 5], a4 in [-1, 1], a8 in [-0.01, 0.01]
    # The seed has a0=-1.13, a4=0.04, a8=0.0003
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
    
    if time.time() < deadline - 30: # Leave time for local refinement
        try:
            de_result = differential_evolution(
                _compute_objective_fast,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-10,
                polish=False # We will polish manually
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x.copy()
        except Exception:
            pass
    
    # Local refinement from multiple starts
    starts = [
        x0,
        best_result,
        best_result + np.array([0.1, 0.01, 0.0001]),
        best_result - np.array([0.1, 0.01, 0.0001]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.02, 0.0002])
    ]
    
    for s in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(_compute_objective_fast, s, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
            
    return best_result
# EVOLVE-BLOCK-END
