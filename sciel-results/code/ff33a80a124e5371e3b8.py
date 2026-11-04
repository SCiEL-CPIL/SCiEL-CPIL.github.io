# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

# Precompute grid and values for speed
# Range [-5, 5] is sufficient for e^{-2pi x^2} which decays very fast.
# e^{-2pi * 25} ~ 10^{-34}.
# Let's use a fine grid for accuracy.
_N_GRID = 5000
_X_GRID = np.linspace(-5, 5, _N_GRID)
_DX = _X_GRID[1] - _X_GRID[0]

# Precompute Hermite polynomials
_H0 = eval_hermite(0, _X_GRID)
_H4 = eval_hermite(4, _X_GRID)
_H8 = eval_hermite(8, _X_GRID)
_H12 = eval_hermite(12, _X_GRID)

# Precompute Gaussian weights
# g(x) = P(x) e^{-pi x^2}
# |g|^2 = P^2 e^{-2pi x^2}
# |g|^4 = P^4 e^{-4pi x^2}
_EXP_2PI = np.exp(-2 * np.pi * _X_GRID**2)
_EXP_4PI = np.exp(-4 * np.pi * _X_GRID**2)

# Values at x=0 for constraint P(0)=0
# H_n(0)
_H0_0 = eval_hermite(0, 0) # 1
_H4_0 = eval_hermite(4, 0) # -12
_H8_0 = eval_hermite(8, 0) # 1680
_H12_0 = eval_hermite(12, 0) # -332640

def _fast_objective(coeffs):
    a0, a4, a8 = coeffs
    
    # Enforce P(0) = 0
    # a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # a0 - 12a4 + 1680a8 - 332640 a12 = 0
    a12 = (a0 - 12*a4 + 1680*a8) / 332640.0
    
    # Compute P and P_minus on the grid
    # P(x) = a0 H0 + a4 H4 + a8 H8 + a12 H12
    # P_minus(x) = a0 H0 + a4 H4 + a8 H8 - a12 H12
    # Note: Fourier transform of H_n(x) e^{-pi x^2} involves i^n H_n(x) e^{-pi x^2} (up to normalization)
    # Specifically, if g(x) = P(x) e^{-pi x^2}, then \hat{g}(xi) = \hat{P}(xi) e^{-pi xi^2}
    # The problem states P_minus is used for the Fourier side.
    # The seed code uses:
    # P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # P_minus = a0*H0 + a4*H4 + a8*H8 - a12*H12
    # This implies the sign change for the highest order term or specific parity properties.
    
    P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    P_minus = a0*_H0 + a4*_H4 + a8*_H8 - a12*_H12
    
    # Norms
    # norm2_sq = integral |g|^2 dx
    # norm4_sq = integral |g|^4 dx
    # C4 = (norm4_sq / norm2_sq^2) * (norm4_sq_hat / norm2_sq_hat^2)
    
    norm2_sq = np.sum(P**2 * _EXP_2PI) * _DX
    norm4_sq = np.sum(P**4 * _EXP_4PI) * _DX
    
    norm2_sq_hat = np.sum(P_minus**2 * _EXP_2PI) * _DX
    norm4_sq_hat = np.sum(P_minus**4 * _EXP_4PI) * _DX
    
    if norm2_sq < 1e-10 or norm2_sq_hat < 1e-10:
        return 1e10
        
    obj = (norm4_sq / norm2_sq**2) * (norm4_sq_hat / norm2_sq_hat**2)
    
    return obj

def construct():
    start_time = time.time()
    deadline = 180.0 # Safety margin before 210s kill
    
    # Seed coefficients
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    seed_score = _fast_objective(x0)
    
    best_result = x0
    best_score = seed_score
    
    # Define starting points for global search
    # We use Nelder-Mead as it is derivative-free and robust to local minima structures
    # that might trap gradient-based methods if the landscape is noisy or has sharp features.
    
    # Perturbations around the seed
    starts = [x0.copy()]
    
    # Add random perturbations
    rng = np.random.default_rng(42)
    for _ in range(30):
        perturbation = rng.normal(0, 0.5, size=3)
        # Scale perturbation based on typical coefficient magnitudes
        # a0 ~ -1, a4 ~ 0.04, a8 ~ 0.0003
        # Let's use relative scaling or fixed small steps
        step = np.array([0.5, 0.05, 0.001])
        new_start = x0 + perturbation * step
        starts.append(new_start)
        
    # Also try some structured starts
    starts.append(np.array([-1.0, 0.1, 0.001]))
    starts.append(np.array([-0.5, 0.05, 0.0005]))
    starts.append(np.array([-2.0, 0.2, 0.002]))
    
    # Phase 1: Global Search with Nelder-Mead
    for s in starts:
        if time.time() > deadline:
            break
        try:
            # Nelder-Mead options
            # maxiter: maximum number of iterations
            # xatol, fatol: tolerances for convergence
            res = minimize(_fast_objective, s, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
    
    # Phase 2: Local Polish with L-BFGS-B (if time permits)
    # L-BFGS-B is very fast and accurate for smooth functions.
    # We use it to polish the best result found in Phase 1.
    if time.time() < deadline - 10:
        try:
            res_polish = minimize(_fast_objective, best_result, method='L-BFGS-B', 
                                  options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50})
            if res_polish.fun < best_score:
                best_score = res_polish.fun
                best_result = res_polish.x
        except Exception:
            pass
            
    # Phase 3: Final High-Precision Check (Optional, if time is very ample)
    # We could re-integrate with a finer grid, but the current grid (4000 points) is quite fine.
    # e^{-2pi x^2} is negligible beyond |x|>2.
    # The main error source is the trapezoidal rule approximation.
    # Increasing N_GRID would be the next step, but it slows down objective evaluation.
    # Given the time budget, we stick to the current grid for the optimization loop.
    
    return best_result
# EVOLVE-BLOCK-END
