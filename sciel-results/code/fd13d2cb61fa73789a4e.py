"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import eval_hermite
import itertools

def construct():
    # Grid for numerical integration
    # Use a finer grid for better accuracy of L4 and FT
    # Extend range slightly to capture tails better
    x = np.linspace(-15, 15, 5000)
    dx = x[1] - x[0]
    
    # Hermite polynomials at x
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Values at 0 for constraint
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    # Precompute exp(-0.5 x^2)
    gauss = np.exp(-0.5 * x**2)
    
    def compute_ratio(c0, c4, c8):
        """Compute the uncertainty bound ratio for given coefficients."""
        # Enforce g(0) = 0 => c12 is determined
        c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
        
        # Construct g(x)
        g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
        
        # Compute norms
        norm2_sq = np.sum(g**2) * dx
        if norm2_sq < 1e-10:
            return 1e10
            
        norm4 = np.sum(np.abs(g)**4) * dx
        norm4_val = norm4 ** 0.25
        
        # Fourier Transform using FFT
        G = np.fft.fft(g) * dx
        dxi = 1 / (len(g) * dx)
        
        norm2_hat_sq = np.sum(np.abs(G)**2) * dxi
        norm4_hat = np.sum(np.abs(G)**4) * dxi
        norm4_hat_val = norm4_hat ** 0.25
        
        if norm2_hat_sq < 1e-10:
            return 1e10

        ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
        
        return ratio

    def objective_log(params, signs):
        """Objective function for log-magnitude optimization with fixed signs."""
        try:
            m0, m4, m8 = params
            c0 = signs[0] * np.exp(m0)
            c4 = signs[1] * np.exp(m4)
            c8 = signs[2] * np.exp(m8)
            return compute_ratio(c0, c4, c8)
        except Exception:
            return 1e10

    # Bounds for log-magnitudes
    # We want to explore a wide range of scales
    bounds = [(-10, 5), (-10, 5), (-10, 5)]
    
    best_score = 1e10
    best_params = None
    best_signs = None
    
    # Iterate over all 8 sign combinations
    for signs in itertools.product([-1, 1], repeat=3):
        # Use differential evolution on the 3 log-magnitudes
        # This is fast and avoids the sign discontinuity issue in the optimizer
        result = differential_evolution(
            objective_log, 
            bounds, 
            args=(signs,),
            seed=42, 
            maxiter=100, 
            popsize=10,
            tol=1e-8, 
            polish=True,
            init='sobol'
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
            best_signs = signs
            
    # Reconstruct coefficients from best result
    if best_params is not None:
        m0, m4, m8 = best_params
        c0 = best_signs[0] * np.exp(m0)
        c4 = best_signs[1] * np.exp(m4)
        c8 = best_signs[2] * np.exp(m8)
        return np.array([c0, c4, c8])
    else:
        # Fallback to seed
        c0_seed = -1.13
        c4_seed = 0.04
        c8_seed = 1.0 / 3240.0
        return np.array([c0_seed, c4_seed, c8_seed])

# EVOLVE-BLOCK-END
