import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import eval_hermite

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 200.0 # Safety margin before 210s

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    # Pre-compute Hermite polynomials at 0 for constraint
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
    # c12 = -(c0 + 3*c4 + 105*c8) / 945
    
    def get_c12(c0, c4, c8):
        return -(c0 + 3.0*c4 + 105.0*c8) / 945.0

    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # Define the polynomial part P(x)
        # We need to integrate x^4 * |P(x) * exp(-x^2/2)| from 0 to inf
        # P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
        
        def integrand(x):
            # Use scipy.special.eval_hermite for efficiency
            h0 = 1.0
            h4 = eval_hermite(4, x)
            h8 = eval_hermite(8, x)
            h12 = eval_hermite(12, x)
            
            p_val = c0*h0 + c4*h4 + c8*h8 + c12*h12
            g_val = p_val * np.exp(-0.5 * x**2)
            
            return (x**4) * np.abs(g_val)
        
        # Integrate from 0 to infinity
        # The function decays fast, so quad should handle it well.
        val, err = quad(integrand, 0, np.inf, limit=200)
        # Multiply by 2 for symmetry over -inf to inf
        return 2.0 * val

    # Initial guesses
    # Seed: [-1.13, 0.04, 1/3240]
    best_score = np.inf
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Try the seed first
    seed_score = objective(best_coeffs)
    if seed_score < best_score:
        best_score = seed_score

    # Multi-start optimization
    # We sample random initial points in a reasonable box
    # c0 is usually negative, c4 positive, c8 small positive/negative
    
    np.random.seed(42)
    
    # Define bounds for optimization
    # Based on typical magnitudes: c0 ~ -1, c4 ~ 0.1, c8 ~ 0.001
    # Let's try a wider range to ensure global min is found
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    num_starts = 50
    start_times = []
    
    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 20: # Stop if close to limit
            break
            
        # Generate random initial guess within bounds
        x0 = np.array([
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1]),
            np.random.uniform(bounds[2][0], bounds[2][1])
        ])
        
        # Use L-BFGS-B or Nelder-Mead
        # Nelder-Mead is robust for non-smooth (absolute value) functions
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.success or res.fun < best_score:
                if res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = res.x
        except Exception:
            pass

    # Final check on time
    if time.time() > TIME_LIMIT - 5:
        return best_coeffs

    # Return the best coefficients found
    return best_coeffs

# Keep the EVOLVE-BLOCK structure as requested by the prompt format, 
# although the prompt says "Rewrite the program from scratch... Keep the function names".
# The current program has construct() inside EVOLVE-BLOCK.
# I will provide the complete program.
