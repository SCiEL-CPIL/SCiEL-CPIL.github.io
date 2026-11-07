"""
Rewritten program: Gram-Schmidt parameterization + differential evolution with explicit penalty.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings("ignore")


def _compute_value_fast(c0, c1, c2):
    """
    Fast numerical computation of the objective value.
    Returns a large penalty for invalid configurations.
    """
    try:
        # Hermite polynomial coefficients (physicists' convention)
        # H0(x) = 1
        # H4(x) = 16x^4 - 48x^2 + 12
        # H8(x) = 256x^8 - 3072x^6 + 13440x^4 - 26880x^2 + 1680
        # H12(x) = 4096x^12 - 98304x^10 + 1032192x^8 - 5806080x^6 + 16515072x^4 - 25401600x^2 + 13305600
        
        # Evaluate at x=0
        H0_0 = 1.0
        H4_0 = 12.0
        H8_0 = 1680.0
        H12_0 = 13305600.0
        
        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
        
        # Build g coefficients (degree 12 down to 0)
        # Index 0 = x^12, index 12 = x^0
        H0_padded = np.zeros(13)
        H0_padded[12] = 1.0
        
        H4_padded = np.zeros(13)
        H4_padded[8] = 16.0
        H4_padded[10] = -48.0
        H4_padded[12] = 12.0
        
        H8_padded = np.zeros(13)
        H8_padded[4] = 256.0
        H8_padded[6] = -3072.0
        H8_padded[8] = 13440.0
        H8_padded[10] = -26880.0
        H8_padded[12] = 1680.0
        
        H12_padded = np.array([4096.0, 0.0, -98304.0, 0.0, 1032192.0, 0.0, 
                               -5806080.0, 0.0, 16515072.0, 0.0, -25401600.0, 0.0, 
                               13305600.0])
        
        g_coeffs = c0 * H0_padded + c1 * H4_padded + c2 * H8_padded + c3 * H12_padded
        
        # Check if g -> -infinity as x -> infinity
        # Leading coefficient is g_coeffs[0] (x^12 term)
        if g_coeffs[0] < 0:
            g_coeffs = -g_coeffs
        
        # q = g / x^2. This requires g[12] = 0 and g[11] = 0 (x^0 and x^1 coefficients are zero)
        # g_coeffs[12] is x^0 coefficient, g_coeffs[11] is x^1 coefficient
        if abs(g_coeffs[12]) > 1e-10 or abs(g_coeffs[11]) > 1e-10:
            return 1e10
        
        # q is a polynomial of degree 10: coefficients g_coeffs[0:11] (x^10 down to x^0)
        q_coeffs = g_coeffs[0:11]
        
        # Check if leading coefficient of q is negative - penalty for this
        if q_coeffs[0] < 0:
            return 1e10
        
        # Find real roots of q
        roots = np.roots(q_coeffs)
        valid_roots = []
        for root in roots:
            if abs(root.imag) < 1e-10:
                r_real = root.real
                # Check sign change using numerical evaluation
                eps = 1e-8
                q_minus = np.polyval(q_coeffs, r_real - eps)
                q_plus = np.polyval(q_coeffs, r_real + eps)
                # Strict sign change required
                if q_minus * q_plus < -1e-12:
                    valid_roots.append(r_real)
        
        if not valid_roots:
            return 1e10
        
        r = max(valid_roots)
        value = r**2 / (2 * np.pi)
        return value
        
    except Exception as e:
        return 1e10


def _objective(params):
    """
    Objective function for differential evolution.
    params = [c0, c1, c2]
    """
    c0, c1, c2 = params
    return _compute_value_fast(c0, c1, c2)


def construct():
    """
    Construct three coefficients [c0, c1, c2] using differential evolution.
    """
    # Initial seed
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    seed_value = _compute_value_fast(seed[0], seed[1], seed[2])
    
    best_value = seed_value
    best_params = seed.copy()
    
    # Run differential evolution with multiple seeds and tighter bounds
    for de_seed in [42, 123, 456, 789, 101112, 202020, 303030]:
        try:
            # Tighter bounds focused on the seed region
            bounds = [(-5, 5), (-2, 2), (-0.5, 0.5)]
            result = differential_evolution(
                _objective,
                bounds,
                seed=de_seed,
                maxiter=80,
                popsize=15,
                tol=1e-10,
                polish=True,
                init='sobol'
            )
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Local refinement around best found using Nelder-Mead
    if best_value < seed_value:
        try:
            from scipy.optimize import minimize
            best_result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if best_result.fun < best_value:
                best_value = best_result.fun
                best_params = best_result.x.copy()
        except Exception:
            pass
    
    if best_value < seed_value:
        return np.array(best_params)
    return seed
# EVOLVE-BLOCK-END
