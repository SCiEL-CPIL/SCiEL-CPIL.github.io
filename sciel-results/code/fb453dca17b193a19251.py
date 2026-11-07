"""Rewritten program: Gram-Schmidt parameterization + differential evolution."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings("ignore")


# Precompute Hermite polynomial coefficients (physicists' convention)
# H_n(x) has only even powers for even n
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 3072x^6 + 13440x^4 - 26880x^2 + 1680
# H12(x) = 4096x^12 - 98304x^10 + 1032192x^8 - 5806080x^6 + 16515072x^4 - 25401600x^2 + 13305600

# Coefficients from highest degree to constant
# H0: [1]
# H4: [16, 0, -48, 0, 12]
# H8: [256, 0, -3072, 0, 13440, 0, -26880, 0, 1680]
# H12: [4096, 0, -98304, 0, 1032192, 0, -5806080, 0, 16515072, 0, -25401600, 0, 13305600]

H0_coeffs = np.array([1.0])
H4_coeffs = np.array([16.0, 0.0, -48.0, 0.0, 12.0])
H8_coeffs = np.array([256.0, 0.0, -3072.0, 0.0, 13440.0, 0.0, -26880.0, 0.0, 1680.0])
H12_coeffs = np.array([4096.0, 0.0, -98304.0, 0.0, 1032192.0, 0.0, -5806080.0, 0.0, 16515072.0, 0.0, -25401600.0, 0.0, 13305600.0])

# Values at x=0
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 13305600.0


def _compute_value_fast(c0, c1, c2):
    """
    Compute the objective value for given coefficients (c0, c1, c2) using fast numerical methods.
    Lower is better. Returns a large penalty for invalid configurations.
    """
    try:
        # Compute c3 so that g(0) = 0
        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
        
        # Build g coefficients (degree 12 down to 0)
        # Index 0 = x^12, index 12 = x^0
        
        # H0 is degree 0: coefficient 1 at x^0
        H0_padded = np.zeros(13)
        H0_padded[12] = 1.0
        
        # H4 is degree 4: coefficients [16, 0, -48, 0, 12] for x^4, x^3, x^2, x^1, x^0
        # In 13-length array: x^4 at index 8, x^3 at 9, x^2 at 10, x^1 at 11, x^0 at 12
        H4_padded = np.zeros(13)
        H4_padded[8] = 16.0
        H4_padded[9] = 0.0
        H4_padded[10] = -48.0
        H4_padded[11] = 0.0
        H4_padded[12] = 12.0
        
        # H8 is degree 8: coefficients [256, 0, -3072, 0, 13440, 0, -26880, 0, 1680] for x^8..x^0
        # In 13-length array: x^8 at index 4, x^7 at 5, x^6 at 6, x^5 at 7, x^4 at 8, x^3 at 9, x^2 at 10, x^1 at 11, x^0 at 12
        H8_padded = np.zeros(13)
        H8_padded[4] = 256.0
        H8_padded[5] = 0.0
        H8_padded[6] = -3072.0
        H8_padded[7] = 0.0
        H8_padded[8] = 13440.0
        H8_padded[9] = 0.0
        H8_padded[10] = -26880.0
        H8_padded[11] = 0.0
        H8_padded[12] = 1680.0
        
        # H12 is degree 12: coefficients already in 13-length array
        H12_padded = H12_coeffs.copy()
        
        g_coeffs = c0 * H0_padded + c1 * H4_padded + c2 * H8_padded + c3 * H12_padded
        
        # Check if g -> -infinity as x -> infinity
        # Leading coefficient is g_coeffs[0] (x^12 term)
        if g_coeffs[0] < 0:
            g_coeffs = -g_coeffs
        
        # q = g / x^2. This requires g[12] = 0 and g[11] = 0 (x^0 and x^1 coefficients are zero)
        # Since all Hermite polynomials have only even powers, x^1 coefficient is always 0
        # x^0 coefficient: g_coeffs[12] = c0*1 + c1*12 + c2*1680 + c3*13305600
        # By construction, this should be 0, but let's verify
        if abs(g_coeffs[12]) > 1e-10:
            return 1e10
        
        # q is a polynomial of degree 10: coefficients g_coeffs[0:11] (x^10 down to x^0)
        q_coeffs = g_coeffs[0:11]
        
        # Find real roots of q
        roots = np.roots(q_coeffs)
        valid_roots = []
        for root in roots:
            if abs(root.imag) < 1e-10:
                r_real = root.real
                # Check sign change using numerical evaluation
                eps = 1e-6
                q_minus = np.polyval(q_coeffs, r_real - eps)
                q_plus = np.polyval(q_coeffs, r_real + eps)
                if q_minus * q_plus < 0:
                    valid_roots.append(r_real)
        
        if not valid_roots:
            return 1e10
        
        r = max(valid_roots)
        value = r**2 / (2 * np.pi)
        return value
        
    except Exception:
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
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    seed_value = _compute_value_fast(seed[0], seed[1], seed[2])
    
    # Differential evolution with multiple restarts
    # Bounds for c0, c1, c2 - focus on region near seed
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    
    best_value = seed_value
    best_params = seed.copy()
    
    # Multiple DE restarts with different seeds and bounds
    for restart in range(5):
        try:
            # Vary the bounds slightly for each restart
            scale = 1.0 + restart * 0.5
            bounds_r = [(-5 * scale, 5 * scale), (-5 * scale, 5 * scale), (-5 * scale, 5 * scale)]
            
            result = differential_evolution(
                _objective,
                bounds_r,
                seed=42 + restart,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                polish=True,
                init='sobol',
                mutation=(0.5, 1.0),
                recombination=0.9
            )
            
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x.copy()
                
        except Exception:
            pass
    
    # Local refinement from best point
    if best_value < seed_value:
        # Try small perturbations around best point
        np.random.seed(123)
        for _ in range(50):
            perturbation_scale = 0.1
            c0 = best_params[0] + np.random.normal(0, perturbation_scale)
            c1 = best_params[1] + np.random.normal(0, perturbation_scale)
            c2 = best_params[2] + np.random.normal(0, perturbation_scale)
            val = _compute_value_fast(c0, c1, c2)
            if val < best_value:
                best_value = val
                best_params = np.array([c0, c1, c2])
    
    # Try random perturbations around seed
    np.random.seed(456)
    for _ in range(50):
        c0 = seed[0] + np.random.normal(0, 0.1)
        c1 = seed[1] + np.random.normal(0, 0.05)
        c2 = seed[2] + np.random.normal(0, 0.01)
        val = _compute_value_fast(c0, c1, c2)
        if val < best_value:
            best_value = val
            best_params = np.array([c0, c1, c2])
    
    # Return best if improved, else seed
    if best_value < seed_value:
        return best_params
    return seed
# EVOLVE-BLOCK-END
