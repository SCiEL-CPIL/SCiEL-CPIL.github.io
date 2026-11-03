"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_c4_bound(roots):
    """Compute the C4 upper bound for given double-root locations."""
    roots = np.asarray(roots, dtype=float)
    if np.any(roots <= 0) or np.any(np.diff(np.sort(roots)) <= 0):
        return 1e10
    
    # For Laguerre combination with m=12, n=1:
    # The polynomial is P(x) = prod_{i=1}^{12} (x - r_i)^2 * L_1^{(24)}(x)
    # where L_1^{(24)}(x) = -x + 25 (first Laguerre with alpha=24)
    # Actually for the Cohn-Goncalves construction, we need to be more careful.
    
    # The key quantity is 1/sqrt(||P||^2) where ||P|| is the L2 norm on [0, infinity)
    # with weight e^{-x}. For a product of (x-r_i)^2 terms times a Laguerre poly,
    # we can compute the norm numerically.
    
    # Simplified: use the fact that for the Fourier sign uncertainty bound,
    # C4 relates to the L2 norm of the constructed function.
    # We'll compute it via Gauss-Laguerre quadrature.
    
    # Number of quadrature points for good accuracy
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Build the polynomial P(x) = prod (x - r_i)^2 * L_1^{(24)}(x)
    # L_1^{(alpha)}(x) = -(x) + (alpha + 1) = -x + 25
    laguerre_part = -x_quad + 25.0
    
    # Product of (x - r_i)^2
    prod_part = np.ones_like(x_quad)
    for r in roots:
        prod_part *= (x_quad - r) ** 2
    
    P_vals = prod_part * laguerre_part
    norm_sq = np.sum(w_quad * P_vals ** 2)
    
    if norm_sq <= 0:
        return 1e10
    
    # C4 bound is proportional to 1/sqrt(norm_sq), but we need the exact constant.
    # The seed returns [4,8,...,48] with score 1.0, so let's calibrate.
    # Actually, the score IS the C4 value (lower is better), and seed gives 1.0.
    # Let's compute what the seed gives and use that as reference.
    
    return 1.0 / np.sqrt(norm_sq)

def _compute_c4_relative(roots):
    """Compute C4 bound relative to seed configuration for calibration."""
    seed_roots = [4.0 * (i + 1) for i in range(12)]
    seed_val = _compute_c4_bound(seed_roots)
    if seed_val <= 0:
        return 1.0
    return _compute_c4_bound(roots) / seed_val

def _objective(params):
    """Objective: minimize C4 bound using geometric parameterization."""
    log_r0, log_q = params
    r0 = np.exp(log_r0)
    q = np.exp(log_q)
    
    if r0 <= 0 or q <= 0:
        return 1e10
    
    roots = r0 * q ** np.arange(12)
    
    if np.any(roots <= 0):
        return 1e10
    
    # Check roots are positive and reasonably bounded
    if np.any(roots > 1e6) or np.any(roots < 1e-6):
        return 1e10
    
    return _compute_c4_relative(roots)

def construct():
    """Construct 12 positive double-root locations minimizing C4."""
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0  # seconds before deadline
    
    best_score = 1.0  # seed value
    best_roots = [4.0 * (i + 1) for i in range(12)]
    
    # Geometric parameterization: roots = r0 * q^i for i=0..11
    # Optimize log(r0) and log(q)
    
    # Multistart with various initial conditions
    initial_params = []
    
    # Start near seed: seed is arithmetic [4,8,...,48]
    # Geometric approximation: r0=4, q such that 4*q^11 ≈ 48 => q ≈ (12)^(1/11)
    q_seed = (48.0/4.0) ** (1.0/11.0)
    initial_params.append([np.log(4.0), np.log(q_seed)])
    
    # Other starting points
    for r0_try in [2.0, 3.0, 5.0, 6.0, 8.0, 10.0]:
        for q_try in [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5]:
            initial_params.append([np.log(r0_try), np.log(q_try)])
    
    # Also try some extreme-ish values
    initial_params.append([np.log(1.0), np.log(1.5)])
    initial_params.append([np.log(10.0), np.log(1.1)])
    initial_params.append([np.log(0.5), np.log(1.6)])
    initial_params.append([np.log(20.0), np.log(1.05)])
    
    np.random.seed(42)
    
    for i, x0 in enumerate(initial_params):
        if time.time() > deadline - safety_margin:
            break
        
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=[(-2, 6), (-0.1, 0.5)],
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                log_r0, log_q = result.x
                r0 = np.exp(log_r0)
                q = np.exp(log_q)
                roots = r0 * q ** np.arange(12)
                
                if np.all(roots > 0) and np.all(np.diff(roots) > 0):
                    score = result.fun
                    if score < best_score:
                        best_score = score
                        best_roots = list(roots)
                        
        except Exception:
            continue
    
    # If we found something better, return it; otherwise return seed
    if best_score < 1.0:
        return best_roots
    else:
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
