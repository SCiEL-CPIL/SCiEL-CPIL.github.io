"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # safety margin before 210s kill
    
    # Use scipy's Hermite polynomial evaluation (physicists' convention)
    # H_n(x) via eval_hermite
    
    def get_c12(c0, c4, c8):
        """Compute c12 from constraint g(0) = 0."""
        H0_0 = 1.0
        H4_0 = 12.0
        H8_0 = 1680.0
        H12_0 = 665280.0
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    def compute_bound(c0, c4, c8):
        """Compute the upper bound on C4 for the given coefficients."""
        c12 = get_c12(c0, c4, c8)
        
        # Use numerical quadrature with a fine grid
        # The integrand involves exp(-x^2) and exp(-2x^2), so range [-10, 10] is sufficient
        x = np.linspace(-12, 12, 200000)
        
        # Evaluate Hermite polynomials at x
        H0_x = np.ones_like(x)
        H4_x = eval_hermite(4, x)
        H8_x = eval_hermite(8, x)
        H12_x = eval_hermite(12, x)
        
        p_x = c0 * H0_x + c4 * H4_x + c8 * H8_x + c12 * H12_x
        
        # ||g||_2^2 = ∫ p(x)^2 * exp(-x^2) dx
        integrand2 = p_x**2 * np.exp(-x**2)
        norm2_sq = np.trapz(integrand2, x)
        
        # ||g||_4^4 = ∫ p(x)^4 * exp(-2x^2) dx
        integrand4 = p_x**4 * np.exp(-2*x**2)
        norm4_4 = np.trapz(integrand4, x)
        
        if norm2_sq <= 1e-30 or norm4_4 <= 1e-30:
            return 1e10
        
        norm2 = np.sqrt(norm2_sq)
        norm4 = norm4_4**0.25
        
        # Bound: C4 = (||g||_4 / ||g||_2)^2 since g = ĝ for even Hermite-Gaussians
        bound = (norm4 / norm2)**2
        
        return bound
    
    # Test seed
    seed_c0, seed_c4, seed_c8 = -113.0/100.0, 1.0/25.0, 1.0/3240.0
    seed_bound = compute_bound(seed_c0, seed_c4, seed_c8)
    
    def objective(coeffs):
        return compute_bound(coeffs[0], coeffs[1], coeffs[2])
    
    best_bound = seed_bound
    best_coeffs = np.array([seed_c0, seed_c4, seed_c8])
    
    # Multi-start optimization
    np.random.seed(123)
    
    # Define starting points: random + structured
    starts = []
    
    # Random starts
    for i in range(80):
        c0 = np.random.uniform(-3, 1)
        c4 = np.random.uniform(-1, 1)
        c8 = np.random.uniform(-0.5, 0.5)
        starts.append([c0, c4, c8])
    
    # Structured starts around the seed and other promising regions
    structured = [
        [seed_c0, seed_c4, seed_c8],
        [seed_c0 * 1.1, seed_c4, seed_c8],
        [seed_c0, seed_c4 * 1.1, seed_c8],
        [seed_c0, seed_c4, seed_c8 * 1.1],
        [seed_c0 * 0.9, seed_c4, seed_c8],
        [seed_c0, seed_c4 * 0.9, seed_c8],
        [seed_c0, seed_c4, seed_c8 * 0.9],
        [-1.0, 0.04, 0.0003],
        [-1.2, 0.04, 0.0003],
        [-0.8, 0.04, 0.0003],
        [-1.13, 0.045, 0.0003],
        [-1.13, 0.035, 0.0003],
        [-1.13, 0.04, 0.0004],
        [-1.13, 0.04, 0.0002],
        [-1.0, 0.0, 0.0],
        [-1.5, 0.05, 0.0005],
        [-0.5, 0.02, 0.0001],
        [0.0, 0.0, 0.0],
        [-2.0, 0.1, 0.001],
    ]
    starts.extend(structured)
    
    for x0 in starts:
        if time.time() > deadline - 30:
            break
        
        x0 = np.array(x0, dtype=float)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                            options={'gtol': 1e-14, 'maxiter': 5000, 'maxfun': 100000})
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Final polish with high precision
    if time.time() < deadline - 10:
        try:
            result = minimize(objective, best_coeffs, method='L-BFGS-B',
                            options={'gtol': 1e-16, 'maxiter': 10000, 'maxfun': 200000})
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return np.array(best_coeffs)

# EVOLVE-BLOCK-END

