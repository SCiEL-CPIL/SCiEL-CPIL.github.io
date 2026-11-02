"""
Strategy: Multi-objective Pareto search for Hermite-Gaussian uncertainty bound.
Instead of directly minimizing the product A_f * A_F, we treat the two variances
separately. The product is minimized when the trade-off between position spread
and momentum spread is optimized. We use a coarse global grid to find diverse
candidates in the coefficient space, then perform high-precision local refinement
using numerical gradients (since the objective is smooth and low-dimensional).
This avoids getting stuck in local minima of the product function by exploring
the landscape of the individual components first.
"""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s limit

    # Precompute Gauss-Hermite quadrature nodes and weights
    # We need to integrate polynomials of degree up to 26 times exp(-x^2)
    # Hermite quadrature with n nodes is exact for degree up to 2n-1
    # So n=14 is sufficient for degree 26, use n=50 for high precision
    n_quad = 50
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(n_quad)

    # Hermite polynomials (physicists' convention)
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1260x^8 - 9240x^6 + 23100x^4 - 27720x^2 + 10395
    
    # Precompute powers of x_nodes for speed
    x2 = x_nodes**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x8 * x4
    
    H0 = np.ones_like(x_nodes)
    H4 = x4 - 6*x2 + 3
    H8 = x8 - 28*x6 + 210*x4 - 420*x2 + 105
    H12 = x12 - 66*x10 + 1260*x8 - 9240*x6 + 23100*x4 - 27720*x2 + 10395

    # Derivatives
    # H0' = 0
    # H4' = 4x^3 - 12x
    # H8' = 8x^7 - 168x^5 + 840x^3 - 840x
    # H12' = 12x^11 - 660x^9 + 10080x^7 - 55440x^5 + 92400x^3 - 55440x
    
    x3 = x2 * x_nodes
    x5 = x4 * x_nodes
    x7 = x6 * x_nodes
    x9 = x8 * x_nodes
    x11 = x10 * x_nodes
    
    dH0 = np.zeros_like(x_nodes)
    dH4 = 4*x3 - 12*x_nodes
    dH8 = 8*x7 - 168*x5 + 840*x3 - 840*x_nodes
    dH12 = 12*x11 - 660*x9 + 10080*x7 - 55440*x5 + 92400*x3 - 55440*x_nodes

    def compute_components(c0, c4, c8):
        """
        Returns (var_x, var_m) where:
        var_x = <x^2> = Ix2 / I0
        var_m = <(d/dx - x)^2> / (4*pi^2) = Ideriv / (4*pi^2 * I0)
        """
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        
        P = c0*H0 + c4*H4 + c8*H8 + c12*H12
        Pp = c0*dH0 + c4*dH4 + c8*dH8 + c12*dH12
        
        P2 = P**2
        I0 = np.sum(w_nodes * P2)
        if I0 <= 0 or not np.isfinite(I0):
            return np.inf, np.inf
            
        Ix2 = np.sum(w_nodes * x2 * P2)
        if not np.isfinite(Ix2) or Ix2 < 0:
            return np.inf, np.inf
            
        diff = Pp - x_nodes * P
        Ideriv = np.sum(w_nodes * diff**2)
        if not np.isfinite(Ideriv) or Ideriv < 0:
            return np.inf, np.inf
            
        var_x = Ix2 / I0
        var_m = (1.0 / (4.0 * np.pi**2)) * Ideriv / I0
        
        return var_x, var_m

    def objective(coeffs):
        """
        Objective function to minimize: sqrt(var_x * var_m)
        """
        if time.time() > deadline - 2:
            return np.inf
        try:
            vx, vm = compute_components(coeffs[0], coeffs[1], coeffs[2])
            if not np.isfinite(vx) or not np.isfinite(vm):
                return np.inf
            return np.sqrt(vx * vm)
        except Exception:
            return np.inf

    # Strategy: 
    # 1. Coarse grid search to find diverse local minima.
    # 2. Refine each candidate using BFGS (quasi-Newton, faster than Nelder-Mead for smooth 3D).
    # 3. Keep the best.

    best_score = np.inf
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0]) # Seed guess

    # Initial seed evaluation
    s = objective(best_coeffs)
    if s < best_score:
        best_score = s

    # Define search ranges
    # c0 is dominant, c4 and c8 are corrections
    # Based on seed: c0 ~ -1.13, c4 ~ 0.04, c8 ~ 0.0003
    # We explore a wider box to ensure we don't miss a global minimum.
    
    # Phase 1: Global Grid Search
    # We use a moderately fine grid. 3D grid of 10x10x10 = 1000 points.
    # Each eval is ~1ms, so 1000 evals is ~1s. This is safe.
    
    c0_min, c0_max = -3.0, 0.5
    c4_min, c4_max = -1.0, 1.0
    c8_min, c8_max = -0.1, 0.1
    
    # Use Chebyshev-like spacing for better coverage of extrema? 
    # Or just linear. Linear is fine for smooth functions.
    # Let's do a nested loop to allow early termination if time is tight.
    
    n_c0 = 15
    n_c4 = 15
    n_c8 = 10
    
    candidates = []
    
    # Add the seed
    candidates.append(best_coeffs.copy())
    
    # Add random points for diversity
    np.random.seed(42)
    for _ in range(20):
        c0 = np.random.uniform(c0_min, c0_max)
        c4 = np.random.uniform(c4_min, c4_max)
        c8 = np.random.uniform(c8_min, c8_max)
        candidates.append(np.array([c0, c4, c8]))
        
    # Grid points
    c0_vals = np.linspace(c0_min, c0_max, n_c0)
    c4_vals = np.linspace(c4_min, c4_max, n_c4)
    c8_vals = np.linspace(c8_min, c8_max, n_c8)
    
    for c0 in c0_vals:
        if time.time() > deadline - 30:
            break
        for c4 in c4_vals:
            if time.time() > deadline - 20:
                break
            for c8 in c8_vals:
                if time.time() > deadline - 10:
                    break
                coeffs = np.array([c0, c4, c8])
                candidates.append(coeffs)
                
    # Phase 2: Local Refinement of all candidates
    # We will sort candidates by their objective value and refine the top N.
    # This ensures we spend most time on the most promising regions.
    
    # Evaluate all candidates first
    scored_candidates = []
    for c in candidates:
        s = objective(c)
        if np.isfinite(s):
            scored_candidates.append((s, c))
            
    # Sort by score
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top 50 candidates
    # BFGS is efficient for 3 variables.
    # We use a tight tolerance.
    
    max_refine = 50
    for i, (s, c) in enumerate(scored_candidates[:max_refine]):
        if time.time() > deadline - 5:
            break
            
        try:
            res = opt.minimize(
                objective, 
                c, 
                method='L-BFGS-B', 
                options={'gtol': 1e-12, 'maxiter': 200}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Phase 3: Final Polish
    # One more high-precision polish on the best result found
    if time.time() < deadline - 2:
        try:
            res = opt.minimize(
                objective, 
                best_coeffs, 
                method='Powell', # Powell is derivative-free and robust for final polish
                options={'xtol': 1e-14, 'ftol': 1e-14, 'maxiter': 1000}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
