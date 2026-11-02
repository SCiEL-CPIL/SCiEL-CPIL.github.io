import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s limit
    
    # Use exact Gauss-Hermite quadrature.
    # g is degree 12. g^2 is degree 24.
    # x^2 g^2 is degree 26.
    # (g' - xg)^2 involves derivatives, so degree increases.
    # g' is degree 11. xg is degree 13.
    # g' - xg is degree 13.
    # (g' - xg)^2 is degree 26.
    # So all integrands are polynomials of degree <= 26 times e^{-x^2}.
    # Gauss-Hermite with N points integrates polynomials up to degree 2N-1 exactly.
    # We need 2N-1 >= 26 => 2N >= 27 => N >= 13.5.
    # So N=14 is sufficient for exact integration.
    N_points = 14
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(N_points)
    
    # Pre-compute powers and Hermite polynomials at nodes
    x = x_nodes
    x2 = x**2
    x3 = x2 * x
    x4 = x2**2
    x5 = x4 * x
    x6 = x4 * x2
    x7 = x6 * x
    x8 = x4**2
    x9 = x8 * x
    x10 = x8 * x2
    x11 = x10 * x
    x12 = x6**2
    
    # Hermite polynomials (physicists' convention)
    H0 = np.ones_like(x)
    H4 = x4 - 6*x2 + 3
    H8 = x8 - 28*x6 + 210*x4 - 420*x2 + 105
    H12 = x12 - 66*x10 + 1260*x8 - 9240*x6 + 23100*x4 - 27720*x2 + 10395
    
    # Derivatives
    dH0 = np.zeros_like(x)
    dH4 = 4*x3 - 12*x
    dH8 = 8*x7 - 168*x5 + 840*x3 - 840*x
    dH12 = 12*x11 - 660*x9 + 10080*x7 - 55440*x5 + 92400*x3 - 55440*x

    def compute_score_fast(coeffs):
        """
        Computes the score using exact Gauss-Hermite quadrature.
        coeffs: (c0, c4, c8)
        """
        if time.time() > deadline - 2:
            return np.inf
            
        c0, c4, c8 = coeffs
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        
        g = c0*H0 + c4*H4 + c8*H8 + c12*H12
        gp = c0*dH0 + c4*dH4 + c8*dH8 + c12*dH12
        
        # Integrals
        I0 = np.sum(w_nodes * g**2)
        if I0 <= 1e-30 or not np.isfinite(I0):
            return np.inf
                
        Ix2 = np.sum(w_nodes * x2 * g**2)
        if not np.isfinite(Ix2) or Ix2 < 0:
            return np.inf
                
        deriv_term = gp - x * g
        Ideriv = np.sum(w_nodes * deriv_term**2)
        if not np.isfinite(Ideriv) or Ideriv < 0:
            return np.inf
                
        A_f_sq = Ix2 / I0
        A_gf_sq = (1.0 / (4.0 * np.pi**2)) * Ideriv / I0
        
        return np.sqrt(A_f_sq * A_gf_sq)

    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = compute_score_fast(x0)
    best_coeffs = x0.copy()
    
    # Structural change: Use a systematic grid search in log-space for the 
    # dominant coefficient c0, with fine-tuning of c4 and c8 via local optimization.
    # The key insight is that c0 sets the scale, while c4/c8 control shape.
    
    # Phase 1: Coarse grid search over c0 in log-space
    # c0 ranges roughly from -5 to 5 based on seed value
    c0_values = np.linspace(-5.0, 5.0, 25)
    c4_values = np.linspace(-1.0, 1.0, 15)
    c8_values = np.linspace(-0.1, 0.1, 10)
    
    # Evaluate on a sparse grid to find promising regions
    grid_best_score = best_score
    grid_best_coeffs = best_coeffs.copy()
    
    for c0_val in c0_values:
        if time.time() > deadline - 60:
            break
        for c4_val in c4_values:
            if time.time() > deadline - 60:
                break
            for c8_val in c8_values:
                if time.time() > deadline - 60:
                    break
                candidate = np.array([c0_val, c4_val, c8_val])
                score = compute_score_fast(candidate)
                if score < grid_best_score and np.isfinite(score):
                    grid_best_score = score
                    grid_best_coeffs = candidate.copy()
    
    if grid_best_score < best_score:
        best_score = grid_best_score
        best_coeffs = grid_best_coeffs.copy()
    
    # Phase 2: Multi-start local optimization from multiple basins
    # Generate diverse starting points based on the grid results and random perturbations
    np.random.seed(42)
    
    # Create a set of diverse starting points
    start_points = []
    
    # Add the grid best
    start_points.append(grid_best_coeffs.copy())
    
    # Add the seed point
    start_points.append(x0.copy())
    
    # Add random points in the promising region
    for _ in range(20):
        if time.time() > deadline - 30:
            break
        # Sample from a Gaussian centered at the grid best with increasing spread
        spread = np.array([2.0, 0.5, 0.05])
        point = grid_best_coeffs + np.random.randn(3) * spread
        start_points.append(point)
    
    # Add some structured points
    structured_starts = [
        np.array([-2.0, 0.1, 0.0]),
        np.array([2.0, 0.1, 0.0]),
        np.array([-1.0, 0.5, 0.01]),
        np.array([1.0, -0.5, -0.01]),
        np.array([-3.0, -0.2, 0.02]),
        np.array([3.0, 0.2, -0.02]),
    ]
    start_points.extend(structured_starts)
    
    # Local optimization from each starting point
    for start_pt in start_points:
        if time.time() > deadline - 15:
            break
        
        try:
            res = opt.minimize(
                compute_score_fast, 
                start_pt, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res_final = opt.minimize(
                compute_score_fast, 
                best_coeffs, 
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if res_final.fun < best_score and np.isfinite(res_final.fun):
                best_score = res_final.fun
                best_coeffs = res_final.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
