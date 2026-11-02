import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 190.0  # Safety margin before 210s limit

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
    # Using N=14 is extremely fast and exact.
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
    
    # Candidate: Constrained Gaussian (c0=1, c4=0, c8=0)
    x_gauss = np.array([1.0, 0.0, 0.0])
    score_gauss = compute_score_fast(x_gauss)
    if score_gauss < best_score:
        best_score = score_gauss
        best_coeffs = x_gauss.copy()

    # Strategy:
    # 1. Coarse Grid Search / Sobol Sequence for global exploration
    # 2. Local Polish with L-BFGS-B
    
    # Define bounds
    bounds = [(-5, 5), (-0.5, 0.5), (-0.05, 0.05)]
    
    # Generate a Sobol sequence for global sampling
    # Use a reasonable number of points, e.g., 256
    n_samples = 256
    # scipy.stats.qmc.Sobol is available in recent versions
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=3, scramble=True, seed=42)
        samples = sampler.random(n=n_samples)
        # Scale to bounds
        x_min = np.array([b[0] for b in bounds])
        x_max = np.array([b[1] for b in bounds])
        samples = x_min + samples * (x_max - x_min)
    except ImportError:
        # Fallback to random if qmc is not available
        np.random.seed(42)
        samples = np.random.uniform(x_min, x_max, size=(n_samples, 3))

    # Evaluate all samples
    for i in range(n_samples):
        if time.time() > deadline - 30:
            break
        x_try = samples[i]
        score_try = compute_score_fast(x_try)
        if score_try < best_score:
            best_score = score_try
            best_coeffs = x_try.copy()

    # Polish the best found so far
    if time.time() < deadline - 10:
        try:
            res_lbfgs = opt.minimize(
                compute_score_fast, 
                best_coeffs, 
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res_lbfgs.fun < best_score and np.isfinite(res_lbfgs.fun):
                best_score = res_lbfgs.fun
                best_coeffs = res_lbfgs.x.copy()
        except Exception:
            pass

    # Additional local refinement around the seed and Gaussian
    for start_pt in [x0, x_gauss]:
        if time.time() > deadline - 5:
            break
        try:
            res_refine = opt.minimize(
                compute_score_fast, 
                start_pt, 
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res_refine.fun < best_score and np.isfinite(res_refine.fun):
                best_score = res_refine.fun
                best_coeffs = res_refine.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
