import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s limit (15s buffer)
    
    # Pre-compute Gauss-Hermite nodes and weights
    # g is degree 12, g^2 is degree 24.
    # N=16 points integrate up to degree 31 exactly.
    # Use N=128 for high precision and robustness against any numerical issues.
    N_points = 128
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(N_points)
    
    # Precompute for faster evaluation
    w_sqrt = np.sqrt(w_nodes)
    xw = x_nodes * w_sqrt
    x2w = x_nodes**2 * w_sqrt
    
    # Precompute powers of x_nodes for efficiency
    x = x_nodes
    x2 = x**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x6**2
    
    # Hermite Polynomials (Physicists' convention)
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1260x^8 - 9240x^6 + 23100x^4 - 27720x^2 + 10395
    
    H0 = np.ones_like(x)
    H4 = x4 - 6*x2 + 3
    H8 = x8 - 28*x6 + 210*x4 - 420*x2 + 105
    H12 = x12 - 66*x10 + 1260*x8 - 9240*x6 + 23100*x4 - 27720*x2 + 10395
    
    # Derivatives
    x3 = x2 * x
    x5 = x4 * x
    x7 = x6 * x
    x9 = x8 * x
    x11 = x10 * x
    
    dH0 = np.zeros_like(x)
    dH4 = 4*x3 - 12*x
    dH8 = 8*x7 - 168*x5 + 840*x3 - 840*x
    dH12 = 12*x11 - 660*x9 + 10080*x7 - 55440*x5 + 92400*x3 - 55440*x

    def compute_bound(coeffs):
        """
        Computes the uncertainty bound for given coefficients (c0, c4, c8).
        Returns infinity if invalid.
        """
        if time.time() > deadline - 2:
            return np.inf
            
        c0, c4, c8 = coeffs
        
        # c12 is fixed to ensure g(0) = 0
        # g(0) = c0*1 + c4*3 + c8*105 + c12*10395 = 0
        # c12 = -(c0 + 3*c4 + 105*c8) / 10395
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        
        # Evaluate g and g'
        g = c0*H0 + c4*H4 + c8*H8 + c12*H12
        gp = c0*dH0 + c4*dH4 + c8*dH8 + c12*dH12
        
        # Integrals
        # I0 = <g^2>
        I0 = np.sum(w_nodes * g**2)
        if I0 <= 1e-30 or not np.isfinite(I0):
            return np.inf
            
        # Ix2 = <x^2 g^2>
        Ix2 = np.sum(w_nodes * x2 * g**2)
        if not np.isfinite(Ix2) or Ix2 < 0:
            return np.inf
            
        # Ideriv = <(g' - xg)^2>
        deriv_term = gp - x * g
        Ideriv = np.sum(w_nodes * deriv_term**2)
        if not np.isfinite(Ideriv) or Ideriv < 0:
            return np.inf
            
        # A_f * A_F
        # A_f^2 = Ix2 / I0
        # A_F^2 = (1/(4*pi^2)) * Ideriv / I0
        # Product = sqrt( (Ix2/I0) * (Ideriv/(4*pi^2*I0)) )
        
        bound_sq = (Ix2 * Ideriv) / (4.0 * np.pi**2 * I0**2)
        
        if bound_sq < 0:
            return np.inf
            
        return np.sqrt(bound_sq)

    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = compute_bound(x0)
    best_coeffs = x0.copy()
    
    # Generate diverse starting points
    # 1. The seed
    # 2. Perturbations of the seed
    # 3. Points on a grid in log-space or scaled space
    
    starts = [x0]
    
    # Add random perturbations
    np.random.seed(42)
    for _ in range(80):
        # Scale the seed by a random factor and add noise
        scale = np.exp(np.random.normal(0, 1.5))
        noise = np.random.normal(0, 0.15 * np.abs(x0) + 0.02, 3)
        starts.append(x0 * scale + noise)
        
    # Add structured points - explore the coefficient space more thoroughly
    # Try points where one coefficient is dominant
    for i in range(3):
        for scale in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
            p = np.zeros(3)
            p[i] = scale
            # Add small noise to break symmetry
            p += np.random.normal(0, 0.02, 3)
            starts.append(p)
    
    # Add points along specific directions
    directions = [
        [1, 1, 1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
    ]
    for d in directions:
        for scale in [0.1, 0.5, 1.0, 5.0]:
            p = np.array(d) * scale
            p += np.random.normal(0, 0.05, 3)
            starts.append(p)
    
    # Optimization Loop with multiple methods
    for i, x_start in enumerate(starts):
        if time.time() > deadline - 15:
            break
        
        # Use Nelder-Mead for robustness
        try:
            res = opt.minimize(compute_bound, x_start, method='Nelder-Mead', 
                              options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 800, 'adaptive': True})
            
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
                
        except Exception:
            pass
        
        # Also try BFGS from the same starting point
        if time.time() < deadline - 10:
            try:
                res_bfgs = opt.minimize(compute_bound, x_start, method='BFGS',
                                       options={'maxiter': 500, 'gtol': 1e-10})
                if np.isfinite(res_bfgs.fun) and res_bfgs.fun < best_score:
                    best_score = res_bfgs.fun
                    best_coeffs = res_bfgs.x.copy()
            except Exception:
                pass

    # Final Polish with multiple high-precision methods
    if time.time() < deadline - 8:
        # L-BFGS-B with tight tolerances
        try:
            res2 = opt.minimize(compute_bound, best_coeffs, method='L-BFGS-B',
                               options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14})
            if np.isfinite(res2.fun) and res2.fun < best_score:
                best_score = res2.fun
                best_coeffs = res2.x.copy()
        except Exception:
            pass
    
    # Additional polish with COBYLA (derivative-free, good for final refinement)
    if time.time() < deadline - 5:
        try:
            res3 = opt.minimize(compute_bound, best_coeffs, method='COBYLA',
                               options={'maxiter': 1000, 'tol': 1e-12})
            if np.isfinite(res3.fun) and res3.fun < best_score:
                best_score = res3.fun
                best_coeffs = res3.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish with Powell (derivative-free)
    if time.time() < deadline - 2:
        try:
            res4 = opt.minimize(compute_bound, best_coeffs, method='Powell',
                               options={'maxiter': 500, 'xtol': 1e-14, 'ftol': 1e-16})
            if np.isfinite(res4.fun) and res4.fun < best_score:
                best_score = res4.fun
                best_coeffs = res4.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
