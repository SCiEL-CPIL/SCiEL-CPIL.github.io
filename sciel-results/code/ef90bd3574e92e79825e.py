import numpy as np
from scipy.special import factorial
from scipy.optimize import minimize, differential_evolution
import time

def rudin_problem10():
    start_time = time.time()
    deadline = start_time + 590  # 10s margin before 600s kill
    
    # Setup quadrature: 70-point Gauss-Legendre in cos(theta), 130 equally spaced phi
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights in cos(theta)
    ct_nodes, ct_weights = np.polynomial.legendre.leggauss(n_theta)
    # Map from [-1,1] to theta: cos(theta) = ct_nodes, sin(theta) dtheta = -d(ct)
    # Surface element: dOmega = sin(theta) dtheta dphi = -d(ct) dphi
    # So integral over sphere = sum_i sum_j w_i * f(ct_i, phi_j) * dphi
    # where w_i are the GL weights (which integrate to 2 over [-1,1])
    # Total measure: sum_i w_i * n_phi * (2*pi/n_phi) = 2 * 2*pi = 4*pi. Correct.
    
    phi_nodes = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # 3D grid
    ct_grid, phi_grid = np.meshgrid(ct_nodes, phi_nodes, indexing='ij')
    # ct_grid: (n_theta, n_phi), phi_grid: (n_theta, n_phi)
    
    # Precompute sin(theta) = sqrt(1 - cos^2(theta))
    st_grid = np.sqrt(1.0 - ct_grid**2)
    
    # Compute real orthonormal spherical harmonics Y_m for m = -10..10
    # m index: c[i] corresponds to m = i - 10
    L = 10
    ms = np.arange(-L, L + 1)  # -10, -9, ..., 9, 10
    n_m = len(ms)  # 21
    
    # Build Y_m grid: shape (n_theta, n_phi, n_m)
    Y = np.zeros((n_theta, n_phi, n_m))
    
    # Need associated Legendre functions P_L^m(x) with Condon-Shortley phase
    # Use scipy.special.lpmv for associated Legendre functions
    from scipy.special import lpmv
    
    for idx, m in enumerate(ms):
        abs_m = abs(m)
        # Normalization: N_m = sqrt(21/(4*pi) * (L-m)!/(L+m)!)
        # Note: 2L+1 = 21
        log_N = 0.5 * (np.log(2 * L + 1) - np.log(4 * np.pi) + 
                      np.math.lgamma(L - abs_m + 1) - np.math.lgamma(L + abs_m + 1))
        N_m = np.exp(log_N)
        
        # P_L^{abs_m}(cos theta) with Condon-Shortley phase (lpmv includes it)
        P_lm = lpmv(abs_m, L, ct_grid)  # shape (n_theta, n_phi)
        
        if m == 0:
            Y[:, :, idx] = N_m * P_lm
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2) * N_m * P_lm * np.cos(m * phi_grid)
        else:  # m < 0
            Y[:, :, idx] = np.sqrt(2) * N_m * P_lm * np.sin(abs_m * phi_grid)
    
    # Flatten for easy computation
    Y_flat = Y.reshape(-1, n_m)  # (n_theta * n_phi, n_m)
    n_quad = n_theta * n_phi
    
    # Weights for quadrature: w_i * dphi for each point
    # ct_weights has shape (n_theta,), each weight w_i corresponds to cos(theta_i)
    # For each theta_i, there are n_phi phi points, each with weight dphi
    # So the total weight for point (i, j) is ct_weights[i] * dphi
    w_theta = np.repeat(ct_weights, n_phi)  # shape (n_theta * n_phi,)
    w_flat = w_theta * dphi  # shape (n_quad,)
    
    # Verify total weight: sum(w_flat) should be 4*pi
    # assert abs(np.sum(w_flat) - 4 * np.pi) < 1e-10
    
    def compute_R(c):
        """Compute R = ||f||_4 / ||f||_2 where f = sum_m c_m Y_m"""
        # f at each quadrature point: shape (n_quad,)
        f_vals = Y_flat @ c  # (n_quad,)
        
        # ||f||_2^2 = integral |f|^2 dOmega = sum w_j * f_j^2
        f2 = f_vals ** 2
        norm2_sq = np.dot(w_flat, f2)
        
        # ||f||_4^4 = integral |f|^4 dOmega = sum w_j * f_j^4
        f4 = f_vals ** 4
        norm4_4 = np.dot(w_flat, f4)
        
        if norm2_sq <= 0 or norm4_4 <= 0:
            return np.inf
        
        norm2 = np.sqrt(norm2_sq)
        norm4 = np.sqrt(np.sqrt(norm4_4))  # ||f||_4 = (integral |f|^4)^{1/4}
        
        return norm4 / norm2
    
    def objective(c):
        return compute_R(c)
    
    def objective_neg(c):
        return -compute_R(c)  # for minimization
    
    # Initial best
    best_c = None
    best_R = np.inf
    
    # Try several random initializations
    n_starts = 30
    rng = np.random.default_rng(42)
    
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        # Random initialization with max|c| around 1
        c0 = rng.normal(0, 1, n_m)
        # Normalize so max|c| = 1
        max_c = np.max(np.abs(c0))
        if max_c > 1e-10:
            c0 = c0 / max_c
        
        R0 = compute_R(c0)
        if R0 < best_R:
            best_R = R0
            best_c = c0.copy()
        
        # Local optimization using L-BFGS-B
        try:
            res = minimize(objective, c0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success or res.fun < best_R:
                R_opt = res.fun
                if R_opt < best_R:
                    best_R = R_opt
                    best_c = res.x.copy()
        except Exception:
            pass
    
    # If we have time, do a differential evolution global search
    if time.time() < deadline - 120:
        try:
            bounds = [(-10, 10)] * n_m
            
            def de_obj(c):
                if time.time() > deadline - 60:
                    return 1e10
                return compute_R(c)
            
            de_res = differential_evolution(de_obj, bounds, maxiter=50, 
                                            popsize=15, tol=1e-6, 
                                            seed=42, polish=False)
            if de_res.fun < best_R:
                best_R = de_res.fun
                best_c = de_res.x.copy()
                
                # Polish with L-BFGS-B
                res2 = minimize(objective, best_c, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                if res2.fun < best_R:
                    best_R = res2.fun
                    best_c = res2.x.copy()
        except Exception:
            pass
    
    # Final validation
    if best_c is None:
        # Fallback: all zeros is invalid, use a simple valid vector
        best_c = np.zeros(n_m)
        best_c[10] = 1.0  # m=10 component
    
    # Ensure validity: max|c| >= 1e-3 and <= 1e5
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3:
        best_c = best_c / max_c * 1.0
    elif max_c > 1e5:
        best_c = best_c / max_c * 1e5
    
    # Check for NaN/inf
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(n_m)
    
    return best_c
