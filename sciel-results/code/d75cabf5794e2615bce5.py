import numpy as np
import time

def rudin_problem10() -> np.ndarray:
    """
    Finds the coefficients for the flattest degree-10 spherical harmonic
    by minimizing the L4/L2 norm ratio.
    """
    start_time = time.time()
    deadline = 580  # seconds, safety margin from 600s limit
    
    k = 10
    num_m = 2 * k + 1
    
    # Quadrature setup
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights for cos(theta) in [-1, 1]
    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    # nodes are u_i = cos(theta_i)
    # weights are w_i for integral over [-1, 1]
    
    # phi nodes
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi
    
    # Create meshgrid for u and phi
    U, Phi = np.meshgrid(nodes, phi, indexing='ij')
    # U shape: (n_theta, n_phi)
    # Phi shape: (n_theta, n_phi)
    
    # Weights for the 2D quadrature
    # Integral approx: sum_i sum_j w_i * (d_phi / 2pi * 2pi? No, standard surface measure)
    # dOmega = dphi du.
    # Integral = sum_i sum_j w_i * d_phi * f(u_i, phi_j)
    W_1d = weights * d_phi
    W_2d = np.outer(W_1d, np.ones(n_phi))
    # Total weight sum should be 4*pi
    # sum(W_2d) = sum(weights) * d_phi * n_phi = 2 * 2pi = 4pi. Correct.
    
    # Compute Spherical Harmonics Y_m on the grid
    # Y_m shape: (n_theta, n_phi, 21) or (21, n_theta, n_phi)
    # Let's use shape (21, n_theta, n_phi) for easier matrix mult
    
    # Factorials for normalization
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    # We need factorials up to 20
    fact = np.zeros(21)
    fact[0] = 1.0
    for i in range(1, 21):
        fact[i] = fact[i-1] * i
        
    Y = np.zeros((num_m, n_theta, n_phi), dtype=np.float64)
    
    # Legendre polynomials P_l^m(u)
    # We can compute them using scipy.special.lpmv or implement recurrence.
    # Since SciPy is available, use scipy.special.lpmv
    from scipy.special import lpmv
    
    for idx, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        # Normalization
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * fact[10 - abs_m] / fact[10 + abs_m])
        
        # Associated Legendre function P_10^abs_m(u)
        # lpmv(m, l, x) computes P_l^m(x)
        P_lm = lpmv(abs_m, 10, U)
        
        if m == 0:
            Y[idx] = N_m * P_lm
        elif m > 0:
            Y[idx] = np.sqrt(2.0) * N_m * P_lm * np.cos(m * Phi)
        else:
            Y[idx] = np.sqrt(2.0) * N_m * P_lm * np.sin(abs_m * Phi)
            
    # Flatten Y for matrix operations
    # Y_flat shape: (num_m, N_points)
    Y_flat = Y.reshape(num_m, -1)
    # W_flat shape: (N_points,)
    W_flat = W_2d.flatten()
    
    # Precompute Y^T * W (for the linear part if we were doing L2, but we do cubic)
    # We need to compute v = Y^T * (W * f^3)
    # Let's define an operator apply_T(c):
    #   f = Y_flat @ c
    #   f3 = f**3
    #   v = Y_flat.T @ (W_flat * f3)
    
    def apply_T(c):
        f = Y_flat @ c
        f3 = f ** 3
        v = Y_flat.T @ (W_flat * f3)
        return v

    best_c = None
    best_score = np.inf
    
    # Run multiple restarts
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize c
        if r == 0:
            c = np.ones(num_m)
        else:
            c = np.random.randn(num_m)
            
        # Normalize
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            continue
        c = c / c_norm
        
        # Iterate
        prev_c = c.copy()
        
        for it in range(2000):
            if time.time() > deadline:
                break
                
            v = apply_T(c)
            
            # Check for convergence
            # If v is close to c (up to sign), we converged
            # v should be parallel to c for the eigenvector
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:
                break
            v = v / v_norm
            
            # Handle sign ambiguity: keep c and v aligned
            dot = np.dot(c, v)
            if dot < 0:
                v = -v
                
            c = v
            
            # Check convergence
            if np.linalg.norm(c - prev_c) < 1e-10:
                break
            prev_c = c.copy()
            
        # Evaluate score for this candidate
        f = Y_flat @ c
        norm2_sq = np.sum(c**2) # Should be 1
        norm4_sq = np.sum(W_flat * f**4)
        
        # R = ||f||_4 / ||f||_2
        # R^2 = ||f||_4^2 / ||f||_2^2
        # ||f||_4^2 = (Integral f^4)^0.5 ? No.
        # ||f||_4 = (Integral |f|^4)^0.25
        # ||f||_4^2 = (Integral |f|^4)^0.5
        # ||f||_2 = (Integral |f|^2)^0.5
        # R = (Integral f^4)^0.25 / (Integral f^2)^0.5
        # R^2 = (Integral f^4)^0.5 / Integral f^2
        
        # The task says: "The evaluator measures the raw value (higher is better)."
        # "R = ||f||_4 / ||f||_2 ... value is proportional to 1/R, so a lower R is better."
        # Wait, "search score is value/seed".
        # If value is 1/R, then higher value is better.
        # So we want to MAXIMIZE 1/R, which is MINIMIZING R.
        
        integral_f4 = np.sum(W_flat * f**4)
        integral_f2 = np.sum(W_flat * f**2)
        
        # Check validity
        if not (np.isfinite(integral_f4) and np.isfinite(integral_f2)):
            continue
        if integral_f2 < 1e-15:
            continue
            
        R_sq = np.sqrt(integral_f4) / integral_f2
        R = np.sqrt(R_sq)
        
        score = 1.0 / R
        
        if score > best_score:
            best_score = score
            best_c = c.copy()
            
    if best_c is None:
        # Fallback to all ones
        best_c = np.ones(num_m)
        
    # Final check on validity constraints for best_c
    # max|c| < 1e-3 or > 1e5 is invalid
    # Our c is normalized to unit L2 norm, so max|c| <= 1.
    # So it's always valid in that range.
    
    return best_c

# EVOLVE-BLOCK-END
