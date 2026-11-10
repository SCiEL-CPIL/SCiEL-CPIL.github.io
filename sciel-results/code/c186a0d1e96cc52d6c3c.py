import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv
import math

def rudin_problem10() -> np.ndarray:
    k = 10
    # Quadrature setup
    # 70 Gauss-Legendre points for cos(theta) in [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    # 130 equally spaced phi in [0, 2*pi)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Weights
    # Integral over sphere: int_{-1}^1 int_0^{2pi} ... dphi dx
    # dphi weight is 2*pi / n_phi
    # dx weight is w_gl[i]
    # Total weight W[i, j] = w_gl[i] * (2*pi / n_phi)
    w_phi = 2 * np.pi / n_phi
    weights_2d = np.outer(w_gl, np.full(n_phi, w_phi))
    
    # Compute Spherical Harmonics Y_m(theta, phi)
    # m ranges from -10 to 10
    # Y_0 = N_0 P_10^0(cos theta)
    # Y_m (m>0) = sqrt(2) N_m P_10^m(cos theta) cos(m phi)
    # Y_m (m<0) = sqrt(2) N_|m| P_10^{|m|}(cos theta) sin(|m| phi)
    
    # Precompute Y_m for all points
    # Shape: (70, 130, 21)
    # We can flatten (70, 130) to (9100,)
    n_theta = 70
    total_points = n_theta * n_phi
    
    # We need P_10^m(x) for m=0..10
    # lpmv(m, l, x) gives associated Legendre P_l^m(x) with Condon-Shortley phase
    # N_m = sqrt((2l+1)/(4*pi) * (l-m)!/(l+m)!)
    
    Y_vals = np.zeros((total_points, 21), dtype=np.float64)
    
    # Map m index to array index
    # c[i] corresponds to m = i - 10
    # So index 0 is m=-10, index 10 is m=0, index 20 is m=10
    
    # Precompute factorials for N_m
    # N_m depends on |m|
    N = np.zeros(11)
    for m_abs in range(11):
        # (10-m_abs)! / (10+m_abs)!
        # Use log gamma to avoid overflow, though 10! is small
        log_num = math.lgamma(11 - m_abs)
        log_den = math.lgamma(11 + m_abs)
        N[m_abs] = math.sqrt(21.0 / (4.0 * np.pi) * math.exp(log_num - log_den))
    
    # Fill Y_vals
    # x_gl is cos(theta)
    # phi is phi
    # We need to evaluate for each (i, j)
    
    # Vectorized evaluation
    # x_gl shape (70,), phi shape (130,)
    # We need P_10^m(x_gl) for each m. Shape (70,)
    P_vals = np.zeros((11, 70), dtype=np.float64)
    for m_abs in range(11):
        P_vals[m_abs, :] = lpmv(m_abs, 10, x_gl)
        
    # Now for each m in -10..10:
    # m=0: Y = N[0] * P[0]
    # m>0: Y = sqrt(2) * N[m] * P[m] * cos(m*phi)
    # m<0: Y = sqrt(2) * N[|m|] * P[|m|] * sin(|m|*phi)
    
    # Let's construct the matrix H directly.
    # H[p, idx] = Y_{m_idx}(theta_i, phi_j) where p = i*n_phi + j
    
    sqrt2 = math.sqrt(2.0)
    
    for i in range(n_theta):
        x_i = x_gl[i]
        for j in range(n_phi):
            phi_j = phi[j]
            p = i * n_phi + j
            
            # m=0 (idx 10)
            Y_vals[p, 10] = N[0] * P_vals[0, i]
            
            # m > 0
            for m in range(1, 11):
                # idx for m
                idx_pos = 10 + m
                Y_vals[p, idx_pos] = sqrt2 * N[m] * P_vals[m, i] * np.cos(m * phi_j)
                
                # idx for -m
                idx_neg = 10 - m
                Y_vals[p, idx_neg] = sqrt2 * N[m] * P_vals[m, i] * np.sin(m * phi_j)
                
    # Weights for each point p
    # weights_2d is (70, 130)
    W_flat = weights_2d.flatten()
    
    # Objective function: minimize ||f||_4^4 = sum W * f^4
    # Subject to ||c||_2 = 1
    
    def objective(c):
        f = Y_vals @ c
        f4 = f**4
        return np.sum(W_flat * f4)
        
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0}]
    
    # Initial guesses
    best_c = np.ones(21)
    best_val = objective(best_c)
    
    # Try a few starting points
    starts = [
        np.ones(21),
        np.array([1.0 if m == 0 else 0.0 for m in range(-10, 11)]), # Zonal
        np.array([1.0 if m == 10 else 0.0 for m in range(-10, 11)]), # Equatorial
        np.random.randn(21)
    ]
    
    # Normalize starts
    for i in range(len(starts)):
        starts[i] /= np.linalg.norm(starts[i])
        
    # Run optimization for each start
    for start in starts:
        try:
            res = minimize(objective, start, method='SLSQP', constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success:
                c_opt = res.x
                val = objective(c_opt)
                if val < best_val:
                    best_val = val
                    best_c = c_opt
        except Exception:
            pass
            
    # Ensure the result is valid and normalized
    best_c = best_c / np.linalg.norm(best_c)
    
    return best_c
