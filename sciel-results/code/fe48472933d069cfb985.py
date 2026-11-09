import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv
import math

def _compute_y_values():
    """
    Precompute the values of the 21 real orthonormal spherical harmonics
    for degree l=10 at the quadrature points.
    Returns:
        Y: array of shape (N_phi, N_theta, 21)
        weights: array of shape (N_theta, N_phi) or flattened
    """
    l = 10
    N_theta = 70
    N_phi = 130
    
    # Gauss-Legendre nodes and weights for cos(theta)
    # x = cos(theta), dx = -sin(theta) dtheta
    # Integral over sphere: ∫_0^π ∫_0^{2π} f(θ,φ) sin(θ) dφ dθ
    # = ∫_{-1}^1 ∫_0^{2π} f(arccos(x), φ) dφ dx
    x, w_gl = np.polynomial.legendre.leggauss(N_theta)
    
    # Phi points
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    
    # Create grids
    theta = np.arccos(x)
    # Meshgrid for broadcasting: (N_phi, N_theta)
    Phi, Theta = np.meshgrid(phi, theta, indexing='ij')
    
    # Weights for the double integral
    # dOmega = dphi * dx
    # dx is handled by w_gl, dphi is 2*pi/N_phi
    # Total weight for point (i, j) is w_gl[j] * (2*pi/N_phi)
    weights = np.outer(np.ones(N_phi), w_gl) * (2 * np.pi / N_phi)
    
    # Compute harmonics
    # Y_0 = N_0 P_10^0
    # Y_m (m>0) = sqrt(2) N_m P_10^m cos(m phi)
    # Y_-m = sqrt(2) N_m P_10^m sin(m phi)  (Note: standard real basis often has sin for negative m)
    # The problem statement says:
    # Y_m = sqrt(2) N_m P^m cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P^|m| sin(|m| phi) for m < 0
    
    Y = np.zeros((N_phi, N_theta, 21), dtype=np.float64)
    
    for m_idx, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        # Condon-Shortley phase is included in lpmv
        # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
        # Using log-gamma for stability
        log_N = 0.5 * (np.log(21.0) - np.log(4.0 * np.pi) + math.lgamma(11 - abs_m) - math.lgamma(11 + abs_m))
        N_m = np.exp(log_N)
        
        # P_10^m(cos theta)
        # lpmv(m, l, x) computes P_l^m(x) with Condon-Shortley phase
        P_lm = lpmv(abs_m, l, x)
        
        if m == 0:
            Y[:, :, m_idx] = N_m * P_lm
        elif m > 0:
            Y[:, :, m_idx] = np.sqrt(2.0) * N_m * P_lm * np.cos(m * Phi)
        else: # m < 0
            Y[:, :, m_idx] = np.sqrt(2.0) * N_m * P_lm * np.sin(abs_m * Phi)
            
    return Y, weights

# Precompute
Y_vals, weights = _compute_y_values()
# Flatten Y_vals and weights for matrix multiplication
Y_flat = Y_vals.flatten() # (N_points, 21)
w_flat = weights.flatten() # (N_points,)

def objective(c):
    """
    Compute the L4/L2 ratio for coefficients c.
    """
    # f = sum c_m Y_m
    f_vals = Y_flat @ c
    
    # L4 norm: (∫ |f|^4 dOmega)^(1/4)
    # ∫ |f|^4 ≈ sum w_i * f_i^4
    integral_l4 = np.sum(w_flat * (f_vals ** 4))
    l4_norm = integral_l4 ** 0.25
    
    # L2 norm: (∫ |f|^2 dOmega)^(1/2) = sqrt(sum c_m^2) due to orthonormality
    l2_norm = np.sqrt(np.dot(c, c))
    
    if l2_norm == 0:
        return 1e10
        
    return l4_norm / l2_norm

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    """
    k = 10
    n_coeffs = 2 * k + 1
    
    # We use multiple random restarts to find a good local minimum.
    # The problem is non-convex, so global optimization is hard,
    # but random restarts with a good local optimizer work well.
    
    best_c = None
    best_val = np.inf
    
    # Set random seed for reproducibility in the function context if needed, 
    # but usually we want exploration.
    rng = np.random.default_rng(42)
    
    n_restarts = 50
    
    for i in range(n_restarts):
        # Random initial guess
        # It helps to scale the initial guess, but since the objective is scale invariant, 
        # the direction matters.
        c0 = rng.normal(size=n_coeffs)
        
        # Normalize c0 to unit length to start on the sphere
        c0 /= np.linalg.norm(c0)
        
        try:
            res = minimize(
                objective, 
                c0, 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best_c = res.x
        except Exception:
            pass
            
    if best_c is None:
        # Fallback to seed if optimization fails completely
        return np.ones(n_coeffs, dtype=np.float64)
        
    return best_c.astype(np.float64)

# EVOLVE-BLOCK-END
