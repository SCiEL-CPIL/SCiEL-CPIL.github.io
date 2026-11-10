import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize
import time

# Precomputation
L = 10
N_GL = 70
N_PHI = 130

# Gauss-Legendre nodes and weights in x = cos(theta)
x_gl, w_gl = np.polynomial.legendre.leggauss(N_GL)
# Theta is arccos(x)
theta_gl = np.arccos(x_gl)
sin_theta_gl = np.sqrt(1 - x_gl**2)

# Phi points
phi_pts = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)

# Create meshgrid for all quadrature points
# We need to evaluate Y_m at all (theta_i, phi_j)
# Total points = 70 * 130 = 9100

# Precompute associated Legendre polynomials P_L^m(cos theta) for all m
# lpmv(m, l, x) includes Condon-Shortley phase
# We need P_10^m for m=0..10
# Note: lpmv takes m, l, x
P_vals = np.zeros((N_GL, L + 1))
for m in range(L + 1):
    P_vals[:, m] = lpmv(m, L, x_gl)

# Normalization constants N_m
# N_m = sqrt( (2L+1)/(4*pi) * (L-m)! / (L+m)! )
# Factorials can be large, use log-gamma or just math.factorial since L=10 is small
import math
N_m = np.zeros(L + 1)
for m in range(L + 1):
    N_m[m] = np.sqrt((2 * L + 1) / (4 * np.pi) * math.factorial(L - m) / math.factorial(L + m))

# Construct the matrix M of size (N_GL * N_PHI, 2L+1)
# m index in vector c: 0..20 corresponds to m_phys = -10..10
# c[i] belongs to m = i - 10
N_COEFF = 2 * L + 1
M = np.zeros((N_GL * N_PHI, N_COEFF))

# Weights for quadrature
# w_total[i, j] = w_gl[i] * (2*pi / N_PHI)
w_phi = 2 * np.pi / N_PHI
# We can just use 1D weights for the flattened array
weights = np.repeat(w_gl * w_phi, N_PHI)

# Fill M
# For each quadrature point (i, j)
# theta = theta_gl[i]
# phi = phi_pts[j]
# For each m in -10..10
#   if m == 0: Y = N_0 * P_10^0(cos theta)
#   if m > 0: Y = sqrt(2) * N_m * P_10^m(cos theta) * cos(m * phi)
#   if m < 0: Y = sqrt(2) * N_|m| * P_10^{|m|}(cos theta) * sin(|m| * phi)

# To speed up, we can vectorize over phi for each theta
# But N_GL=70 is small, so loop over i is fine.

for i in range(N_GL):
    # P_vals[i, m] is P_10^m(x_gl[i])
    # sin_theta is not needed for P, but phi terms need cos/sin
    row_start = i * N_PHI
    row_end = (i + 1) * N_PHI
    
    # m = 0
    m_idx = L # index 10 in M corresponds to m=0
    M[row_start:row_end, m_idx] = N_m[0] * P_vals[i, 0]
    
    # m > 0
    for m in range(1, L + 1):
        # m positive
        m_idx_pos = L + m
        # m negative
        m_idx_neg = L - m
        
        cos_term = np.cos(m * phi_pts)
        sin_term = np.sin(m * phi_pts)
        
        factor = np.sqrt(2) * N_m[m] * P_vals[i, m]
        
        M[row_start:row_end, m_idx_pos] = factor * cos_term
        M[row_start:row_end, m_idx_neg] = factor * sin_term

# Precompute M^T W for gradients?
# grad_A = 2 * M^T (w * f)
# grad_B = 4 * M^T (w * f^3)
# We can precompute nothing specific, just use M.T.dot(...)

def compute_ratio_and_grad(c):
    """
    Computes R = ||f||_4 / ||f||_2 and its gradient.
    """
    f = M.dot(c)
    
    # Norms
    # ||f||_2^2 = sum w f^2
    A = np.dot(weights, f**2)
    # ||f||_4^2 = sum w f^4
    B = np.dot(weights, f**4)
    
    if A < 1e-300:
        return 1e10, np.zeros_like(c)
        
    R = np.sqrt(B / A)
    
    # Gradients
    # dA/dc = 2 * M^T (w * f)
    grad_A = 2.0 * M.T.dot(weights * f)
    
    # dB/dc = 4 * M^T (w * f^3)
    grad_B = 4.0 * M.T.dot(weights * f**3)
    
    # dR/dc = (A * grad_B - B * grad_A) / (2 * R * A^2)
    grad_R = (A * grad_B - B * grad_A) / (2.0 * R * A**2)
    
    return R, grad_R

def rudin_problem10() -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    time_limit = 600.0
    
    best_c = np.ones(21) / np.sqrt(21)
    best_R, _ = compute_ratio_and_grad(best_c)
    
    # List of initializations
    # 1. All ones (normalized)
    inits = []
    inits.append(np.ones(21) / np.sqrt(21))
    
    # 2. Random restarts
    # We will generate them on the fly
    
    # Optimization settings
    # Using SLSQP with constraint c.T c = 1
    # Constraint function: c.T c - 1 = 0
    # Gradient of constraint: 2c
    
    def constraint_fun(c):
        return c.dot(c) - 1.0
        
    def constraint_grad(c):
        return 2.0 * c
        
    constraints = ({'type': 'eq', 'fun': constraint_fun, 'jac': constraint_grad})
    
    restart_count = 0
    max_restarts = 500 # Adjust based on time
    
    while time.time() - start_time < time_limit:
        restart_count += 1
        
        # Choose initialization
        if restart_count == 1:
            c0 = inits[0]
        else:
            # Random Gaussian
            c0 = np.random.normal(size=21)
            norm = np.linalg.norm(c0)
            if norm < 1e-12:
                continue
            c0 = c0 / norm
            
        # Run optimization
        try:
            res = minimize(
                compute_ratio_and_grad, 
                c0, 
                method='SLSQP', 
                jac=True, 
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.success or res.fun < best_R:
                c_opt = res.x
                R_opt = res.fun
                # Check validity
                if np.isfinite(R_opt) and R_opt < best_R:
                    best_R = R_opt
                    best_c = c_opt
        except Exception:
            pass
            
        # If time is running out, break
        if time.time() - start_time > time_limit - 10:
            break

    return best_c
