import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def _get_harmonics_matrix():
    k = 10
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights
    # cos(theta) in [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # theta = arccos(x)
    theta = np.arccos(x)
    
    # phi points
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Meshgrid
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    
    # Weights for quadrature
    # Integral = sum_{i,j} W_{i,j} f(theta_i, phi_j)
    # W_{i,j} = (2*pi/n_phi) * w_i
    # Note: leggauss weights sum to 2.
    # Total weight sum = (2*pi/n_phi) * n_phi * sum(w_i) = 2*pi * 2 = 4*pi. Correct.
    W = (2 * np.pi / n_phi) * w[:, np.newaxis]
    W = np.repeat(W, n_phi, axis=1)
    W = W.flatten()
    
    # Build harmonics matrix M of shape (21, n_points)
    # m = -10 .. 10
    # c[i] corresponds to m = i - 10
    n_pts = n_theta * n_phi
    M = np.zeros((2 * k + 1, n_pts), dtype=np.float64)
    
    cos_theta = x[:, np.newaxis] # shape (70, 1)
    cos_theta_flat = np.repeat(cos_theta, n_phi, axis=1).flatten()
    
    for i in range(2 * k + 1):
        m = i - k
        abs_m = abs(m)
        
        # Normalization N_m
        # N_m = sqrt( (2k+1)/(4*pi) * (k-m)!/(k+m)! )
        # Note: for m=0, the formula gives N_0.
        # For m != 0, the real harmonics are sqrt(2) * N_m * P_k^m * cos/sin.
        # The problem statement says:
        # Y_0 = N_0 P^0
        # Y_m = sqrt(2) N_m P^m cos(m phi) for m > 0
        # Y_m = sqrt(2) N_|m| P^|m| sin(|m| phi) for m < 0
        # Where N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
        
        N_m = np.sqrt( (2 * k + 1) / (4 * np.pi) * factorial(k - abs_m) / factorial(k + abs_m) )
        
        # Associated Legendre P_k^m(cos theta)
        # lpmv(m, k, x) includes Condon-Shortley phase
        P_val = lpmv(abs_m, k, x) # shape (70,)
        P_val_flat = np.repeat(P_val, n_phi, axis=0).flatten()
        
        if m == 0:
            Y_val = N_m * P_val_flat
        elif m > 0:
            Y_val = np.sqrt(2) * N_m * P_val_flat * np.cos(m * Phi.flatten())
        else:
            Y_val = np.sqrt(2) * N_m * P_val_flat * np.sin(abs_m * Phi.flatten())
            
        M[i, :] = Y_val
        
    return M, W

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600 # 600s limit, leave margin
    
    M, W = _get_harmonics_matrix()
    n_coeffs = 21
    n_pts = M.shape[1]
    
    # Precompute W * M for efficiency?
    # We need sum W * f^4 and sum W * f^2
    # f = M.T @ c
    
    best_c = None
    best_R = np.inf
    
    # Define objective and gradient
    def objective_and_grad(c):
        f = M.T @ c
        f2 = f * f
        f4 = f2 * f2
        
        norm2_sq = np.sum(W * f2)
        norm4_4 = np.sum(W * f4)
        
        if norm2_sq <= 0:
            return 0, np.zeros_like(c)
            
        R = (norm4_4 ** 0.25) / (norm2_sq ** 0.5)
        
        # Gradient of R
        # R = A^0.25 B^-0.5
        # dR/dc = 0.25 A^-0.75 B^-0.5 dA/dc - 0.5 A^0.25 B^-1.5 dB/dc
        # dA/dc = 4 sum W f^3 M
        # dB/dc = 2 sum W f M
        
        # u = sum W f^3 M  (vector)
        # v = sum W f M  (vector)
        # Note: M is (21, N). W is (N,).
        # f is (N,).
        # u_m = sum_j W_j f_j^3 M_{m,j}
        # u = M @ (W * f^3)
        
        u = M @ (W * (f ** 3))
        v = M @ (W * f)
        
        grad_R = (0.25 * (norm4_4 ** -0.75) * (norm2_sq ** -0.5) * 4 * u) - \
                 (0.5 * (norm4_4 ** 0.25) * (norm2_sq ** -1.5) * 2 * v)
        
        return R, grad_R

    # Optimization strategy:
    # 1. Try zonal harmonic (c_0 = 1, others 0)
    # 2. Try random restarts
    
    # Zonal
    c0 = np.zeros(n_coeffs)
    c0[10] = 1.0 # m=0 is index 10
    R0, _ = objective_and_grad(c0)
    best_R = R0
    best_c = c0.copy()
    
    # Random restarts
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # Random initial guess
        c_init = np.random.randn(n_coeffs)
        c_init /= np.linalg.norm(c_init)
        
        # Use L-BFGS-B
        # Since the objective is scale invariant (R(kc) = R(c)), 
        # we can minimize R(c) directly. 
        # However, L-BFGS-B might struggle with the scale invariance (flat directions).
        # It is better to constrain ||c||=1.
        # We can use a penalty or just normalize at each step?
        # scipy.optimize.minimize doesn't easily handle equality constraints for L-BFGS-B.
        # We can use SLSQP with constraint ||c||^2 - 1 = 0.
        
        # Alternatively, we can minimize R(c) and hope it converges to a normalized solution.
        # Or, we can minimize log(R(c)) to improve conditioning?
        
        # Let's try SLSQP with constraint.
        constraints = [{'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0}]
        
        res = minimize(objective_and_grad[0] if False else lambda c: objective_and_grad(c)[0], 
                       c_init, 
                       jac=lambda c: objective_and_grad(c)[1], 
                       method='SLSQP', 
                       constraints=constraints, 
                       options={'maxiter': 500, 'ftol': 1e-12})
        
        if res.success:
            R_res = res.fun
            if R_res < best_R:
                best_R = R_res
                best_c = res.x
        else:
            # Check if the final point is better
            R_res, _ = objective_and_grad(res.x)
            if R_res < best_R:
                best_R = R_res
                best_c = res.x

    # Final polish with higher precision?
    # We can do a few steps of gradient descent on the sphere manually.
    # But SLSQP should be good.
    
    # Ensure output is valid
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        best_c = np.zeros(21)
        best_c[10] = 1.0
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c[10] = 1.0
    elif max_abs > 1e5:
        best_c = best_c / max_abs * 1.0

    return best_c
