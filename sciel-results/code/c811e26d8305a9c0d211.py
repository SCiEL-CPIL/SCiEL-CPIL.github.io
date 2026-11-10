import numpy as np
import time
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c for degree-10 spherical harmonics minimizing ||f||_4 / ||f||_2.
    
    Strategy:
    1. Precompute Spherical Harmonic matrix Y at quadrature points.
    2. Use a projected Newton method with a BFGS Hessian approximation 
       constrained to the unit sphere (||c||_2 = 1).
    3. Multiple restarts from diverse initializations.
    4. The constraint ||c||_2 = 1 makes the problem well-posed and the 
       L4 norm is strictly convex on this sphere for degree 10.
    """
    k = 10
    n_coeffs = 2 * k + 1  # 21
    
    start_time = time.time()
    deadline = start_time + 590
    
    # --- 1. Precompute Spherical Harmonics ---
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    x_grid, phi_grid = np.meshgrid(x_gl, phi_vals, indexing='ij')
    x_flat = x_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    # Effective weights
    W_eff = np.repeat(w_gl, n_phi) * (2.0 * np.pi / n_phi)
    
    # Normalization constants
    N = np.zeros(n_coeffs)
    for m in range(0, k + 1):
        norm_factor = (2 * k + 1) / (4.0 * np.pi) * factorial(k - m) / factorial(k + m)
        N[m] = np.sqrt(norm_factor)
        if m > 0:
            N[-m] = N[m]
            
    # Build Y matrix
    n_points = len(x_flat)
    Y_matrix = np.zeros((n_points, n_coeffs))
    
    for m_idx, m in enumerate(range(-k, k + 1)):
        if m == 0:
            P_val = lpmv(0, k, x_flat)
            Y_matrix[:, m_idx] = N[0] * P_val
        elif m > 0:
            P_val = lpmv(m, k, x_flat)
            Y_matrix[:, m_idx] = np.sqrt(2) * N[m] * P_val * np.cos(m * phi_flat)
        else:
            abs_m = -m
            P_val = lpmv(abs_m, k, x_flat)
            Y_matrix[:, m_idx] = np.sqrt(2) * N[abs_m] * P_val * np.sin(abs_m * phi_flat)
            
    # Precompute useful matrices
    # Y^T W is (21, 9100) -> (21, 21) for gradient computation
    YT_W = Y_matrix.T @ (W_eff[:, None] * Y_matrix)  # (21, 21)
    
    # --- 2. Define Objective and Gradient on the sphere ---
    def objective_sphere(c):
        """Objective: ||Yc||_4^4 / ||Yc||_2^4, but we fix ||c||_2 = 1 so ||Yc||_2^2 = 1"""
        f = Y_matrix @ c
        f2 = f * f
        D_val = np.dot(W_eff, f2)  # ||f||_2^2 = 1 if c is on sphere
        N_val = np.dot(W_eff, f2 * f2)  # ||f||_4^4
        
        # Return the ratio (should be close to N_val if D_val ≈ 1)
        val = N_val / (D_val * D_val)
        return val
    
    def gradient_sphere(c):
        """Gradient of the objective on the sphere."""
        f = Y_matrix @ c
        f2 = f * f
        D_val = np.dot(W_eff, f2)
        N_val = np.dot(W_eff, f2 * f2)
        
        # Gradient of N_val = 4 * Y^T (W * f^3)
        f3 = f2 * f
        grad_N = 4.0 * (Y_matrix.T @ (W_eff * f3))
        
        # Gradient of D_val = 2 * Y^T (W * f)
        grad_D = 2.0 * (Y_matrix.T @ (W_eff * f))
        
        # val = N / D^2
        # dval/dc = (dN D - 2 N dD) / D^3
        grad = (grad_N * D_val - 2.0 * N_val * grad_D) / (D_val ** 3)
        
        return grad
    
    def hessian_approx(c, B):
        """
        Approximate Hessian using BFGS update.
        B is the current inverse Hessian approximation.
        """
        f = Y_matrix @ c
        f2 = f * f
        D_val = np.dot(W_eff, f2)
        N_val = np.dot(W_eff, f2 * f2)
        
        f3 = f2 * f
        f4 = f2 * f2
        
        # Second derivatives
        # d2N/dc2 = 4 * Y^T diag(W * 3*f^2) Y
        # d2D/dc2 = 2 * Y^T diag(W) Y
        
        # For BFGS, we need the gradient difference
        return None  # We'll use scipy's BFGS with projection
    
    # --- 3. Projected Newton / BFGS on the sphere ---
    def project_to_sphere(c):
        """Project c onto the unit sphere."""
        norm = np.linalg.norm(c)
        if norm < 1e-15:
            return c
        return c / norm
    
    def projected_bfgs(x0, max_iter=5000, tol=1e-12):
        """
        BFGS optimization projected onto the unit sphere.
        """
        x = x0.copy()
        x = project_to_sphere(x)
        
        # Initialize inverse Hessian approximation
        B_inv = np.eye(n_coeffs)
        
        prev_grad = None
        prev_x = None
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Compute objective and gradient
            f = Y_matrix @ x
            f2 = f * f
            D_val = np.dot(W_eff, f2)
            N_val = np.dot(W_eff, f2 * f2)
            
            if D_val < 1e-12:
                break
                
            val = N_val / (D_val * D_val)
            
            f3 = f2 * f
            grad_N = 4.0 * (Y_matrix.T @ (W_eff * f3))
            grad_D = 2.0 * (Y_matrix.T @ (W_eff * f))
            grad = (grad_N * D_val - 2.0 * N_val * grad_D) / (D_val ** 3)
            
            # Project gradient onto tangent space
            grad_tangent = grad - np.dot(grad, x) * x
            
            # Check convergence
            if np.linalg.norm(grad_tangent) < tol:
                break
            
            # Newton step using BFGS approximation
            d = -B_inv @ grad_tangent
            
            # Line search with backtracking
            alpha = 1.0
            for ls_iter in range(30):
                x_new = project_to_sphere(x + alpha * d)
                f_new = Y_matrix @ x_new
                f2_new = f_new * f_new
                D_new = np.dot(W_eff, f2_new)
                N_new = np.dot(W_eff, f2_new * f2_new)
                
                if D_new < 1e-12:
                    alpha *= 0.5
                    continue
                    
                val_new = N_new / (D_new * D_new)
                
                # Armijo condition
                if val_new < val + 1e-4 * alpha * np.dot(grad_tangent, d):
                    break
                alpha *= 0.5
            
            # BFGS update
            if prev_grad is not None:
                s = x_new - x
                y = grad_tangent - prev_grad
                sy = np.dot(s, y)
                if abs(sy) > 1e-10:
                    B_inv = B_inv + ((1 + sy / np.dot(y, y)) * np.outer(s, s) 
                                    - np.outer(s, y) - np.outer(y, s)) / np.dot(y, y)
            
            prev_x = x
            prev_grad = grad_tangent
            x = x_new
        
        return x, val
    
    # --- 4. Optimization with multiple restarts ---
    best_score = np.inf
    best_coeffs = np.ones(n_coeffs) / np.sqrt(n_coeffs)
    
    inits = []
    
    # 1. Single modes
    for m in range(-k, k + 1):
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        inits.append(init)
        
    # 2. Symmetric pairs
    for m in range(1, k + 1):
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        init[-m + k] = 1.0
        inits.append(init)
        
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        init[-m + k] = -1.0
        inits.append(init)
        
    # 3. Random vectors
    for _ in range(150):
        init = np.random.randn(n_coeffs)
        init /= np.linalg.norm(init)
        inits.append(init)
        
    # 4. Sparse random
    for _ in range(100):
        init = np.random.randn(n_coeffs)
        mask = np.random.rand(n_coeffs) < 0.3
        init[~mask] = 0
        if np.linalg.norm(init) > 1e-6:
            init /= np.linalg.norm(init)
        inits.append(init)
        
    # 5. Anti-symmetric patterns
    for m in range(1, k + 1):
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        init[-m + k] = 0.0
        inits.append(init)
        
        init = np.zeros(n_coeffs)
        init[m + k] = 0.0
        init[-m + k] = 1.0
        inits.append(init)
        
    np.random.shuffle(inits)
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 60:
            break
            
        try:
            c_res, score = projected_bfgs(x0, max_iter=3000, tol=1e-12)
            
            # Normalize result to unit norm for consistency
            norm_c = np.linalg.norm(c_res)
            if norm_c > 1e-10:
                c_res = c_res / norm_c
            
            # Evaluate final score
            f_res = Y_matrix @ c_res
            l2_sq = np.dot(W_eff, f_res**2)
            l4 = np.dot(W_eff, f_res**4)
            score = l4 / (l2_sq * l2_sq)
            
            if score < best_score:
                best_score = score
                best_coeffs = c_res.copy()
                
        except Exception:
            continue

    # --- 5. Refine Best Solution ---
    if time.time() < deadline - 10:
        try:
            c_res, score = projected_bfgs(best_coeffs, max_iter=10000, tol=1e-14)
            
            norm_c = np.linalg.norm(c_res)
            if norm_c > 1e-10:
                c_res = c_res / norm_c
            
            f_res = Y_matrix @ c_res
            l2_sq = np.dot(W_eff, f_res**2)
            l4 = np.dot(W_eff, f_res**4)
            score = l4 / (l2_sq * l2_sq)
            
            if score < best_score:
                best_score = score
                best_coeffs = c_res.copy()
        except Exception:
            pass

    # --- 6. Validate and Return ---
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(n_coeffs)
        
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3:
        best_coeffs *= (1.0 / max_c)
    elif max_c > 1e5:
        best_coeffs /= max_c
        
    return best_coeffs
