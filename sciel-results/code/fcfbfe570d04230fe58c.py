import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.special import lpmv, factorial

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c for degree-10 spherical harmonics minimizing ||f||_4 / ||f||_2.
    
    Strategy:
    1. Precompute Spherical Harmonic matrix Y at quadrature points.
    2. The objective R(c) = ||f||_4 / ||f||_2 is scale invariant. 
       We define the objective as J(c) = (∫ f^4 dΩ) / (∫ f^2 dΩ)^2.
       Note: Minimizing J(c) is equivalent to minimizing R(c).
       J(c) is homogeneous of degree 0.
    3. Instead of local gradient descent (L-BFGS) which gets stuck in local minima,
       we use Differential Evolution (DE), a global stochastic optimization algorithm.
       DE does not require gradients and explores the search space broadly,
       making it much more robust for non-convex quartic objectives.
    4. We optimize over the 21 coefficients directly, with bounds.
       Since the objective is scale-invariant, we can fix a scale or let DE find it.
       To avoid numerical issues with very large/small scales, we bound the coefficients
       to [-1, 1] and rely on the scale invariance. Actually, it's safer to normalize
       inside the objective or fix the norm. 
       However, DE works best with bounded variables. 
       Let's parameterize c via its direction on the sphere using 20 free variables 
       (stereographic projection) OR just use 21 variables and normalize inside the 
       objective function. Normalizing inside the objective makes the landscape 
       "flat" along the radial direction, which DE handles poorly.
       
       Better: Use the 20-dimensional stereographic representation as before, 
       but apply Differential Evolution to these 20 variables.
       The domain of the stereographic projection is R^20, which is unbounded.
       We can bound it to [-10, 10] which covers most of the sphere.
    """
    k = 10
    n_coeffs = 2 * k + 1  # 21
    
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
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
            
    # --- 2. Optimization via Differential Evolution ---
    
    def z_to_c(z):
        """Map z in R^{n-1} to c in S^{n-1} via stereographic projection."""
        z_norm_sq = np.dot(z, z)
        denom = 1.0 + z_norm_sq
        c1 = (1.0 - z_norm_sq) / denom
        c_rest = 2.0 * z / denom
        c = np.zeros(n_coeffs)
        c[0] = c1
        c[1:] = c_rest
        return c

    def objective_de(z):
        """
        Objective for Differential Evolution.
        Returns J(c) = (∫ f^4) / (∫ f^2)^2.
        Since c is on the unit sphere, ∫ f^2 = 1. So J(c) = ∫ f^4.
        We minimize this.
        """
        c = z_to_c(z)
        f = Y_matrix @ c
        # Calculate norms
        int_f2 = np.dot(W_eff, f**2)
        int_f4 = np.dot(W_eff, f**4)
        
        # Avoid division by zero, though int_f2 should be close to 1
        if int_f2 < 1e-12:
            return 1e10
            
        # The ratio R = (int_f4)^(1/4) / (int_f2)^(1/2)
        # Minimizing R is equivalent to minimizing R^4 = int_f4 / (int_f2)^2
        return int_f4 / (int_f2**2)

    # Bounds for z variables. 
    # Stereographic projection maps the sphere (minus pole) to R^20.
    # Large z values correspond to points near the south pole (c0 ~ -1).
    # We bound z to [-10, 10] to cover the sphere sufficiently.
    bounds = [(-10.0, 10.0)] * (n_coeffs - 1)
    
    # DE Parameters
    # popsize: 15 * 20 = 300 individuals. 
    # maxiter: 200.
    # This might be slow. Let's estimate.
    # Each evaluation: Y_matrix @ c (9100x21 dot 21) -> 9100 ops. 
    # f**4, dot W_eff -> 9100 ops.
    # Total ~ 20k ops per eval.
    # 300 pop * 200 iter * 20k = 1.2e9 ops. 
    # NumPy is fast, this should take < 10 seconds.
    
    # We use a time-aware wrapper to stop DE if it takes too long.
    # scipy's differential_evolution doesn't have a direct time limit, 
    # but we can check in the objective function and raise an exception 
    # or return a large value to signal termination? 
    # Better: Run DE with a modest maxiter and popsize, then refine with L-BFGS if time permits.
    
    best_z = None
    best_val = np.inf
    
    try:
        # Use DE to find a good global candidate
        res_de = differential_evolution(
            objective_de, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15, 
            tol=1e-6,
            polish=False  # We will polish manually
        )
        
        if res_de.success or res_de.fun < best_val:
            best_z = res_de.x
            best_val = res_de.fun
            
    except Exception as e:
        pass
    
    # --- 3. Polish with L-BFGS-B ---
    # If we found a good starting point from DE, refine it with local optimization.
    
    def z_to_c_polish(z):
        return z_to_c(z)

    def objective_z(z):
        c = z_to_c(z)
        f = Y_matrix @ c
        int_f2 = np.dot(W_eff, f**2)
        int_f4 = np.dot(W_eff, f**4)
        return int_f4 / (int_f2**2)

    def gradient_z(z):
        c = z_to_c(z)
        f = Y_matrix @ c
        int_f2 = np.dot(W_eff, f**2)
        
        # Gradient of J = I4 / I2^2
        # dJ/dc = (1/I2^2) dI4/dc - (2 I4 / I2^3) dI2/dc
        # dI4/dc = 4 Y^T (W f^3)
        # dI2/dc = 2 Y^T (W f)
        
        dI4_dc = 4.0 * (Y_matrix.T @ (W_eff * f**3))
        dI2_dc = 2.0 * (Y_matrix.T @ (W_eff * f))
        
        grad_J_c = dI4_dc / (int_f2**2) - (2.0 * int_f4 / (int_f2**3)) * dI2_dc
        
        # Chain rule for z -> c
        z_norm_sq = np.dot(z, z)
        denom_sq = (1.0 + z_norm_sq)**2
        
        term_c1 = grad_J_c[0] * (-4.0 * z / denom_sq)
        
        grad_c_rest = grad_J_c[1:]
        z_dot_grad_rest = np.dot(z, grad_c_rest)
        
        term_rest = 2.0 / denom_sq * ((1.0 + z_norm_sq) * grad_c_rest - 2.0 * z * z_dot_grad_rest)
        
        return term_c1 + term_rest

    if best_z is not None:
        try:
            from scipy.optimize import minimize
            res_lbfgs = minimize(objective_z, best_z, jac=gradient_z, method='L-BFGS-B', 
                                 options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res_lbfgs.fun < best_val:
                best_z = res_lbfgs.x
                best_val = res_lbfgs.fun
        except Exception:
            pass

    # If DE failed or didn't run, fall back to random starts + L-BFGS
    if best_z is None or time.time() > deadline - 10:
        # Fallback: Just use a random start with L-BFGS
        np.random.seed(42)
        z0 = np.random.randn(n_coeffs - 1) * 0.5
        try:
            from scipy.optimize import minimize
            res_lbfgs = minimize(objective_z, z0, jac=gradient_z, method='L-BFGS-B', 
                                 options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res_lbfgs.fun < best_val:
                best_z = res_lbfgs.x
                best_val = res_lbfgs.fun
        except Exception:
            best_z = np.zeros(n_coeffs - 1) # Default to c0=1

    # --- 4. Final Validation and Scaling ---
    
    c_final = z_to_c(best_z)
    
    if np.any(np.isnan(c_final)) or np.any(np.isinf(c_final)):
        c_final = np.ones(n_coeffs)
        
    max_c = np.max(np.abs(c_final))
    if max_c < 1e-3:
        c_final *= (1.0 / max_c)
    elif max_c > 1e5:
        c_final /= max_c
        
    return c_final
