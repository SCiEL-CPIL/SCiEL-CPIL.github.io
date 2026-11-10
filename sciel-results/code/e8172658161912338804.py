import numpy as np
from scipy.special import lpmv, factorial
import time
from scipy.optimize import line_search

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Strategy:
    1. Precompute basis matrix A and weighted basis B on the specified quadrature grid.
    2. Define objective R(u) = ||f||_4 / ||f||_2 on the unit sphere and its analytic gradient.
    3. Generate diverse initial candidates.
    4. Polish each candidate with sphere-constrained gradient descent (conjugate gradient style).
    5. Return the best valid coefficient vector found within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    k = 10
    n_m = 2 * k + 1  # 21 coefficients

    # 1. Setup Quadrature Grid
    mu, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    n_points = len(mu_flat)

    # 2. Construct Basis Functions
    A = np.zeros((n_points, n_m))
    
    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs, exact=False) / factorial(10 + m_abs, exact=False))
        return np.sqrt(val)
    
    # m=0
    idx = 10
    P = lpmv(0, 10, mu_flat)
    N = get_norm(0)
    A[:, idx] = N * P
    
    for m in range(1, k + 1):
        idx_pos = m + 10
        idx_neg = -m + 10
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, idx_pos] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, idx_neg] = sqrt2_N * P * np.sin(m * phi_flat)
    
    # Precompute weighted basis for fast gradient
    B = A * w_flat[:, None]

    # 3. Objective and Gradient on unit sphere (u is already unit norm)
    def objective_and_grad_sphere(u):
        """Compute R(u) and gradient for unit vector u."""
        f_vals = A @ u
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = d_phi * np.sum(w_flat * f2)
        norm4_4 = d_phi * np.sum(w_flat * f4)
        
        if norm2_sq < 1e-12:
            return 1e10, np.zeros(n_m)
            
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        R = N4 / N2
        
        # Gradient
        sum_wf_A = d_phi * (B.T @ f_vals)
        sum_wf3_A = d_phi * (B.T @ (f_vals ** 3))
        
        grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
        
        return R, grad_u

    # Sphere-constrained conjugate gradient polish
    def sphere_polish(c, max_iter=2000, lr_init=0.01, tol=1e-12):
        """Polish on the unit sphere using projected conjugate gradient with Wolfe line search."""
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return c
        u = c / norm_c
        R_curr, grad_curr = objective_and_grad_sphere(u)
        
        if not np.isfinite(R_curr):
            return c
        
        # Project gradient to tangent space
        u_dot_grad = np.dot(u, grad_curr)
        grad_tangent = grad_curr - u * u_dot_grad
        d = -grad_tangent
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-14:
            return u
        
        beta = 0.0
        
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            # Use scipy's line_search with Wolfe conditions adapted for sphere constraint
            def f_sphere(x):
                """Objective function for line search (x is step size along direction d)."""
                u_test = u + x * d
                u_test_norm = np.linalg.norm(u_test)
                if u_test_norm < 1e-12:
                    return np.inf, None
                u_test = u_test / u_test_norm
                val, _ = objective_and_grad_sphere(u_test)
                return val, None
            
            # Simple backtracking with Armijo condition
            step = lr_init
            alpha = 0.5
            c1 = 1e-4  # Armijo parameter
            success = False
            for _ in range(30):
                u_new = u + step * d
                u_new_norm = np.linalg.norm(u_new)
                if u_new_norm < 1e-12:
                    step *= 0.5
                    continue
                u_new = u_new / u_new_norm
                
                R_new, grad_new = objective_and_grad_sphere(u_new)
                
                # Armijo condition
                if R_new < R_curr + c1 * step * np.dot(grad_tangent, d):
                    u = u_new
                    R_curr = R_new
                    success = True
                    break
                else:
                    step *= alpha
            
            if not success:
                # Try a smaller step
                for trial_step in [lr_init * 0.1, lr_init * 0.01, lr_init * 0.001]:
                    u_new = u + trial_step * d
                    u_new_norm = np.linalg.norm(u_new)
                    if u_new_norm < 1e-12:
                        continue
                    u_new = u_new / u_new_norm
                    R_new, grad_new = objective_and_grad_sphere(u_new)
                    if R_new < R_curr:
                        u = u_new
                        R_curr = R_new
                        success = True
                        break
                if not success:
                    break
            
            # Update gradient
            u_dot_grad_new = np.dot(u, grad_new)
            grad_tangent_new = grad_new - u * u_dot_grad_new
            
            grad_norm = np.linalg.norm(grad_tangent_new)
            if grad_norm < tol:
                break
            
            # Compute beta (Polak-Ribiere)
            d_old = d
            d_new = -grad_tangent_new
            
            if it > 0:
                num = np.dot(grad_tangent_new, grad_tangent_new - grad_tangent)
                den = np.dot(grad_tangent, grad_tangent)
                if den > 1e-14:
                    beta = max(0, num / den)
                else:
                    beta = 0
            
            d = d_new + beta * d_old if it > 0 else d_new
            d = d - u * np.dot(u, d)  # Re-project to tangent space
            
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-14:
                break
            
            # Adaptive step size
            if it % 30 == 0:
                # Try to increase step size if we've been making good progress
                if R_curr < 0.95 * (R_curr + 0.01):
                    lr_init = min(lr_init * 1.5, 0.1)
                else:
                    lr_init = max(lr_init * 0.7, 0.001)
        
        return u

    # 4. Generate Initial Candidates
    candidates = []
    rng = np.random.default_rng(42)
    
    # Pure modes
    for m in range(0, 11):
        c = np.zeros(n_m)
        c[10 + m] = 1.0
        candidates.append(c)
    
    # Also try negative m (sin modes) as pure
    for m in range(1, 11):
        c = np.zeros(n_m)
        c[10 - m] = 1.0
        candidates.append(c)
    
    # Pairs of positive m modes
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            c = np.zeros(n_m)
            c[10 + m1] = 1.0
            c[10 + m2] = 1.0
            c /= np.linalg.norm(c)
            candidates.append(c)
    
    # Pairs mixing cos and sin for same |m|
    for m in range(1, 11):
        for angle in np.linspace(0, np.pi/2, 5):
            c = np.zeros(n_m)
            c[10 + m] = np.cos(angle)
            c[10 - m] = np.sin(angle)
            c /= np.linalg.norm(c)
            candidates.append(c)
    
    # Triplets
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            for m3 in range(m2 + 1, 11):
                if time.time() > deadline - 300:
                    break
                c = np.zeros(n_m)
                c[10 + m1] = 1.0
                c[10 + m2] = 1.0
                c[10 + m3] = 1.0
                c /= np.linalg.norm(c)
                candidates.append(c)
    
    # Random normalized vectors
    for _ in range(150):
        if time.time() > deadline - 300:
            break
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)
    
    # Random with emphasis on low-m modes
    for _ in range(80):
        if time.time() > deadline - 300:
            break
        c_rand = np.zeros(n_m)
        for m in range(0, 11):
            if rng.random() < 0.7:
                c_rand[10 + m] = rng.standard_normal()
        if np.linalg.norm(c_rand) > 1e-12:
            c_rand /= np.linalg.norm(c_rand)
            candidates.append(c_rand)
    
    # 5. Evaluate and sort candidates
    candidate_vals = []
    for c_init in candidates:
        if time.time() > deadline - 180:
            break
        val, _ = objective_and_grad_sphere(c_init)
        candidate_vals.append((val, c_init))
    
    candidate_vals.sort(key=lambda x: x[0])
    
    # 6. Polish top candidates with sphere-constrained CG using multiple learning rates
    best_c = None
    best_val = np.inf
    n_to_polish = min(40, len(candidate_vals))
    
    for i in range(n_to_polish):
        if time.time() > deadline - 30:
            break
        
        val_init, c_init = candidate_vals[i]
        
        # Try multiple learning rates to escape local minima
        for lr in [0.01, 0.005, 0.02]:
            if time.time() > deadline - 10:
                break
            polished = sphere_polish(c_init, max_iter=1000, lr_init=lr, tol=1e-10)
            val_polished, _ = objective_and_grad_sphere(polished)
            
            if val_polished < best_val and np.all(np.isfinite(polished)):
                best_val = val_polished
                best_c = polished.copy()
    
    # 7. Final high-precision polish on the best with tighter tolerance
    if best_c is not None and time.time() < deadline - 10:
        # First pass with moderate tolerance
        final1 = sphere_polish(best_c, max_iter=2000, lr_init=0.005, tol=1e-12)
        val_final1, _ = objective_and_grad_sphere(final1)
        if val_final1 < best_val:
            best_val = val_final1
            best_c = final1
        
        # Second pass with very tight tolerance for final precision
        if time.time() < deadline - 5:
            final2 = sphere_polish(best_c, max_iter=3000, lr_init=0.001, tol=1e-14)
            val_final2, _ = objective_and_grad_sphere(final2)
            if val_final2 < best_val:
                best_val = val_final2
                best_c = final2

    # 8. Final Validity Check and Scaling
    if best_c is None or not np.all(np.isfinite(best_c)):
        best_c = np.zeros(n_m)
        best_c[10] = 1.0
        
    # Ensure max|c| is in valid range
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1e-2 / max_abs)
    elif max_abs > 1e5:
        best_c = best_c * (1e4 / max_abs)
        
    return best_c
