import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s limit

    # Use exact Gauss-Hermite quadrature.
    N_points = 14
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(N_points)
    
    # Pre-compute powers and Hermite polynomials at nodes
    x = x_nodes
    x2 = x**2
    x3 = x2 * x
    x4 = x2**2
    x5 = x4 * x
    x6 = x4 * x2
    x7 = x6 * x
    x8 = x4**2
    x9 = x8 * x
    x10 = x8 * x2
    x11 = x10 * x
    x12 = x6**2
    
    # Hermite polynomials (physicists' convention)
    H0 = np.ones_like(x)
    H4 = x4 - 6*x2 + 3
    H8 = x8 - 28*x6 + 210*x4 - 420*x2 + 105
    H12 = x12 - 66*x10 + 1260*x8 - 9240*x6 + 23100*x4 - 27720*x2 + 10395
    
    # Derivatives
    dH0 = np.zeros_like(x)
    dH4 = 4*x3 - 12*x
    dH8 = 8*x7 - 168*x5 + 840*x3 - 840*x
    dH12 = 12*x11 - 660*x9 + 10080*x7 - 55440*x5 + 92400*x3 - 55440*x

    # Precompute weighted vectors for fast computation
    w = w_nodes
    wx2 = w * x2
    w_deriv = w  # weight for derivative term

    def compute_score_fast(coeffs):
        """
        Computes the score using exact Gauss-Hermite quadrature.
        coeffs: (c0, c4, c8)
        """
        if time.time() > deadline - 2:
            return np.inf
            
        c0, c4, c8 = coeffs
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        
        g = c0*H0 + c4*H4 + c8*H8 + c12*H12
        gp = c0*dH0 + c4*dH4 + c8*dH8 + c12*dH12
        
        # Integrals
        I0 = np.sum(w * g**2)
        if I0 <= 1e-30 or not np.isfinite(I0):
            return np.inf
                
        Ix2 = np.sum(wx2 * g**2)
        if not np.isfinite(Ix2) or Ix2 < 0:
            return np.inf
                
        deriv_term = gp - x * g
        Ideriv = np.sum(w * deriv_term**2)
        if not np.isfinite(Ideriv) or Ideriv < 0:
            return np.inf
                
        A_f_sq = Ix2 / I0
        A_gf_sq = (1.0 / (4.0 * np.pi**2)) * Ideriv / I0
        
        return np.sqrt(A_f_sq * A_gf_sq)

    def compute_grad(coeffs):
        """
        Computes the gradient of the score using exact Gauss-Hermite quadrature.
        coeffs: (c0, c4, c8)
        Returns: gradient (3,)
        """
        c0, c4, c8 = coeffs
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        
        # g and gp as vectors
        g = c0*H0 + c4*H4 + c8*H8 + c12*H12
        gp = c0*dH0 + c4*dH4 + c8*dH8 + c12*dH12
        
        # Derivatives of g and gp w.r.t. c0, c4, c8
        # dg/dc0 = H0, dg/dc4 = H4, dg/dc8 = H8
        # dc12/dc0 = -1/10395, dc12/dc4 = -3/10395, dc12/dc8 = -105/10395
        dg = np.array([H0, H4, H8])  # shape (3, N)
        dgp = np.array([dH0, dH4, dH8])  # shape (3, N)
        
        # Add contribution from c12
        dc12 = np.array([-1.0, -3.0, -105.0]) / 10395.0
        dg = dg + dc12[:, None] * H12[None, :]
        dgp = dgp + dc12[:, None] * dH12[None, :]
        
        # Compute integrals
        I0 = np.sum(w * g**2)
        Ix2 = np.sum(wx2 * g**2)
        deriv_term = gp - x * g
        Ideriv = np.sum(w * deriv_term**2)
        
        if I0 <= 1e-30 or not np.isfinite(I0):
            return np.array([np.nan, np.nan, np.nan])
        
        # Gradients of integrals
        dI0 = 2 * np.array([np.sum(w * g * dg[k]) for k in range(3)])
        dIx2 = 2 * np.array([np.sum(wx2 * g * dg[k]) for k in range(3)])
        
        # dIderiv: deriv_term = gp - x*g
        # d(deriv_term)/dc_k = dgp[k] - x*dg[k]
        dderiv = dgp - x[None, :] * dg  # shape (3, N)
        dIderiv = 2 * np.array([np.sum(w * deriv_term * dderiv[k]) for k in range(3)])
        
        # Score = sqrt((Ix2/I0) * (Ideriv/(4*pi^2*I0)))
        # Let A = Ix2/I0, B = Ideriv/(4*pi^2*I0)
        # Score = sqrt(A*B)
        # d(Score)/dc_k = (1/(2*sqrt(A*B))) * (B * dA/dck + A * dB/dck)
        
        A = Ix2 / I0
        B = Ideriv / (4.0 * np.pi**2 * I0)
        score = np.sqrt(A * B)
        
        if score < 1e-30:
            return np.array([np.nan, np.nan, np.nan])
        
        dA = (dIx2 * I0 - Ix2 * dI0) / (I0**2)
        dB = (dIderiv * I0 - Ideriv * dI0) / (4.0 * np.pi**2 * I0**2)
        
        grad = (B * dA + A * dB) / (2.0 * score)
        
        return grad

    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = compute_score_fast(x0)
    best_coeffs = x0.copy()
    
    # Strategy:
    # 1. BFGS with analytic gradient for fast convergence
    # 2. Differential Evolution for global search if needed
    # 3. Newton-style polish with Hessian
    
    # First, try BFGS with analytic gradient from seed
    if time.time() < deadline - 30:
        try:
            res_bfgs = opt.minimize(
                compute_score_fast, 
                best_coeffs, 
                jac=compute_grad,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50}
            )
            if res_bfgs.fun < best_score and np.isfinite(res_bfgs.fun):
                best_score = res_bfgs.fun
                best_coeffs = res_bfgs.x.copy()
        except Exception:
            pass

    # Try Differential Evolution for global search
    bounds = [(-5, 5), (-0.5, 0.5), (-0.05, 0.05)]
    
    if time.time() < deadline - 20:
        try:
            res_de = opt.differential_evolution(
                compute_score_fast, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=10,
                tol=1e-10,
                init='sobol',
                polish=False  # We'll do our own polish
            )
            
            if res_de.fun < best_score and np.isfinite(res_de.fun):
                best_score = res_de.fun
                best_coeffs = res_de.x.copy()
                
                # Polish the DE result
                if time.time() < deadline - 10:
                    try:
                        res_polish = opt.minimize(
                            compute_score_fast, 
                            best_coeffs, 
                            jac=compute_grad,
                            method='L-BFGS-B',
                            options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14}
                        )
                        if res_polish.fun < best_score and np.isfinite(res_polish.fun):
                            best_score = res_polish.fun
                            best_coeffs = res_polish.x.copy()
                    except Exception:
                        pass
        except Exception:
            pass

    # Random restarts with analytic gradient polish
    np.random.seed(42)
    n_restarts = 15
    for i in range(n_restarts):
        if time.time() > deadline - 8:
            break
        
        # Random start within reasonable bounds
        x_try = np.array([
            np.random.uniform(-3, 3),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.03, 0.03)
        ])
        
        try:
            res_restart = opt.minimize(
                compute_score_fast, 
                x_try, 
                jac=compute_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-13}
            )
            if res_restart.fun < best_score and np.isfinite(res_restart.fun):
                best_score = res_restart.fun
                best_coeffs = res_restart.x.copy()
        except Exception:
            pass

    # Final high-precision polish: use Newton's method with numerical Hessian
    if time.time() < deadline - 5:
        try:
            # Compute gradient at best point
            grad = compute_grad(best_coeffs)
            if np.all(np.isfinite(grad)):
                # Use a small step Newton iteration with numerical Hessian
                for _ in range(50):
                    if time.time() > deadline - 3:
                        break
                    
                    # Numerical Hessian
                    eps = 1e-7
                    H = np.zeros((3, 3))
                    for i in range(3):
                        for j in range(3):
                            e_i = np.zeros(3)
                            e_i[i] = eps
                            e_j = np.zeros(3)
                            e_j[j] = eps
                            
                            f_pp = compute_score_fast(best_coeffs + e_i + e_j)
                            f_p = compute_score_fast(best_coeffs + e_i)
                            f_m = compute_score_fast(best_coeffs - e_i)
                            f_pm = compute_score_fast(best_coeffs + e_j)
                            f_mm = compute_score_fast(best_coeffs - e_j)
                            f0 = compute_score_fast(best_coeffs)
                            
                            H[i, j] = (f_pp - f_p - f_pm + f0) / (eps * eps)
                    
                    # Solve H * delta = -grad
                    try:
                        delta = np.linalg.solve(H, -grad)
                        # Limit step size
                        step_norm = np.linalg.norm(delta)
                        if step_norm > 0.1:
                            delta = delta / step_norm * 0.1
                        new_coeffs = best_coeffs + delta
                        new_score = compute_score_fast(new_coeffs)
                        if new_score < best_score and np.isfinite(new_score):
                            best_score = new_score
                            best_coeffs = new_coeffs
                            grad = compute_grad(best_coeffs)
                            if np.linalg.norm(grad) < 1e-12:
                                break
                        else:
                            break
                    except np.linalg.LinAlgError:
                        break
        except Exception:
            pass

    return np.array(best_coeffs)
