import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 595

    k = 10
    n_m = 2 * k + 1

    # Quadrature grid
    mu, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    n_points = len(mu_flat)

    # Basis functions
    A = np.zeros((n_points, n_m))

    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs, exact=False) / factorial(10 + m_abs, exact=False))
        return np.sqrt(val)

    A[:, 10] = get_norm(0) * lpmv(0, 10, mu_flat)

    for m in range(1, k + 1):
        idx_pos = m + 10
        idx_neg = -m + 10
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, idx_pos] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, idx_neg] = sqrt2_N * P * np.sin(m * phi_flat)

    B = A * w_flat[:, None]

    # Precompute A^T A and A^T (A^3) for the objective
    # The objective is R = ||f||_4 / ||f||_2
    # Let f = A u, where u is a unit vector.
    # ||f||_2^2 = u^T (A^T A) u = u^T G u
    # ||f||_4^4 = u^T H u, where H_{ij} = sum w (A_i A_j)^3 ? No.
    # ||f||_4^4 = sum w f^4 = sum w (A u)^4
    # This is a 4th order tensor. It's not a quadratic form in u.
    # So we can't just precompute a matrix H.
    # We must compute f = A u, then sum w f^4.

    # However, the gradient can be computed efficiently.
    # d/du (||f||_4^4) = 4 sum w f^3 (A u)^3 ... no, chain rule.
    # Let S = sum w f^4. dS/du = 4 sum w f^3 (df/du) = 4 A^T (w * f^3).
    # Let Q = sum w f^2 = u^T G u. dQ/du = 2 G u.
    # R = S^{1/4} Q^{-1/2}.
    # dR/du = (1/4) S^{-3/4} S' Q^{-1/2} - (1/2) S^{1/4} Q^{-3/2} Q'
    #       = (1/4) S^{-3/4} Q^{-1/2} * 4 A^T(w f^3) - (1/2) S^{1/4} Q^{-3/2} * 2 G u
    #       = S^{-3/4} Q^{-1/2} A^T(w f^3) - S^{1/4} Q^{-3/2} G u
    #       = (S^{1/4} Q^{1/2})^{-3} [ (S^{1/4})^3 / (Q^{1/2})^3 * ... ] -> let's just use the formula.
    #       = (1/R^3) * (N2/N4) * A^T(w f^3) - (1/R) * (1/N2^3) * G u  -- wait, let's stick to the previous derivation.
    
    # Previous derivation:
    # N2 = sqrt(Q), N4 = S^{1/4}. R = N4/N2.
    # grad_u = (N4^3) * (1/4 * 4 A^T(w f^3)) * (1/N4) ... no.
    # dR = d(N4/N2) = (N2 dN4 - N4 dN2) / N2^2.
    # dN4 = (1/4) S^{-3/4} dS = (1/4) N4^{-3} * 4 A^T(w f^3) = N4^{-3} A^T(w f^3).
    # dN2 = (1/2) Q^{-1/2} dQ = (1/2) N2^{-1} * 2 G u = N2^{-1} G u.
    # dR = N2^{-2} (N2 N4^{-3} A^T(w f^3) - N4 N2^{-1} G u)
    #    = N2^{-1} N4^{-3} A^T(w f^3) - N4 N2^{-3} G u.
    # This matches: grad_u = (N4**3) * sum_wf3_A * (1/N4**4 ... no)
    # My previous code: grad_u = (N4**3) * sum_wf_A ... wait.
    # sum_wf_A was d_phi * (B.T @ f_vals) = d_phi * A^T (w f). This is dQ/du / 2?
    # B = A * w. B.T @ f = A^T (w f).
    # Q = sum w f^2. dQ/du = 2 A^T (w f).
    # sum_wf_A = A^T (w f). So dQ/du = 2 sum_wf_A.
    # My previous code: grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A.
    # Let's check dimensions.
    # dN4 = N4^{-3} A^T (w f^3). Let T = A^T (w f^3).
    # dN2 = N2^{-1} G u. Note G u = A^T A u = A^T f. And A^T (w f) is NOT A^T f unless w=1.
    # Wait, the measure is w dmu dphi.
    # Q = sum w f^2. dQ/du = 2 sum w f (A) = 2 A^T (w f).
    # So sum_wf_A = A^T (w f) = dQ/du / 2.
    # My formula: grad_u = N2^{-1} N4^{-3} T - N4 N2^{-3} (dQ/du / 2 * 2) ...
    # dR = N2^{-1} N4^{-3} T - N4 N2^{-3} (N2 * dN2) ... no.
    # dR = N2^{-1} N4^{-3} T - N4 N2^{-3} G u.
    # G u = A^T f. This is NOT the same as A^T (w f).
    # So the previous gradient was WRONG because it used A^T (w f) instead of A^T f for the G u term.
    # Let's re-derive carefully.
    
    # R = S^{1/4} Q^{-1/2}
    # dR/du = (1/4) S^{-3/4} S' Q^{-1/2} - (1/2) S^{1/4} Q^{-3/2} Q'
    # S' = 4 A^T (w f^3). Let V3 = A^T (w f^3).
    # Q' = 2 A^T (w f). Let V1 = A^T (w f).
    # dR/du = S^{-3/4} Q^{-1/2} V3 - S^{1/4} Q^{-3/2} V1
    #       = (S^{1/4} Q^{1/2})^{-3} * S^{1/4} Q^{1/2} * ... 
    #       = R^{-3} * (N4 N2) * (N2^{-1} N4^{-3} ... no)
    #       = R^{-3} * (V3 / N2^3 ... no)
    # Let's just use: dR/du = (1/(N4^3 N2)) V3 - (N4/(N2^3)) V1.
    
    # Let's check the previous code:
    # grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
    # sum_wf3_A = d_phi * (B.T @ f**3) = d_phi * A^T (w f**3).
    # sum_wf_A = d_phi * (B.T @ f) = d_phi * A^T (w f).
    # So previous code: N4^3 * V3 - N4/N2^3 * V1.
    # Correct formula: (1/(N4^3 N2)) V3 - (N4/N2^3) V1.
    # The first term was wrong by a factor of N4^6 N2.
    # This explains why the optimization might have been unstable or slow.
    
    # Corrected objective and gradient
    def objective_and_grad(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return 1e10, np.zeros(n_m)
        u = c / norm_c
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
        
        V3 = d_phi * (B.T @ (f_vals ** 3))
        V1 = d_phi * (B.T @ f_vals)
        
        grad_u = (1.0 / (N4**3 * N2)) * V3 - (N4 / N2**3) * V1
        
        u_dot_grad = np.dot(u, grad_u)
        grad_c = (grad_u - u * u_dot_grad) / norm_c
        
        return R, grad_c

    # Sphere polish with correct gradient
    def sphere_polish(c, max_iter=500, lr=0.05):
        c = c.copy()
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return c
        u = c / norm_c
        
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
            f_vals = A @ u
            f2 = f_vals ** 2
            f4 = f2 ** 2
            norm2_sq = d_phi * np.sum(w_flat * f2)
            norm4_4 = d_phi * np.sum(w_flat * f4)
            if norm2_sq < 1e-12:
                break
            N2 = np.sqrt(norm2_sq)
            N4 = norm4_4 ** 0.25
            
            V3 = d_phi * (B.T @ (f_vals ** 3))
            V1 = d_phi * (B.T @ f_vals)
            
            grad_u = (1.0 / (N4**3 * N2)) * V3 - (N4 / N2**3) * V1
            
            u_dot_grad = np.dot(u, grad_u)
            grad_tangent = grad_u - u * u_dot_grad
            
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-14:
                break
            
            step = lr
            R_curr = N4 / N2
            improved = False
            for _ in range(20):
                u_new = u - step * grad_tangent
                u_new_norm = np.linalg.norm(u_new)
                if u_new_norm < 1e-12:
                    step *= 0.5
                    continue
                u_new = u_new / u_new_norm
                
                f_new = A @ u_new
                f2_new = f_new ** 2
                f4_new = f2_new ** 2
                norm2_sq_new = d_phi * np.sum(w_flat * f2_new)
                norm4_4_new = d_phi * np.sum(w_flat * f4_new)
                if norm2_sq_new < 1e-12:
                    step *= 0.5
                    continue
                R_new = (norm4_4_new ** 0.25) / np.sqrt(norm2_sq_new)
                
                if R_new < R_curr:
                    u = u_new
                    R_curr = R_new
                    improved = True
                    break
                else:
                    step *= 0.5
            
            if not improved and step < 1e-12:
                break
        
        return u

    # Generate Initial Candidates
    candidates = []
    rng = np.random.default_rng(1234)

    # Pure modes
    for m in range(11):
        c = np.zeros(n_m)
        c[10 + m] = 1.0
        candidates.append(c)

    # Pairs
    for m1 in range(11):
        for m2 in range(m1 + 1, 11):
            c = np.zeros(n_m)
            c[10 + m1] = 1.0
            c[10 + m2] = 1.0
            c /= np.linalg.norm(c)
            candidates.append(c)

    # Random
    for _ in range(150):
        if time.time() > deadline - 100:
            break
        c = rng.standard_normal(n_m)
        c /= np.linalg.norm(c)
        candidates.append(c)

    # Evaluate and sort
    candidate_vals = []
    for c_init in candidates:
        if time.time() > deadline - 80:
            break
        val, _ = objective_and_grad(c_init)
        candidate_vals.append((val, c_init))
    
    candidate_vals.sort(key=lambda x: x[0])

    # Optimize
    best_c = None
    best_val = np.inf

    for val_init, c_init in candidate_vals:
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            val = res.fun
            if val < best_val and np.all(np.isfinite(res.x)):
                best_val = val
                best_c = res.x.copy()
        except Exception:
            if val_init < best_val:
                best_val = val_init
                best_c = c_init.copy()
        
        if best_c is not None and time.time() < deadline - 10:
            polished = sphere_polish(best_c, max_iter=400, lr=0.05)
            val_p, _ = objective_and_grad(polished)
            if val_p < best_val:
                best_val = val_p
                best_c = polished.copy()

    # Final polish
    if best_c is not None and time.time() < deadline - 5:
        polished = sphere_polish(best_c, max_iter=1000, lr=0.01)
        val_p, _ = objective_and_grad(polished)
        if val_p < best_val:
            best_val = val_p
            best_c = polished.copy()

    if best_c is None or not np.all(np.isfinite(best_c)):
        best_c = np.zeros(n_m)
        best_c[10] = 1.0

    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)

    return best_c
