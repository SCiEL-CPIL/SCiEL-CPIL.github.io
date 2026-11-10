import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Uses spectral-domain formulation with precomputed integration tensors for
    fast objective evaluation, combined with multi-restart optimization.
    """
    start_time = time.time()
    deadline = start_time + 595  # Safety margin before 630s kill

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
    
    # m = 0
    P = lpmv(0, 10, mu_flat)
    N = get_norm(0)
    A[:, 10] = N * P
    
    # m > 0 and m < 0
    for m in range(1, k + 1):
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, m + 10] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, -m + 10] = sqrt2_N * P * np.sin(m * phi_flat)
    
    # Precompute weighted basis for fast gradient
    B = A * w_flat[:, None]
    
    # 3. Precompute spectral integration tensors for L2 and L4
    # L2: ||f||_2^2 = sum_{i,j} c_i c_j * G2[i,j]
    # where G2[i,j] = integral Y_i Y_j dOmega
    # L4: ||f||_4^4 = sum_{i,j,k,l} c_i c_j c_k c_l * G4[i,j,k,l]
    # where G4[i,j,k,l] = integral Y_i Y_j Y_k Y_l dOmega
    
    # G2 is just the Gram matrix (should be identity for orthonormal basis)
    G2 = d_phi * (A.T @ (w_flat[:, None] * A))
    
    # For L4, we need the 4th moment tensor. This is 21^4 = 194481 elements.
    # Compute efficiently: G4[i,j,k,l] = integral Y_i Y_j Y_k Y_l
    # We can compute this as: for each point, outer product of 4 basis values
    # But that's expensive. Instead, use the fact that:
    # ||f||_4^4 = integral (sum_i c_i Y_i)^4
    #           = sum_{i,j,k,l} c_i c_j c_k c_l * integral Y_i Y_j Y_k Y_l
    
    # Precompute the 4th-order tensor using quadrature
    # T[i,j,k,l] = sum_p w_p * A[p,i] * A[p,j] * A[p,k] * A[p,l]
    # This is O(n_points * n_m^4) which is 9100 * 194481 ≈ 1.77e9 - too slow!
    
    # Alternative: compute on-the-fly but cache intermediate results
    # ||f||_4^4 = sum_p w_p * (sum_i c_i A[p,i])^4
    # We can expand: (sum_i c_i A_i)^4 = sum_{i,j,k,l} c_i c_j c_k c_l A_i A_j A_k A_l
    # So G4[i,j,k,l] = sum_p w_p A[p,i] A[p,j] A[p,k] A[p,l]
    
    # Instead of storing full G4, compute L4 norm directly from f_vals
    # But precompute G2 for L2 (which is just identity for orthonormal basis)
    
    # Verify orthonormality
    # G2 should be close to identity
    
    # 4. Fast objective using precomputed A and B
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
        
        sum_wf_A = d_phi * (B.T @ f_vals)
        sum_wf3_A = d_phi * (B.T @ (f_vals ** 3))
        
        grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
        
        u_dot_grad = np.dot(u, grad_u)
        grad_c = (grad_u - u * u_dot_grad) / norm_c
        
        return R, grad_c

    # 5. Sphere-constrained gradient descent polish with adaptive step
    def sphere_polish(c, max_iter=800, lr_init=0.05):
        c = c.copy()
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return c
        u = c / norm_c
        lr = lr_init
        
        for it in range(max_iter):
            if time.time() > deadline - 3:
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
            R_curr = N4 / N2
            
            sum_wf_A = d_phi * (B.T @ f_vals)
            sum_wf3_A = d_phi * (B.T @ (f_vals ** 3))
            
            grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
            
            u_dot_grad = np.dot(u, grad_u)
            grad_tangent = grad_u - u * u_dot_grad
            
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-13:
                break
            
            # Adaptive step size with backtracking
            step = lr
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
                    # Increase step slightly if we made good progress
                    if R_curr < R_curr - 0.01 * step * grad_norm:
                        lr = min(lr * 1.1, 0.1)
                    break
                else:
                    step *= 0.5
            
            if not improved or step < 1e-12:
                break
            # Decrease learning rate slowly
            lr = max(lr * 0.999, 0.001)
        
        return u

    # 6. Generate diverse initial candidates (reduced set, prioritized)
    candidates = []
    
    # Pure m=0
    c0 = np.zeros(n_m)
    c0[10] = 1.0
    candidates.append(c0)
    
    # Pure m>0 cos modes
    for m in range(1, 11):
        c_cos = np.zeros(n_m)
        c_cos[m + 10] = 1.0
        candidates.append(c_cos)
    
    # Pairs of m modes (all pairs)
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            c_pair = np.zeros(n_m)
            c_pair[10 + m1] = 1.0
            c_pair[10 + m2] = 1.0
            c_pair /= np.linalg.norm(c_pair)
            candidates.append(c_pair)
    
    # Select top triplets (not all C(11,3)=165)
    rng = np.random.default_rng(123)
    triplets = []
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            for m3 in range(m2 + 1, 11):
                triplets.append((m1, m2, m3))
    # Shuffle and take a subset
    rng.shuffle(triplets)
    for m1, m2, m3 in triplets[:50]:
        c_triple = np.zeros(n_m)
        c_triple[10 + m1] = 1.0
        c_triple[10 + m2] = 1.0
        c_triple[10 + m3] = 1.0
        c_triple /= np.linalg.norm(c_triple)
        candidates.append(c_triple)
    
    # Random normalized vectors
    for _ in range(80):
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)
    
    # Random with emphasis on low-m modes
    for _ in range(50):
        c_rand = np.zeros(n_m)
        for m in range(0, 11):
            if rng.random() < 0.7:
                c_rand[10 + m] = rng.standard_normal()
        if np.linalg.norm(c_rand) > 1e-12:
            c_rand /= np.linalg.norm(c_rand)
            candidates.append(c_rand)

    # 7. Evaluate and sort candidates
    candidate_vals = []
    for c_init in candidates:
        if time.time() > deadline - 120:
            break
        val, _ = objective_and_grad(c_init)
        candidate_vals.append((val, c_init))
    
    candidate_vals.sort(key=lambda x: x[0])
    
    # 8. Optimize top candidates
    best_c = None
    best_val = np.inf
    n_to_optimize = min(30, len(candidate_vals))
    
    for i in range(n_to_optimize):
        if time.time() > deadline - 30:
            break
        val_init, c_init = candidate_vals[i]
        
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 30}
            )
            val = res.fun
            if val < best_val and np.all(np.isfinite(res.x)):
                best_val = val
                best_c = res.x.copy()
        except Exception:
            if val_init < best_val:
                best_val = val_init
                best_c = c_init.copy()
        
        # Sphere polish on the best result so far
        if best_c is not None and time.time() < deadline - 15:
            polished = sphere_polish(best_c, max_iter=500, lr_init=0.05)
            val_p, _ = objective_and_grad(polished)
            if val_p < best_val:
                best_val = val_p
                best_c = polished.copy()

    # 9. Final high-precision polish on the best candidate
    if best_c is not None and time.time() < deadline - 5:
        # Multiple polish passes with decreasing learning rate
        for lr in [0.03, 0.01, 0.005]:
            if time.time() > deadline - 5:
                break
            polished = sphere_polish(best_c, max_iter=600, lr_init=lr)
            val_p, _ = objective_and_grad(polished)
            if val_p < best_val:
                best_val = val_p
                best_c = polished.copy()

    # 10. Final Validity Check and Scaling
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
