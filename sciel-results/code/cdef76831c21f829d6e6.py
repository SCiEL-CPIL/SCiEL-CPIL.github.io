"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_score = -1.0
    best_points = None
    
    def calculate_score(points):
        if points is None:
            return 0.0
        pts = points.reshape(N, 3)
        if not np.all(np.isfinite(pts)):
            return 0.0
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 0.0
            
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary_dists)
        
        r = min(min_pair / 2.0, min_boundary)
        return r

    def objective_and_grad(x, alpha):
        pts = x.reshape(N, 3)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Avoid numerical issues with very small distances
        dists_safe = np.maximum(dists, 1e-10)
        
        # Log-Sum-Exp for pairwise part
        # f_pair = alpha * log(sum(exp(-dists/alpha)))
        # Gradient: grad_i = sum_j (exp(-d_ij/alpha) / sum(exp(-d_kl/alpha))) * (1/alpha) * (x_i - x_j) / d_ij * d_ij 
        # Actually, let's derive carefully.
        # LSE = alpha * log(sum_j exp(-d_ij/alpha))
        # d(LSE)/d(x_i) = alpha * (1/sum_j exp(-d_ij/alpha)) * sum_j (d(exp(-d_ij/alpha))/d(x_i))
        # d(exp(-d_ij/alpha))/d(x_i) = exp(-d_ij/alpha) * (-1/alpha) * d(d_ij)/d(x_i)
        # d(d_ij)/d(x_i) = (x_i - x_j) / d_ij
        # So d(LSE)/d(x_i) = alpha * (1/S) * sum_j exp(-d_ij/alpha) * (-1/alpha) * (x_i - x_j)/d_ij
        # d(LSE)/d(x_i) = -(1/S) * sum_j exp(-d_ij/alpha) * (x_i - x_j)/d_ij
        
        # To avoid underflow/overflow, we can shift the exponents
        # exp(-d/alpha) is very small for large d.
        # Let's use the stable log-sum-exp trick if alpha is small, but here alpha is ~0.1, dists ~0.2, so -d/alpha ~ -2. exp(-2) is fine.
        
        exp_terms = np.exp(-dists_safe / alpha)
        np.fill_diagonal(exp_terms, 0.0)
        
        sum_exp = np.sum(exp_terms, axis=1)
        
        # Objective value for pairwise part
        obj_pair = np.sum(alpha * np.log(sum_exp + 1e-10))
        
        # Gradient for pairwise part
        # We need vectorized computation of sum_j exp(-d_ij/alpha) * (x_i - x_j)/d_ij
        # Weight matrix W_ij = exp(-d_ij/alpha) / sum_exp_i
        W = exp_terms / sum_exp[:, np.newaxis]
        np.fill_diagonal(W, 0.0)
        
        # Term (x_i - x_j) / d_ij
        # This is the unit vector from j to i.
        # Let U_ij = (x_i - x_j) / d_ij
        # We can compute this as:
        # U = diff / dists[:, :, np.newaxis]
        U = diff / dists_safe[:, :, np.newaxis]
        # Set diagonal to 0 to avoid inf*0
        np.fill_diagonal(U, 0.0)
        
        # grad_pair_i = - sum_j W_ij * U_ij
        grad_pair = -np.sum(W[:, :, np.newaxis] * U, axis=1)
        
        # Boundary part
        # We want to maximize min(0.5 - |x_i|)
        # Soft-min approximation: alpha * log(sum_k exp(-(0.5 - |x_ik|)/alpha))
        # Let b_ik = 0.5 - |x_ik|
        # d(b_ik)/dx_ik = -sign(x_ik)
        
        abs_pts = np.abs(pts)
        b_dists = 0.5 - abs_pts
        b_dists_safe = np.maximum(b_dists, 1e-10)
        
        exp_b = np.exp(-b_dists_safe / alpha)
        sum_exp_b = np.sum(exp_b, axis=1)
        
        obj_bound = np.sum(alpha * np.log(sum_exp_b + 1e-10))
        
        # Gradient for boundary part
        # d(LSE_b)/dx_ik = -(1/sum_exp_b_i) * sum_k exp(-b_ik/alpha) * d(b_ik)/dx_ik
        # d(b_ik)/dx_ik = -sign(x_ik)
        # So d(LSE_b)/dx_ik = (1/sum_exp_b_i) * sum_k exp(-b_ik/alpha) * sign(x_ik)
        # Wait, the sum is over k for each i.
        # grad_bound_ik = (1/sum_exp_b_i) * exp(-b_ik/alpha) * sign(x_ik)
        
        W_b = exp_b / sum_exp_b[:, np.newaxis]
        signs = np.sign(pts)
        # Handle sign(0) -> 1
        signs[signs == 0] = 1.0
        
        grad_bound = W_b * signs
        
        total_obj = obj_pair + obj_bound
        total_grad = (grad_pair + grad_bound).flatten()
        
        return total_obj, total_grad

    def init_grid(m1, m2, m3):
        tx = (np.arange(m1) + 0.5) / m1 - 0.5
        ty = (np.arange(m2) + 0.5) / m2 - 0.5
        tz = (np.arange(m3) + 0.5) / m3 - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        return grid.flatten()

    def init_random():
        return np.random.uniform(-0.45, 0.45, size=(N, 3)).flatten()

    # More diverse initial conditions
    np.random.seed(42)
    starts = [
        init_grid(5, 5, 4),  # 100 points
        init_grid(4, 5, 5),
        init_grid(5, 4, 5),
        init_grid(5, 5, 4) + np.random.normal(0, 0.02, size=(N*3,)),
        init_grid(4, 5, 5) + np.random.normal(0, 0.02, size=(N*3,)),
        init_grid(5, 4, 5) + np.random.normal(0, 0.02, size=(N*3,)),
        init_random(),
        init_random(),
    ]
    
    # Clip starts to valid range
    for i in range(len(starts)):
        starts[i] = np.clip(starts[i], -0.48, 0.48)
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Two-phase approach: coarse then fine optimization
    # Phase 1: Use moderate alpha for exploration
    # Phase 2: Use small alpha for refinement
    
    alphas_coarse = [0.3, 0.15]
    alphas_fine = [0.05, 0.02, 0.01]
    
    for i, x0 in enumerate(starts):
        if time.time() > deadline - 15:
            break
            
        current_x = x0.copy()
        
        # Coarse phase
        for alpha in alphas_coarse:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    lambda x: objective_and_grad(x, alpha)[0],
                    current_x,
                    jac=lambda x: objective_and_grad(x, alpha)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30}
                )
                if res.fun < 1e10:
                    current_x = res.x
                    pts = np.clip(current_x.reshape(N, 3), -0.5, 0.5)
                    score = calculate_score(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
            except Exception:
                continue
        
        # Fine phase
        for alpha in alphas_fine:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    lambda x: objective_and_grad(x, alpha)[0],
                    current_x,
                    jac=lambda x: objective_and_grad(x, alpha)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                if res.fun < 1e10:
                    current_x = res.x
                    pts = np.clip(current_x.reshape(N, 3), -0.5, 0.5)
                    score = calculate_score(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
            except Exception:
                continue
    
    # Final polishing with very small alpha
    if best_points is not None and time.time() < deadline - 8:
        x0 = best_points.flatten()
        for alpha in [0.005, 0.002]:
            if time.time() > deadline - 3:
                break
            try:
                res = minimize(
                    lambda x: objective_and_grad(x, alpha)[0],
                    x0,
                    jac=lambda x: objective_and_grad(x, alpha)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 40}
                )
                if res.fun < 1e10:
                    pts = res.x.reshape(N, 3)
                    pts = np.clip(pts, -0.5, 0.5)
                    score = calculate_score(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
                        x0 = pts.flatten()
            except Exception:
                pass

    if best_points is None:
        best_points = init_grid(5, 5, 4).reshape(N, 3)
        
    return best_points
# EVOLVE-BLOCK-END
