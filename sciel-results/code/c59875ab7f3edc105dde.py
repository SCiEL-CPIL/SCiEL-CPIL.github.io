import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 590  # safety margin for 630s kill

    def score(pts):
        pts = pts - np.mean(pts, axis=0)
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    # Icosahedron vertices (12 points)
    phi = (1 + np.sqrt(5)) / 2
    ico12 = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico12 = ico12 / np.linalg.norm(ico12, axis=1, keepdims=True)
    
    # Build initial configurations
    initial_configs = []
    
    # Config 1: Icosahedron + 2 points at ±z direction
    cfg1 = np.vstack([ico12, np.array([[0, 0, 0.8], [0, 0, -0.8]])])
    initial_configs.append(cfg1)
    
    # Config 2: Fibonacci sphere
    def fibonacci_sphere(n_pts):
        indices = np.arange(0, n_pts, dtype=float) + 0.5
        phi_f = np.arccos(1 - 2 * indices / n_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.cos(theta) * np.sin(phi_f)
        y = np.sin(theta) * np.sin(phi_f)
        z = np.cos(phi_f)
        return np.column_stack([x, y, z])
    
    fib14 = fibonacci_sphere(14)
    initial_configs.append(fib14)
    
    # Random restarts
    rng = np.random.RandomState(42)
    for i in range(50):
        perturbation = 0.05 + 0.3 * (i / 50)
        pert = ico12 + rng.randn(12, 3) * perturbation
        extra = rng.randn(2, 3) * 0.5
        cfg = np.vstack([pert, extra])
        initial_configs.append(cfg)
    
    for i in range(40):
        cfg = rng.randn(n, d)
        initial_configs.append(cfg)
    
    # Smooth objective with soft-min/soft-max
    def objective(x, beta):
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        dists = pdist(pts)
        if np.max(dists) < 1e-12:
            return 0.0
        
        neg_d = -beta * dists
        max_neg_d = np.max(neg_d)
        log_sum_exp_neg = max_neg_d + np.log(np.sum(np.exp(neg_d - max_neg_d)))
        soft_min = -log_sum_exp_neg / beta
        
        pos_d = beta * dists
        max_pos_d = np.max(pos_d)
        log_sum_exp_pos = max_pos_d + np.log(np.sum(np.exp(pos_d - max_pos_d)))
        soft_max = log_sum_exp_pos / beta
        
        if soft_max < 1e-12:
            return 0.0
        return -(soft_min / soft_max) ** 2
    
    def objective_grad(x, beta):
        pts = x.reshape((n, d))
        pts_centered = pts - np.mean(pts, axis=0)
        
        i_idx, j_idx = np.triu_indices(n, k=1)
        diffs = pts_centered[i_idx] - pts_centered[j_idx]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        
        if np.max(dists) < 1e-12:
            return np.zeros_like(x)
        
        neg_d = -beta * dists
        max_neg_d = np.max(neg_d)
        exp_neg = np.exp(neg_d - max_neg_d)
        sum_exp_neg = np.sum(exp_neg)
        w_min = exp_neg / sum_exp_neg
        
        pos_d = beta * dists
        max_pos_d = np.max(pos_d)
        exp_pos = np.exp(pos_d - max_pos_d)
        sum_exp_pos = np.sum(exp_pos)
        w_max = exp_pos / sum_exp_pos
        
        log_sum_exp_neg = max_neg_d + np.log(sum_exp_neg)
        soft_min = -log_sum_exp_neg / beta
        
        log_sum_exp_pos = max_pos_d + np.log(sum_exp_pos)
        soft_max = log_sum_exp_pos / beta
        
        if soft_max < 1e-12:
            return np.zeros_like(x)
        
        ratio = soft_min / soft_max
        coeff = 2 * ratio * (w_min * soft_max - w_max * soft_min) / (soft_max ** 2)
        coeff = coeff / (dists + 1e-15)
        
        grad = np.zeros((n, d))
        for k in range(len(i_idx)):
            if dists[k] > 1e-12:
                unit = diffs[k] / dists[k]
                grad[i_idx[k]] += coeff[k] * unit
                grad[j_idx[k]] -= coeff[k] * unit
        
        sum_grad = np.sum(grad, axis=0)
        grad = grad - (1.0 / n) * sum_grad
        
        return -grad.flatten()
    
    best_points = None
    best_score = -np.inf
    
    # Optimize from multiple initial configurations
    for i, init in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        current_pts = init.copy()
        
        # Stage 1: Coarse optimization with increasing beta
        for beta in [10, 50, 200, 500]:
            if time.time() > deadline - 20:
                break
                
            try:
                result = minimize(
                    lambda x: objective(x, beta),
                    current_pts.flatten(),
                    jac=lambda x: objective_grad(x, beta),
                    method='L-BFGS-B',
                    options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                
                if np.all(np.isfinite(result.x)):
                    new_pts = result.x.reshape((n, d))
                    if np.all(np.isfinite(new_pts)):
                        current_pts = new_pts
                        
            except Exception:
                pass
        
        # Evaluate and update best
        s = score(current_pts)
        if s > best_score:
            best_score = s
            best_points = current_pts.copy()
        
        # Stage 2: Fine optimization with Nelder-Mead on exact objective
        if time.time() < deadline - 10:
            def obj_nm(x):
                pts_nm = x.reshape((n, d))
                return -score(pts_nm)
            
            try:
                result_nm = minimize(
                    obj_nm,
                    current_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True}
                )
                pts_nm = result_nm.x.reshape((n, d))
                if np.all(np.isfinite(pts_nm)):
                    s_nm = score(pts_nm)
                    if s_nm > best_score:
                        best_score = s_nm
                        best_points = pts_nm.copy()
                        
            except Exception:
                pass
    
    if best_points is None:
        best_points = initial_configs[0]
        
    # Final normalization: center only, no scaling (metric is scale-invariant)
    best_points = best_points - np.mean(best_points, axis=0)

    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))

    return best_points
