import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580

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

    def objective_and_grad(x, beta):
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        
        # Normalize to unit max norm to prevent scale drift
        norms = np.linalg.norm(pts, axis=1)
        max_norm = np.max(norms)
        if max_norm < 1e-12:
            return np.inf, np.zeros_like(x)
        pts = pts / max_norm
        
        dists = pdist(pts)
        
        # Soft min and max
        # softmin = -1/beta * log(sum(exp(-beta * d)))
        # softmax = 1/beta * log(sum(exp(beta * d)))
        
        # For numerical stability, use log-sum-exp trick
        # log(sum(exp(a))) = max(a) + log(sum(exp(a - max(a))))
        
        neg_d = -beta * dists
        max_neg_d = np.max(neg_d)
        log_sum_exp_neg = max_neg_d + np.log(np.sum(np.exp(neg_d - max_neg_d)))
        soft_min = -log_sum_exp_neg / beta
        
        pos_d = beta * dists
        max_pos_d = np.max(pos_d)
        log_sum_exp_pos = max_pos_d + np.log(np.sum(np.exp(pos_d - max_pos_d)))
        soft_max = log_sum_exp_pos / beta
        
        if soft_max < 1e-12:
            return np.inf, np.zeros_like(x)
            
        val = -soft_min / soft_max
        
        # Gradient calculation
        # d(soft_min)/d(d_i) = exp(-beta * d_i) / sum(exp(-beta * d_j))
        # d(soft_max)/d(d_i) = exp(beta * d_i) / sum(exp(beta * d_j))
        
        exp_neg = np.exp(neg_d - max_neg_d)
        sum_exp_neg = np.sum(exp_neg)
        w_min = exp_neg / sum_exp_neg
        
        exp_pos = np.exp(pos_d - max_pos_d)
        sum_exp_pos = np.sum(exp_pos)
        w_max = exp_pos / sum_exp_pos
        
        # d(soft_min)/dx = sum_i w_min_i * d(d_i)/dx
        # d(soft_max)/dx = sum_i w_max_i * d(d_i)/dx
        
        # d(d_ij)/dx_k = (x_i - x_j) / d_ij
        # We need to accumulate gradients for all points.
        
        grad = np.zeros((n, d))
        
        # Loop over all pairs
        # This is O(n^2 * d) which is 14^2 * 3 = 588 ops per gradient evaluation.
        # This is fast enough.
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                w_m = w_min[idx]
                w_x = w_max[idx]
                dist = dists[idx]
                if dist < 1e-12:
                    idx += 1
                    continue
                diff = pts[i] - pts[j]
                # d(d_ij)/dx_i = diff / d_ij
                # d(d_ij)/dx_j = -diff / d_ij
                grad[i] += (w_m - (soft_min / soft_max**2) * w_x * soft_min) * diff / dist # Wait, chain rule
                # Let R = soft_min / soft_max
                # dR/dx = (soft_min' * soft_max - soft_min * soft_max') / soft_max^2
                # soft_min' = sum w_min * d(d)/dx
                # soft_max' = sum w_max * d(d)/dx
                # So dR/dx = sum_i [ w_min_i * soft_max - w_max_i * soft_min ] / soft_max^2 * d(d_i)/dx
                # coeff = (w_min_i * soft_max - w_max_i * soft_min) / soft_max^2
                
                coeff = (w_m * soft_max - w_x * soft_min) / (soft_max * soft_max)
                
                grad[i] += coeff * diff / dist
                grad[j] -= coeff * diff / dist
                idx += 1
        
        # The objective is -R, so grad_obj = -grad_R
        grad = -grad.flatten()
        
        return val, grad

    def objective(x):
        val, _ = objective_and_grad(x, 50.0)
        return val

    def grad(x):
        _, g = objective_and_grad(x, 50.0)
        return g

    best_points = None
    best_score = -np.inf

    # Generate initial configurations
    initial_configs = []
    
    # 1. Icosahedron + 2 face centers
    phi = (1 + np.sqrt(5)) / 2
    ico12 = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico12 = ico12 / np.linalg.norm(ico12, axis=1, keepdims=True)
    
    # Find face centers
    # Icosahedron faces are triangles.
    # We can just pick 2 opposite vertices and move them slightly?
    # Or use the 12 vertices and add 2 points at (0,0,1) and (0,0,-1) after rotating?
    # Let's just use ico12 + 2 random points near the center of opposite faces.
    # To find face centers, we need the adjacency.
    # Simpler: Just use ico12 and 2 points at (0, 0, 1.1) and (0, 0, -1.1) in a rotated frame?
    # Let's just use ico12 + 2 points at (0, 0, 1) and (0, 0, -1) scaled.
    # Actually, the icosahedron has 6 axes of symmetry.
    # Let's add points at (0, 0, 1.2) and (0, 0, -1.2).
    cfg1 = np.vstack([ico12, np.array([[0, 0, 1.2], [0, 0, -1.2]])])
    initial_configs.append(cfg1)
    
    # 2. Icosahedron + 2 points at face centers (approx)
    # The distance from center to face center is 1/phi.
    # Let's place points at distance 1/phi along the 6 axes.
    # The 6 axes are (1,0,0), (0,1,0), (0,0,1) in the icosahedron coordinate system?
    # No, the vertices are not on axes.
    # Let's just use random restarts with the icosahedron as base.
    
    rng = np.random.RandomState(42)
    for i in range(10):
        # Perturb icosahedron
        pert = ico12 + rng.randn(12, 3) * 0.1
        # Add 2 points
        extra = rng.randn(2, 3) * 0.5
        cfg = np.vstack([pert, extra])
        initial_configs.append(cfg)
        
    # 3. Fibonacci sphere
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
    
    # Optimize
    for i, init in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        init_flat = init.flatten()
        
        try:
            # Use L-BFGS-B with analytical gradient
            result = minimize(
                objective,
                init_flat,
                jac=grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            pts = result.x.reshape((n, d))
            s = score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
                
            # Polish with Nelder-Mead
            if time.time() < deadline - 10:
                def obj_nm(x):
                    pts_nm = x.reshape((n, d))
                    return -score(pts_nm)
                
                result_nm = minimize(
                    obj_nm,
                    pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                pts_nm = result_nm.x.reshape((n, d))
                s_nm = score(pts_nm)
                if s_nm > best_score:
                    best_score = s_nm
                    best_points = pts_nm
                    
        except Exception:
            pass

    if best_points is None:
        best_points = initial_configs[0]
        
    # Final normalization
    best_points = best_points - np.mean(best_points, axis=0)
    norm_max = np.max(np.linalg.norm(best_points, axis=1))
    if norm_max > 0:
        best_points = best_points / norm_max * 100

    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))

    return best_points
