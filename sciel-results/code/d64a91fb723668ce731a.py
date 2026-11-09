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
        
        norms = np.linalg.norm(pts, axis=1)
        max_norm = np.max(norms)
        if max_norm < 1e-12:
            return np.inf, np.zeros_like(x)
        pts = pts / max_norm
        
        dists = pdist(pts)
        
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
        
        exp_neg = np.exp(neg_d - max_neg_d)
        sum_exp_neg = np.sum(exp_neg)
        w_min = exp_neg / sum_exp_neg
        
        exp_pos = np.exp(pos_d - max_pos_d)
        sum_exp_pos = np.sum(exp_pos)
        w_max = exp_pos / sum_exp_pos
        
        grad = np.zeros((n, d))
        
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
                
                # d(-soft_min/soft_max)/d(d_k) = (w_min_k * soft_max - w_max_k * soft_min) / soft_max^2
                # But we want to minimize -soft_min/soft_max, so gradient of objective = -gradient of val
                # Actually val = -soft_min/soft_max, we want to minimize val
                # d(val)/d(d_k) = -[d(soft_min)/d(d_k) * soft_max - soft_min * d(soft_max)/d(d_k)] / soft_max^2
                # d(soft_min)/d(d_k) = -w_min_k
                # d(soft_max)/d(d_k) = w_max_k
                # d(val)/d(d_k) = -[-w_min_k * soft_max - soft_min * w_max_k] / soft_max^2
                # d(val)/d(d_k) = [w_min_k * soft_max + soft_min * w_max_k] / soft_max^2
                # Wait, let me recompute:
                # val = -soft_min / soft_max
                # d(val)/d(d_k) = -[d(soft_min)/d(d_k) * soft_max - soft_min * d(soft_max)/d(d_k)] / soft_max^2
                # = -[(-w_min_k) * soft_max - soft_min * w_max_k] / soft_max^2
                # = -[-w_min_k * soft_max - soft_min * w_max_k] / soft_max^2
                # = [w_min_k * soft_max + soft_min * w_max_k] / soft_max^2
                
                coeff = (w_m * soft_max + soft_min * w_x) / (soft_max * soft_max)
                
                # d(d_k)/d(pts[i]) = (pts[i] - pts[j]) / d_k
                # d(d_k)/d(pts[j]) = (pts[j] - pts[i]) / d_k = -diff / d_k
                grad[i] += coeff * diff / dist
                grad[j] -= coeff * diff / dist
                idx += 1
        
        grad = grad.flatten()
        
        return val, grad

    def make_objective(beta):
        def obj(x):
            val, _ = objective_and_grad(x, beta)
            return val
        def obj_grad(x):
            _, g = objective_and_grad(x, beta)
            return g
        return obj, obj_grad

    best_points = None
    best_score = -np.inf

    # Generate initial configurations
    initial_configs = []
    
    # 1. Icosahedron-based with better 2 extra points
    phi = (1 + np.sqrt(5)) / 2
    ico12 = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico12 = ico12 / np.linalg.norm(ico12, axis=1, keepdims=True)
    
    # Add 2 points at face centers of icosahedron (opposite faces)
    # Face centers of icosahedron are at (±1/√3, ±1/√3, ±1/√3) type positions
    # Actually for the icosahedron above, face centers are along the 6-fold axes
    # Let's use points along the z-axis at appropriate distance
    cfg1 = np.vstack([ico12, np.array([[0, 0, 0.72], [0, 0, -0.72]])])
    initial_configs.append(cfg1)
    
    # 2. Fibonacci sphere
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
    
    # 3. Octahedron + 8 more points
    oct6 = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    # Add 8 points at cube vertices
    cube8 = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ], dtype=float)
    cube8 = cube8 / np.linalg.norm(cube8, axis=1, keepdims=True)
    cfg2 = np.vstack([oct6, cube8])
    initial_configs.append(cfg2)
    
    # 4. Random restarts from perturbed icosahedron
    rng = np.random.RandomState(123)
    for i in range(30):
        perturbation = 0.03 + 0.15 * (i / 30)
        pert = ico12 + rng.randn(12, 3) * perturbation
        extra = rng.randn(2, 3) * 0.3
        cfg = np.vstack([pert, extra])
        initial_configs.append(cfg)
    
    # 5. More random configs
    for i in range(20):
        cfg = rng.randn(n, d)
        initial_configs.append(cfg)
    
    # 6. Perturbed Fibonacci
    for i in range(10):
        pert = fib14 + rng.randn(n, d) * 0.1
        initial_configs.append(pert)

    # Beta annealing schedule - start higher for better conditioning
    beta_schedule = [10, 25, 50, 100, 200, 500, 1000, 2000]
    
    # Optimize with multiple restarts and beta annealing
    for i, init in enumerate(initial_configs):
        if time.time() > deadline - 40:
            break
        
        current_pts = init.copy()
        
        # Multi-stage beta annealing
        for beta in beta_schedule:
            if time.time() > deadline - 20:
                break
                
            obj, obj_grad = make_objective(beta)
            
            try:
                result = minimize(
                    obj,
                    current_pts.flatten(),
                    jac=obj_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                
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
        
        # Polish with Nelder-Mead on exact objective
        if time.time() < deadline - 15:
            def obj_nm(x):
                pts_nm = x.reshape((n, d))
                return -score(pts_nm)
            
            try:
                result_nm = minimize(
                    obj_nm,
                    current_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
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
        
    # Final normalization
    best_points = best_points - np.mean(best_points, axis=0)
    norm_max = np.max(np.linalg.norm(best_points, axis=1))
    if norm_max > 0:
        best_points = best_points / norm_max * 100

    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))

    return best_points
