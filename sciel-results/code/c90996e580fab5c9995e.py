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

    def coulomb_obj_and_grad(x, k=50.0):
        """Coulomb repulsion to spread points evenly."""
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        
        # Compute all pairwise differences
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(dists_sq, 1e-10)
        dists = np.sqrt(dists_sq)
        
        # Coulomb potential: sum of 1/dist for all pairs
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_dists = 1.0 / dists
        np.fill_diagonal(inv_dists, 0.0)
        
        val = k * np.sum(inv_dists)
        
        # Gradient
        grad = np.zeros((n, d))
        for i in range(n):
            for j in range(i + 1, n):
                dist = dists[i, j]
                if dist < 1e-10:
                    continue
                diff = pts[i] - pts[j]
                coeff = k / (dist ** 2)
                grad[i] += coeff * diff
                grad[j] -= coeff * diff
        
        return val, grad.flatten()

    def make_coulomb():
        def obj(x):
            val, _ = coulomb_obj_and_grad(x)
            return val
        def obj_grad(x):
            _, g = coulomb_obj_and_grad(x)
            return g
        return obj, obj_grad

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
        
        # Vectorized gradient computation
        # Create index pairs
        i_idx, j_idx = np.triu_indices(n, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        dists_vec = dists
        
        valid = dists_vec > 1e-12
        if not np.any(valid):
            return val, np.zeros_like(x)
        
        w_m = w_min[valid]
        w_x = w_max[valid]
        d_valid = dists_vec[valid]
        diff_valid = diffs[valid]
        
        coeff = (w_m * soft_max - w_x * soft_min) / (soft_max * soft_max)
        coeff = coeff / d_valid
        
        grad = np.zeros((n, d))
        for k in range(len(i_idx)):
            if not valid[k]:
                continue
            grad[i_idx[k]] += coeff[k] * diff_valid[k]
            grad[j_idx[k]] -= coeff[k] * diff_valid[k]
        
        grad = -grad.flatten()
        
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
    
    # 1. Icosahedron-based
    phi = (1 + np.sqrt(5)) / 2
    ico12 = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico12 = ico12 / np.linalg.norm(ico12, axis=1, keepdims=True)
    
    # Add 2 points near opposite face centers
    cfg1 = np.vstack([ico12, np.array([[0, 0, 0.7], [0, 0, -0.7]])])
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
    
    # 3. Random restarts from perturbed icosahedron
    rng = np.random.RandomState(123)
    for i in range(30):
        perturbation = 0.03 + 0.15 * (i / 30)
        pert = ico12 + rng.randn(12, 3) * perturbation
        extra = rng.randn(2, 3) * 0.3
        cfg = np.vstack([pert, extra])
        initial_configs.append(cfg)
    
    # 4. More random configs
    for i in range(20):
        cfg = rng.randn(n, d)
        initial_configs.append(cfg)
    
    # 5. Additional structured configs
    # Regular tetrahedron extended
    tetra = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    tetra = tetra / np.linalg.norm(tetra, axis=1, keepdims=True)
    
    # Create 14 points by scaling and rotating tetrahedra
    for angle in [0, np.pi/3, 2*np.pi/3]:
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        scaled_tetra = tetra @ rot.T
        # Add 3 more points from another tetrahedron rotated
        rot2 = np.array([
            [1, 0, 0],
            [0, np.cos(np.pi/5), -np.sin(np.pi/5)],
            [0, np.sin(np.pi/5), np.cos(np.pi/5)]
        ])
        scaled_tetra2 = tetra @ rot2.T
        # Take 7 from each
        cfg = np.vstack([scaled_tetra[:7], scaled_tetra2[:7]])
        initial_configs.append(cfg)

    # Beta annealing schedule
    beta_schedule = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    
    coulomb_obj, coulomb_grad = make_coulomb()
    
    # Optimize with multiple restarts and beta annealing
    for i, init in enumerate(initial_configs):
        if time.time() > deadline - 25:
            break
        
        current_pts = init.copy()
        
        # Stage 1: Coulomb repulsion to spread points
        if time.time() < deadline - 20:
            try:
                result_coulomb = minimize(
                    coulomb_obj,
                    current_pts.flatten(),
                    jac=coulomb_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                new_pts = result_coulomb.x.reshape((n, d))
                if np.all(np.isfinite(new_pts)):
                    current_pts = new_pts
            except Exception:
                pass
        
        # Stage 2: Multi-stage beta annealing
        for beta in beta_schedule:
            if time.time() > deadline - 15:
                break
                
            obj, obj_grad = make_objective(beta)
            
            try:
                result = minimize(
                    obj,
                    current_pts.flatten(),
                    jac=obj_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
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
        if time.time() < deadline - 10:
            def obj_nm(x):
                pts_nm = x.reshape((n, d))
                return -score(pts_nm)
            
            try:
                result_nm = minimize(
                    obj_nm,
                    current_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
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
