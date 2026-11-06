"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 140.0  # keep a safety margin
    
    # Generate hexagonal lattice points inside unit disk
    def generate_hex_lattice():
        best_points = None
        for spacing in np.linspace(0.055, 0.045, 30):
            k = int(1.0 / spacing) + 2
            points = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    r = np.hypot(x, y)
                    if r <= 1.0 - spacing / 2.0:
                        points.append((x, y))
            points = np.array(points)
            if len(points) >= N:
                # Sort by radius and take first N
                radii = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(radii, kind='stable')[:N]
                if best_points is None or len(points) > len(best_points):
                    best_points = points[idx]
                break
        if best_points is None:
            # Fallback: uniform random
            rng = np.random.RandomState(42)
            while True:
                pts = rng.uniform(-1, 1, size=(N, 2))
                radii = np.hypot(pts[:, 0], pts[:, 1])
                if np.all(radii <= 0.99):
                    best_points = pts
                    break
        return best_points
    
    # Compute the actual score
    def compute_radius(pts):
        n = len(pts)
        tree = cKDTree(pts)
        # Find nearest neighbors for each point
        dists, _ = tree.query(pts, k=2)
        min_pair_dist = np.min(dists[:, 1])  # second nearest is the closest other point
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        return min(min_pair_dist / 2.0, min_boundary)
    
    # Smooth objective using KD-tree for efficiency
    def smooth_objective(x, t, target_r):
        p = x.reshape(-1, 2)
        n = len(p)
        
        # Build KD-tree
        tree = cKDTree(p)
        
        # Find k nearest neighbors
        k = min(n, 20)  # use more neighbors for better approximation
        dists, _ = tree.query(p, k=k)
        
        # Soft-min of pair distances using log-sum-exp
        # Exclude self (first column)
        pair_dists = dists[:, 1:]
        
        # Soft-min approximation
        min_d = np.min(pair_dists, axis=1)
        avg_d = np.mean(pair_dists, axis=1)
        
        # Boundary: soft-max of radii
        radii = np.hypot(p[:, 0], p[:, 1])
        
        # We want to maximize min(min_pair_dist/2, 1 - max_radius)
        # Minimize: max(max_radius, 1 - min_pair_dist/2)
        
        # Smooth max via log-sum-exp
        a = radii
        b = 1.0 - pair_dists / 2.0
        
        # Combine into single soft-max
        # min_pair_dist/2 is at least the min of all pair_dists/2
        # Let's use: maximize r such that all constraints satisfied
        # Penalty: sum of violations
        
        # For each pair, if dist < 2*r, penalty
        # For each point, if radius > 1-r, penalty
        
        # Use current target_r as estimate
        pair_violation = np.sum(np.maximum(0, 2*target_r - pair_dists)**2)
        boundary_violation = np.sum(np.maximum(0, radii - (1-target_r))**2)
        
        # Also add a term to encourage larger minimum distances
        # Soft-min of all pair distances
        log_sum_exp = -t * np.log(np.sum(np.exp(-pair_dists / t), axis=1))
        soft_min_pair = np.mean(log_sum_exp)
        
        # Soft-max of radii
        log_sum_exp_r = t * np.log(np.sum(np.exp(radii / t)))
        soft_max_radius = log_sum_exp_r
        
        # Objective: minimize (soft_max_radius - soft_min_pair/2 + violations)
        obj = soft_max_radius - soft_min_pair / 2.0 + 0.1 * (pair_violation + boundary_violation)
        
        return obj
    
    # Compute gradient numerically for L-BFGS-B
    def objective_with_grad(x, t, target_r):
        f = smooth_objective(x, t, target_r)
        # Numerical gradient
        eps = 1e-7
        n = len(x)
        grad = np.zeros(n)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = smooth_objective(x_plus, t, target_r)
            grad[i] = (f_plus - f) / eps
        return f, grad
    
    # Main optimization loop with annealing
    pts = generate_hex_lattice().astype(np.float64)
    current_score = compute_radius(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # Annealing schedule
    n_restarts = 8
    t_start = 0.01
    t_end = 0.001
    
    for restart in range(n_restarts):
        if time.time() > deadline - 25.0:
            break
        
        if restart == 0:
            start_pts = pts.copy()
        else:
            # Perturb based on current best
            rng = np.random.RandomState(restart * 1000 + 42)
            noise_scale = 0.002 * (1 + restart * 0.5)
            start_pts = best_pts + rng.normal(0, noise_scale, size=best_pts.shape)
        
        x0 = start_pts.flatten()
        
        # Multi-stage annealing
        n_stages = 3
        for stage in range(n_stages):
            if time.time() > deadline - 15.0:
                break
            
            t = t_start * (t_end / t_start) ** (stage / max(n_stages - 1, 1))
            target_r = best_score * (1 + 0.05 * stage)
            
            try:
                # Use L-BFGS-B with numerical gradient
                res = minimize(
                    lambda x: smooth_objective(x, t, target_r),
                    x0,
                    jac=lambda x: None,  # let scipy compute numerically
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30}
                )
                
                if time.time() > deadline - 10.0:
                    break
                
                new_pts = res.x.reshape(-1, 2)
                new_score = compute_radius(new_pts)
                
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    x0 = new_pts.flatten()
            except Exception:
                continue
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        x0 = best_pts.flatten()
        t_final = 0.0005
        target_r = best_score
        
        try:
            res = minimize(
                lambda x: smooth_objective(x, t_final, target_r),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            new_pts = res.x.reshape(-1, 2)
            new_score = compute_radius(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
