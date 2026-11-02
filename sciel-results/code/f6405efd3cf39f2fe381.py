"""Packing of 100 equal spheres in a unit cube using smooth optimization."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 100

def _compute_radius(c):
    """Compute the largest common radius for centres c (N,3) inside [-0.5,0.5]^3."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c > 0.5) or np.any(c < -0.5):
        return 0.0
    # Wall distances
    wall = 0.5 - np.abs(c)
    min_wall = wall.min()
    if min_wall <= 0:
        return 0.0
    # Pairwise distances
    tree = cKDTree(c)
    dists, _ = tree.query(c, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    return min(min_wall, min_pair)

def _soft_min_objective(x, beta):
    """
    Smooth approximation of -min(distances) using log-sum-exp.
    Maximizing this is equivalent to maximizing the minimum distance.
    """
    c = x.reshape(N, 3)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    
    # Pairwise distances (upper triangle only)
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    pair_dists = np.sqrt(np.sum(diff**2, axis=1)) / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([wall_dists.ravel(), pair_dists])
    
    # Soft-min via log-sum-exp: -log(sum(exp(-beta*d))) / beta ≈ min(d) for large beta
    # Use stable log-sum-exp
    m = np.min(all_dists)
    if m < -1000:
        return 1e10, np.zeros_like(x)
    shifted = -beta * (all_dists - m)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = m + log_sum / beta
    
    obj = -soft_min_val  # We want to maximize soft_min, so minimize -soft_min
    
    # Gradient
    grad_c = np.zeros_like(c)
    
    # Wall gradient
    exp_weights_wall = np.exp(-beta * (wall_dists.ravel() - m))
    sum_exp = np.sum(np.exp(shifted))
    weights_wall = exp_weights_wall / sum_exp
    for k in range(3):
        grad_c[:, k] += weights_wall[:N] * np.sign(c[:, k])
    
    # Pair gradient
    exp_weights_pair = np.exp(-beta * (pair_dists - m))
    weights_pair = exp_weights_pair / sum_exp
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        d = 2 * pair_dists[idx]
        if d > 1e-12:
            direction = (c[i] - c[j]) / d
            grad_c[i] += weights_pair[idx] * direction
            grad_c[j] -= weights_pair[idx] * direction
    
    return obj, grad_c.ravel()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin (150s limit)
    
    # Initial: 5x5x4 grid with perturbation
    m = 5
    n = 4
    xs = np.linspace(-0.35, 0.35, m)
    ys = np.linspace(-0.35, 0.35, m)
    zs = np.linspace(-0.35, 0.35, n)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    centers = grid[:N].copy()
    
    rng = np.random.RandomState(42)
    centers += rng.uniform(-0.02, 0.02, size=centers.shape)
    centers = np.clip(centers, -0.45, 0.45)
    
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    # Phase 1: Coarse optimization with small beta
    if time.time() < deadline - 40:
        x0 = centers.ravel()
        beta = 50.0
        try:
            result = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.4999, 0.4999)] * (N * 3),
                options={'maxiter': 200, 'maxfun': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_centers = np.clip(result.x.reshape(N, 3), -0.4999, 0.4999)
            new_r = _compute_radius(new_centers)
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers
        except Exception:
            pass
    
    # Phase 2: Fine optimization with larger beta
    if time.time() < deadline - 20:
        x0 = best_centers.ravel()
        beta = 200.0
        try:
            result = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.4999, 0.4999)] * (N * 3),
                options={'maxiter': 300, 'maxfun': 400, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            new_centers = np.clip(result.x.reshape(N, 3), -0.4999, 0.4999)
            new_r = _compute_radius(new_centers)
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers
        except Exception:
            pass
    
    # Phase 3: Very fine optimization with very large beta
    if time.time() < deadline - 5:
        x0 = best_centers.ravel()
        beta = 500.0
        try:
            result = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.4999, 0.4999)] * (N * 3),
                options={'maxiter': 400, 'maxfun': 500, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            new_centers = np.clip(result.x.reshape(N, 3), -0.4999, 0.4999)
            new_r = _compute_radius(new_centers)
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers
        except Exception:
            pass
    
    # Phase 4: Local force-directed polish
    if time.time() < deadline - 1:
        centers = best_centers.copy()
        base_step = 0.01
        decay_rate = 0.9995
        stagnation = 0
        it = 0
        while time.time() < deadline - 0.5:
            it += 1
            wall_dists = 0.5 - np.abs(centers)
            r_wall = np.min(wall_dists)
            tree = cKDTree(centers)
            dists, _ = tree.query(centers, k=2)
            r_pair = np.min(dists[:, 1]) / 2.0
            current_r = min(r_wall, r_pair)
            
            if current_r <= best_r * 1.000001:
                stagnation += 1
                if stagnation > 200:
                    break
            else:
                stagnation = 0
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            force = np.zeros_like(centers)
            target_r = current_r + 0.001
            
            for k in range(3):
                d_wall = 0.5 - np.abs(centers[:, k])
                mask = d_wall < target_r
                if np.any(mask):
                    viol = target_r - d_wall[mask]
                    dirs = -np.sign(centers[mask, k])
                    dirs[dirs == 0] = 1.0
                    force[mask, k] += viol * dirs
            
            target_dist = 2 * target_r
            try:
                pairs = tree.query_pairs(target_dist)
            except Exception:
                pairs = []
            
            for i, j in pairs:
                diff = centers[i] - centers[j]
                d = np.linalg.norm(diff)
                if d > 1e-10 and d < target_dist:
                    direction = diff / d
                    viol = target_dist - d
                    force[i] += viol * direction
                    force[j] -= viol * direction
            
            norm_force = np.linalg.norm(force)
            if norm_force > 1e-10:
                step = base_step * decay_rate ** it
                centers += step * force / norm_force
                centers = np.clip(centers, -0.499999, 0.499999)
    
    return best_centers
