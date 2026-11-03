"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_deadline = 0.0

def _compute_score(points):
    """Compute the maximum common radius for given points."""
    if len(points) < 2:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(len(points), k=1)
    min_pair = np.min(dists[iu]) / 2.0
    norms = np.hypot(points[:, 0], points[:, 1])
    min_boundary = 1.0 - np.max(norms)
    return min(min_pair, min_boundary)

def _ring_to_points(params, ring_counts):
    """Convert ring parameters to point coordinates.
    params: [r0, r1, ..., r_{K-1}, a0, a1, ..., a_{K-1}]
    ring_counts: list of integers for each ring.
    """
    n_rings = len(ring_counts)
    radii = params[:n_rings]
    offsets = params[n_rings:]
    
    # Ensure radii are positive and sorted (though DE might not sort them, we map them to rings)
    # For the specific structure 1,6,12,18,13, we assume the order is fixed.
    # r0 is for ring 1 (center, so r0 should be 0, but we let it float and clamp later if needed, 
    # or better, fix r0=0 for the center point to save a dimension).
    # Let's fix the center point at (0,0) to reduce dimensionality and ensure validity.
    
    points = []
    
    # Ring 0: Center
    points.append([0.0, 0.0])
    
    for idx in range(1, n_rings):
        count = ring_counts[idx]
        r = radii[idx]
        # Clamp radius to be within valid range
        r = np.clip(r, 0.0, 1.0)
        offset = offsets[idx]
        
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        
        for i in range(count):
            points.append([xs[i], ys[i]])
            
    pts = np.array(points)
    
    # Ensure points are inside the unit circle (with a small margin for safety in DE)
    norms = np.hypot(pts[:, 0], pts[:, 1])
    mask = norms > 1.0
    if np.any(mask):
        pts[mask] = pts[mask] / norms[mask, np.newaxis] * 0.999
        
    return pts

def _soft_min_obj_grad(points_flat, beta):
    """Objective and gradient for L-BFGS-B using soft-min via log-sum-exp."""
    points = points_flat.reshape(N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary_dists = 1.0 - norms
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Clip negative distances to 0 to avoid log issues if points overlap heavily
    all_dists = np.maximum(all_dists, 1e-12)
    
    scaled = -beta * all_dists
    max_val = np.max(scaled)
    exp_scaled = np.exp(scaled - max_val)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    grad = np.zeros_like(points_flat)
    
    # Pair distance gradients
    # weights[i] corresponds to the i-th pair in iu
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        w = weights[idx]
        d = pair_dists[idx]
        if d > 1e-15:
            direction = (points[i] - points[j]) / d
            grad[2*i] += w * direction[0]
            grad[2*i+1] += w * direction[1]
            grad[2*j] -= w * direction[0]
            grad[2*j+1] -= w * direction[1]
    
    # Boundary distance gradients
    for idx in range(N):
        w = weights[N + idx]
        norm = norms[idx]
        if norm > 1e-15:
            grad[2*idx] -= w * points[idx, 0] / norm
            grad[2*idx+1] -= w * points[idx, 1] / norm
    
    soft_min_val = -(max_val + np.log(sum_exp)) / beta
    obj = -soft_min_val
    return obj, grad

def _local_repulsion_polish(points, max_iters=200, tol=1e-12):
    """Exact local search: push apart overlapping pairs and pull inward if outside."""
    points = points.copy()
    n = len(points)
    
    for iteration in range(max_iters):
        if time.time() > _deadline:
            break
        improved = False
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        norms = np.hypot(points[:, 0], points[:, 1])
        
        iu = np.triu_indices(n, k=1)
        min_pair = np.min(dists[iu]) / 2.0
        min_boundary = 1.0 - np.max(norms)
        current_r = min(min_pair, min_boundary)
        
        # For each pair that is too close, push them apart
        for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            d = dists[i, j]
            if d < 2 * current_r - 1e-10:
                if d > 1e-15:
                    direction = (points[i] - points[j]) / d
                    overlap = (2 * current_r - d) * 0.5
                    move = overlap * 0.1
                    points[i] += direction * move
                    points[j] -= direction * move
                    improved = True
        
        # For each point too close to boundary, pull inward
        for i in range(n):
            norm_i = norms[i]
            if norm_i > 1.0 - current_r + 1e-10:
                if norm_i > 1e-15:
                    direction = points[i] / norm_i
                    overlap = (norm_i - (1.0 - current_r)) * 0.5
                    move = overlap * 0.1
                    points[i] -= direction * move
                    improved = True
        
        new_score = _compute_score(points)
        if new_score <= current_r - tol:
            break
        if not improved:
            break
    
    return points

def construct():
    global _deadline
    t_start = time.time()
    _deadline = t_start + 140.0  # Safety margin before 150s kill
    
    # Known optimal structure: 1 center + rings of 6, 12, 18, 13
    ring_counts = [1, 6, 12, 18, 13]
    
    # Parameters to optimize: radii for rings 1-4 (4 params) and offsets for rings 1-4 (4 params)
    # Total 8 parameters for DE
    n_params = 4 + 4
    
    # Bounds for radii (0.0 to 1.0) and offsets (0 to 2*pi)
    bounds = []
    for i in range(4):
        bounds.append((0.05, 0.95)) # Radii
    for i in range(4):
        bounds.append((0.0, 2 * np.pi)) # Offsets
    
    # Initial best guess
    # Approximate radii for optimal packing: 0.0, 0.25, 0.5, 0.75, 0.95
    init_params = np.array([0.25, 0.5, 0.75, 0.95, 0.0, 0.0, 0.0, 0.0])
    
    def de_objective(params):
        pts = _ring_to_points(params, ring_counts)
        return -_compute_score(pts)
    
    best_points = None
    best_score = 0.0
    
    # 1. Differential Evolution Global Search
    # We use a relatively small population and maxiter to stay within time limits
    # DE is stochastic, so we run it once with a good seed.
    try:
        if time.time() < _deadline - 30.0:
            de_result = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                polish=False  # We will polish ourselves
            )
            
            if de_result.success or de_result.fun < 1e10:
                best_params = de_result.x
                best_points = _ring_to_points(best_params, ring_counts)
                best_score = _compute_score(best_points)
    except Exception:
        pass
    
    # Fallback if DE fails or hasn't run
    if best_points is None:
        best_points = _ring_to_points(init_params, ring_counts)
        best_score = _compute_score(best_points)
    
    # 2. Local Optimization with L-BFGS-B (Soft-Min)
    # We run multiple stages with increasing beta to refine the solution.
    x0 = best_points.flatten().copy()
    
    betas = [10.0, 30.0, 100.0, 300.0, 1000.0]
    
    for beta in betas:
        if time.time() > _deadline - 10.0:
            break
            
        try:
            result = minimize(
                _soft_min_obj_grad,
                x0,
                method='L-BFGS-B',
                args=(beta,),
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 30}
            )
            
            if result.success or result.fun < 1e10:
                opt_points = result.x.reshape(N, 2)
                
                # Verify validity
                if np.all(np.isfinite(opt_points)):
                    norms = np.hypot(opt_points[:, 0], opt_points[:, 1])
                    if np.all(norms <= 1.0):
                        score = _compute_score(opt_points)
                        if score > best_score:
                            best_score = score
                            best_points = opt_points.copy()
                            x0 = opt_points.flatten().copy()
        except Exception:
            pass
    
    # 3. Final Polish with Exact Local Repulsion
    if time.time() < _deadline - 2.0:
        polished = _local_repulsion_polish(best_points, max_iters=500, tol=1e-14)
        polished_score = _compute_score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    
    return best_points
# EVOLVE-BLOCK-END
