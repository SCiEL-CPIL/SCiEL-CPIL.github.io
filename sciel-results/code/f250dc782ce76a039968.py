"""
Packing 50 equal circles in a unit disk using gradient-based optimization
with a soft-max objective function to handle the min-distance constraint.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = None
DEADLINE = 140.0  # 10s safety margin before 150s kill

def _score(points):
    """Compute packing radius: min(min pairwise dist/2, min (1-|c_i|))."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = np.linalg.norm(points, axis=1)
    if np.any(dists >= 1.0):
        return 0.0
    boundary_margin = np.min(1.0 - dists)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = np.min(dist_mat)
    return min(min_pair_dist / 2.0, boundary_margin)

def _hex_lattice_init(r_guess):
    """Hexagonal lattice initialization."""
    spacing = 2.0 * r_guess
    pts = []
    # Estimate grid size
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - r_guess) ** 2:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= N:
        # Select N points closest to center or just first N
        # Sorting by distance to center helps with symmetry
        radii = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(radii, kind='stable')[:N]
        return pts[idx]
    else:
        # Fallback to ring init if hex lattice is too sparse
        return _ring_init()

def _ring_init():
    """Concentric ring initialization."""
    points = np.zeros((N, 2))
    points[0] = [0, 0]
    remaining = N - 1
    ring_idx = 1
    idx = 1
    while remaining > 0:
        n_ring = 6 * ring_idx
        n_ring = min(n_ring, remaining)
        if n_ring == 0:
            break
        angles = np.linspace(0, 2*np.pi, n_ring, endpoint=False)
        r_ring = ring_idx * 0.18
        points[idx:idx+n_ring, 0] = r_ring * np.cos(angles)
        points[idx:idx+n_ring, 1] = r_ring * np.sin(angles)
        idx += n_ring
        remaining -= n_ring
        ring_idx += 1
    return points

def _soft_max_objective(pts, r_target, beta):
    """
    Objective function to minimize.
    We want to maximize the minimum distance, which is equivalent to minimizing
    the maximum violation of the constraints:
    1. Pairwise distance >= 2*r_target
    2. Boundary distance <= 1 - r_target
    
    Violation v_i = constraint_value - actual_value
    For pairs: v_ij = 2*r_target - dist_ij (positive if too close)
    For boundary: v_i = norm_i - (1 - r_target) (positive if too far)
    
    We use a soft-max to approximate the maximum violation.
    """
    n = len(pts)
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    cont_viol = norms - (1.0 - r_target)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Only consider upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    pair_viol = 2.0 * r_target - dists[iu]
    
    all_viol = np.concatenate([cont_viol, pair_viol])
    
    # Soft-max
    max_v = np.max(all_viol)
    if max_v > 50.0 / beta:
        # Avoid overflow
        return max_v, np.zeros_like(pts)
        
    exp_v = np.exp(beta * (all_viol - max_v))
    sum_exp = np.sum(exp_v)
    soft_max_val = max_v + np.log(sum_exp) / beta
    weights = exp_v / sum_exp
    
    # Gradient
    grad = np.zeros_like(pts)
    
    # Container gradient
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    unit_vecs = pts / safe_norms[:, np.newaxis]
    grad += weights[:n, np.newaxis] * unit_vecs
    
    # Pairwise gradient
    iw = weights[n:]
    valid = dists[iu] > 1e-10
    if np.any(valid):
        iw_valid = iw[valid]
        iu_valid = (iu[0][valid], iu[1][valid])
        d_valid = dists[iu_valid]
        diff_valid = pts[iu_valid[0]] - pts[iu_valid[1]]
        unit_diff = diff_valid / d_valid[:, np.newaxis]
        
        np.add.at(grad, iu_valid[0], -iw_valid[:, np.newaxis] * unit_diff)
        np.add.at(grad, iu_valid[1], iw_valid[:, np.newaxis] * unit_diff)
    
    return soft_max_val, grad

def _optimize_from_start(pts_init, r_target, start_beta=50.0, end_beta=1000.0, max_iter=500):
    """
    Run L-BFGS-B optimization with adaptive beta.
    """
    current_pts = pts_init.copy()
    best_pts = current_pts.copy()
    best_score = _score(current_pts)
    
    if best_score <= 0:
        return current_pts, best_score
        
    # We will perform a sequence of optimizations with increasing beta
    # to sharpen the objective function.
    n_stages = 5
    betas = np.linspace(start_beta, end_beta, n_stages)
    
    for stage, beta in enumerate(betas):
        if time.time() - START_TIME > DEADLINE - 5:
            break
            
        # Define objective and gradient for this stage
        def obj_grad(x):
            val, grad = _soft_max_objective(x, r_target, beta)
            return val, grad
        
        # Run L-BFGS-B
        res = minimize(obj_grad, current_pts, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < _soft_max_objective(current_pts, r_target, beta)[0]:
            current_pts = res.x
            score = _score(current_pts)
            if score > best_score:
                best_score = score
                best_pts = current_pts.copy()
                
    return best_pts, best_score

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initial guess for radius
    # Area argument: 50 * pi * r^2 <= pi * 1^2 => r <= 1/sqrt(50) ~ 0.141
    # Known best is ~0.1258
    r_guess = 0.125
    
    # Generate multiple initializations
    inits = []
    
    # 1. Hex lattice
    inits.append(_hex_lattice_init(r_guess))
    
    # 2. Ring init
    inits.append(_ring_init())
    
    # 3. Random perturbations of hex
    rng = np.random.RandomState(42)
    for _ in range(3):
        pts = _hex_lattice_init(r_guess).copy()
        pts += rng.randn(N, 2) * 0.02
        # Ensure inside
        dists = np.hypot(pts[:, 0], pts[:, 1])
        mask = dists > 1.0 - r_guess
        if np.any(mask):
            pts[mask] *= (1.0 - r_guess) / dists[mask]
        inits.append(pts)
        
    best_pts = None
    best_r = 0.0
    
    for init in inits:
        if time.time() - START_TIME > DEADLINE - 10:
            break
            
        # Try to optimize with a slightly larger target radius to push it
        # If it fails, it will settle at the max possible radius for that init
        # We start with r_target slightly above the current best estimate
        # Actually, it's better to start with a conservative r_target and increase
        # or just let the optimizer find the maximum.
        # The soft-max objective minimizes the violation. If we set r_target too high,
        # the violation is large everywhere. 
        # Strategy: Start with r_target = 0.1, optimize. Get score. 
        # Then set r_target = score + epsilon, optimize again.
        
        current_r = 0.1
        pts = init.copy()
        
        # Iterative refinement of r_target
        for _ in range(10):
            if time.time() - START_TIME > DEADLINE - 5:
                break
                
            # Optimize for current r_target
            pts, score = _optimize_from_start(pts, current_r, start_beta=50, end_beta=500, max_iter=200)
            
            if score <= current_r:
                # We can't increase radius further with this config
                break
            else:
                # We achieved score > current_r, so we can try a higher target
                # But we need to be careful not to overshoot too much
                # The score is the actual max radius achievable.
                # If score > current_r, it means we satisfied current_r with room to spare.
                # We should set current_r to score to tighten the constraint.
                if score > best_r:
                    best_r = score
                    best_pts = pts.copy()
                
                # Increase target to push for more
                current_r = score
                # Add a small margin to encourage further improvement
                current_r += 0.001
                
                if time.time() - START_TIME > DEADLINE - 5:
                    break
                    
    # Final polish: Use the best found configuration and do a final high-precision optimization
    if best_pts is not None and time.time() - START_TIME < DEADLINE - 5:
        # Set r_target to the current best radius
        final_r = best_r
        # Do a final optimization with high beta to sharpen
        best_pts, final_score = _optimize_from_start(best_pts, final_r, start_beta=100, end_beta=2000, max_iter=500)
        if final_score > best_r:
            best_r = final_score
            best_pts = best_pts.copy()
            
    # Ensure valid output
    if best_pts is None:
        best_pts = _hex_lattice_init(r_guess)
        
    # Final validation and clipping
    dists = np.hypot(best_pts[:, 0], best_pts[:, 1])
    for i in range(N):
        if dists[i] >= 1.0:
            if dists[i] > 1e-10:
                best_pts[i] *= (1.0 - 1e-6) / dists[i]
            else:
                best_pts[i] = [0, 0]
                
    return best_pts

# EVOLVE-BLOCK-END
