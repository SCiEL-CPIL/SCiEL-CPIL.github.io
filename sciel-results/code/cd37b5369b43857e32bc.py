"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_init(n, seed=0):
    """Generate n points from a hexagonal lattice inside the unit disk."""
    rng = np.random.RandomState(seed)
    # Try different spacings to find a good initial grid
    best_pts = None
    best_score = -1
    for attempt in range(5):
        # Spacing roughly 2*sqrt(3)/n^(1/2) for hex packing density
        spacing = 0.05 + rng.random() * 0.01
        k = int(2.0 / spacing) + 3
        xs, ys = np.meshgrid(
            np.arange(-k, k + 1) * spacing,
            np.arange(-k, k + 1) * spacing * np.sqrt(3.0) / 2.0
        )
        # Offset odd rows for hexagonal packing
        for j in range(2 * k + 1):
            if j % 2 == 1:
                xs[j, :] += spacing / 2.0
        
        pts = np.column_stack([xs.ravel(), ys.ravel()])
        r = np.hypot(pts[:, 0], pts[:, 1])
        # Ensure points are strictly inside with some margin
        mask = r <= 1.0 - spacing / 2.0 - 0.001
        pts = pts[mask]
        
        if len(pts) >= n:
            # Select n points. To maximize initial radius, we might want to pick
            # a subset that is well-distributed. Picking from center out is a start.
            # A better heuristic: pick points that maximize min distance locally? 
            # For init, just taking the first n or random n is okay if optimizer is good.
            # Let's take points sorted by radius to keep them compact initially.
            idx = np.argsort(r[mask])[:n]
            candidate_pts = pts[idx].copy()
            
            # Quick score check
            diff = candidate_pts[:, np.newaxis, :] - candidate_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            boundary = 1.0 - np.hypot(candidate_pts[:, 0], candidate_pts[:, 1])
            min_boundary = np.min(boundary)
            score = min(min_pair / 2.0, min_boundary)
            
            if score > best_score:
                best_score = score
                best_pts = candidate_pts
    
    if best_pts is None:
        # Fallback random
        angles = rng.uniform(0, 2 * np.pi, n)
        radii = rng.uniform(0, 0.8, n)
        best_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
    return best_pts


def _compute_radius(pts):
    """Compute the maximum radius for given points."""
    n = len(pts)
    if n < 2:
        return min(1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1])), 0.5)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    boundary_dists = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _force_optimize(pts, max_iters=2000, step_size=0.01, time_limit=None):
    """
    Optimize points using a force-directed method.
    Forces:
    1. Repulsion between circles: Proportional to overlap.
    2. Repulsion from boundary: Proportional to how much it exceeds boundary.
    3. Weak attraction to center: To prevent drift.
    """
    n = len(pts)
    pts = pts.copy()
    current_r = _compute_radius(pts)
    best_pts = pts.copy()
    best_r = current_r
    
    # Adaptive step size
    step = step_size
    decay = 0.999
    min_step = 1e-5
    
    # We want to maximize the minimum distance.
    # We can frame this as minimizing the negative of the soft-min of distances.
    # Or simply applying forces that push apart when close.
    
    # Let's use a "potential energy" approach.
    # E = sum(max(0, 2*r_target - d_ij)^2) + sum(max(0, r_target - (1-|c_i|))^2)
    # But r_target is unknown. 
    # Instead, we can just push apart any pair that is closer than the current best radius * 2.
    # This is a bit tricky.
    
    # Alternative: Gradient ascent on the objective function f(pts) = min(min_pair/2, min_boundary).
    # The gradient of the min function is sparse.
    # Let's use a simple iterative relaxation:
    # For each point, calculate the force from all other points and boundary.
    # Force from point j on i: if d_ij < 2*current_r, push i away from j.
    # Force from boundary: if 1-|c_i| < current_r, push i towards center.
    
    # This approach relies on current_r increasing.
    
    for it in range(max_iters):
        if time_limit and time.time() > time_limit:
            break
            
        # Compute distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        boundary_dists = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
        
        # Determine active constraints
        # We want to increase the radius. 
        # Let's define a target radius slightly above current.
        # Actually, a more stable method is to maximize the sum of squared distances to the closest neighbor?
        # No, let's stick to the force model based on the *current* minimum radius.
        
        min_pair_dist = np.min(dists)
        min_boundary = np.min(boundary_dists)
        current_r = min(min_pair_dist / 2.0, min_boundary)
        
        # Update best
        if current_r > best_r:
            best_r = current_r
            best_pts = pts.copy()
        
        # Calculate forces
        forces = np.zeros_like(pts)
        
        # 1. Pairwise repulsion
        # Only consider pairs that are "close" (distance < 2 * current_r * 1.1)
        # This makes it sparse if we were using a grid, but here we have full matrix.
        # We can mask it.
        mask = dists < 2.0 * current_r * 1.2
        np.fill_diagonal(mask, False)
        
        # Force magnitude: k * (2*r_target - d)
        # Let's use r_target = current_r
        # If d < 2*current_r, push apart.
        # We want to increase the *minimum* distance.
        # So we should push apart the pairs that define the minimum.
        # A simple heuristic: push apart all pairs with d < 2*current_r.
        
        # Vectorized force calculation
        # F_ij = (d_ij_vec / d_ij) * (2*current_r - d_ij) for d_ij < 2*current_r
        # Note: dists is symmetric.
        
        # Create a matrix of forces
        # We need the unit vectors
        # diff is pts[i] - pts[j]
        # If we want to push i away from j, force on i is in direction (pts[i] - pts[j])
        # So F_i += (pts[i] - pts[j]) / d_ij * (2*r - d_ij)
        
        # To avoid division by zero and handle masking:
        inv_dists = np.zeros_like(dists)
        np.divide(1.0, dists, out=inv_dists, where=dists > 1e-8)
        
        # Magnitude of force
        # Only for masked pairs
        mag = np.zeros_like(dists)
        active = mask & (dists < 2.0 * current_r)
        mag[active] = 2.0 * current_r - dists[active]
        
        # Force tensor: F_ij = mag_ij * (diff_ij / d_ij)
        # diff is (i,j,k). We want to sum over j for each i.
        # force_on_i = sum_j (mag_ij * diff_ij / d_ij)
        # diff_ij = pts[i] - pts[j]
        # So force_on_i = sum_j mag_ij * (pts[i] - pts[j]) * inv_dists_ij
        
        # This is equivalent to:
        # sum_j mag_ij * pts[i] - sum_j mag_ij * pts[j]
        # = pts[i] * sum_j mag_ij - sum_j (mag_ij * pts[j])
        
        sum_mag = np.sum(mag, axis=1)
        # weighted sum of pts[j]
        # mag is (n, n). pts is (n, 2).
        # weighted_pts = mag.T @ pts  -> shape (n, 2)
        # Row i of weighted_pts is sum_j mag[i,j] * pts[j]
        weighted_pts = mag.T @ pts
        
        forces_pair = pts * sum_mag[:, np.newaxis] - weighted_pts
        
        # 2. Boundary repulsion
        # If 1 - |c_i| < current_r, push towards center.
        # Direction is -pts[i] / |pts[i]|
        # Magnitude is current_r - (1 - |c_i|)
        
        boundary_violation = np.maximum(0, current_r - boundary_dists)
        # Normalize pts
        norms = np.hypot(pts[:, 0], pts[:, 1])
        inv_norms = np.zeros_like(norms)
        np.divide(1.0, norms, out=inv_norms, where=norms > 1e-8)
        
        # Force direction is towards center: -pts / norm
        # Force magnitude is violation
        forces_boundary = -pts * (boundary_violation * inv_norms)[:, np.newaxis]
        
        # 3. Center attraction (weak) to prevent drift
        forces_center = -0.01 * pts
        
        total_forces = forces_pair + forces_boundary + forces_center
        
        # Update positions
        pts += step * total_forces
        
        # Keep inside unit circle (hard constraint)
        new_norms = np.hypot(pts[:, 0], pts[:, 1])
        mask_out = new_norms > 1.0
        if np.any(mask_out):
            pts[mask_out] /= new_norms[mask_out, np.newaxis]
            # Push in a bit
            pts[mask_out] *= 0.99
        
        # Decay step size
        step *= decay
        if step < min_step:
            step = min_step
            
    return best_pts, best_r


def construct():
    n = N
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Get initial hexagonal packing
    x0 = _hex_init(n, seed=42)
    best_pts = x0.copy()
    best_r = _compute_radius(best_pts)
    
    # Run force optimization
    # We can run multiple restarts if time permits
    n_restarts = 3
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        if restart == 0:
            x_init = x0.copy()
        else:
            # Perturb previous best or use new init
            x_init = _hex_init(n, seed=restart * 100 + 7)
            
        # Run force optimization with time limit
        t_limit = time.time() + (deadline - time.time()) / (n_restarts - restart) - 5
        pts_opt, r_opt = _force_optimize(x_init, max_iters=5000, step_size=0.005, time_limit=t_limit)
        
        if r_opt > best_r:
            best_r = r_opt
            best_pts = pts_opt.copy()
            
        # Polish step: use scipy minimize on the negative of the min radius?
        # Or just a few more iterations of force optimization with smaller step?
        # Let's try a local polish using Nelder-Mead on a small subset? 
        # No, N=300 is too large for Nelder-Mead directly.
        # Let's just do a final force pass with very small step.
        
        if time.time() < deadline - 5:
            # Polish
            t_polish = time.time() + 5
            pts_polished, r_polished = _force_optimize(best_pts, max_iters=1000, step_size=0.001, time_limit=t_polish)
            if r_polished > best_r:
                best_r = r_polished
                best_pts = pts_polished.copy()

    return best_pts
# EVOLVE-BLOCK-END
