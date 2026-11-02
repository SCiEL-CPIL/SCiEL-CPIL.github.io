# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 300

def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed."""
    # Estimate radius to fit n circles
    # Area of unit circle = pi. Area of n circles = n * pi * r^2.
    # pi = n * pi * r^2 * packing_factor (~0.9)
    # r ~ 1 / sqrt(n * 0.9)
    r_est = 1.0 / np.sqrt(n * 0.9)
    spacing = 2.0 * r_est * 1.02
    
    points = []
    k_max = int(1.0 / spacing) + 5
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    
    margin = spacing * 0.5
    mask = radii <= 1.0 - margin
    points = points[mask]
    radii = radii[mask]
    
    if len(points) < n:
        # Shrink spacing if not enough points
        for _ in range(20):
            spacing *= 0.95
            points = []
            k_max = int(1.0 / spacing) + 5
            for j in range(-k_max, k_max + 1):
                for i in range(-k_max, k_max + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    points.append([x, y])
            points = np.array(points)
            radii = np.hypot(points[:, 0], points[:, 1])
            mask = radii <= 1.0 - margin
            points = points[mask]
            radii = radii[mask]
            if len(points) >= n:
                break
    
    idx = np.argsort(radii)
    return points[idx[:n]]

def _compute_score(centers):
    """Compute the packing radius for given centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    radii = np.linalg.norm(centers, axis=1)
    min_boundary = np.min(1.0 - radii)
    return min(min_pair_dist / 2.0, min_boundary)

def _overlap_energy_and_grad(x, r_target):
    """
    Compute overlap energy and gradient.
    x: flattened array of centers (2*N)
    r_target: target radius
    Returns: energy, gradient
    """
    n = len(x) // 2
    centers = x.reshape(n, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair_dists = dists[mask]
    
    overlap_pairs = np.maximum(0, 2 * r_target - pair_dists)
    energy_pairs = np.sum(overlap_pairs ** 2)
    
    radii = np.linalg.norm(centers, axis=1)
    overlap_bounds = np.maximum(0, r_target + radii - 1.0)
    energy_bounds = np.sum(overlap_bounds ** 2)
    
    total_energy = energy_pairs + energy_bounds
    
    grad = np.zeros_like(x)
    
    # Gradient for pair overlaps
    w_pairs = 2.0 * overlap_pairs
    W_mat = np.zeros((n, n))
    W_mat[mask] = w_pairs
    W_mat = W_mat + W_mat.T
    
    # grad of sum max(0, 2r - d)^2 w.r.t c_i
    # = sum_j 2(2r - d_ij) * (c_i - c_j) / d_ij
    normalized_diff = diff / (dists[:, :, np.newaxis] + 1e-12)
    grad_pair = np.sum(W_mat[:, :, np.newaxis] * normalized_diff, axis=1)
    
    # Gradient for boundary overlaps
    # grad of sum max(0, r + |c_i| - 1)^2 w.r.t c_i
    # = 2(r + |c_i| - 1) * c_i / |c_i|
    safe_radii = np.where(radii < 1e-12, 1e-12, radii)
    w_bounds = 2.0 * overlap_bounds
    grad_bound = w_bounds[:, np.newaxis] * centers / safe_radii[:, np.newaxis]
    
    grad = (grad_pair + grad_bound).flatten()
    
    return total_energy, grad

def _relax_packing(centers, r_target, maxiter=1000, ftol=1e-12):
    """
    Relax the packing using L-BFGS to minimize overlap energy for a given r.
    """
    x0 = centers.flatten()
    
    def objective(x):
        e, _ = _overlap_energy_and_grad(x, r_target)
        return e
        
    def gradient(x):
        _, g = _overlap_energy_and_grad(x, r_target)
        return g
        
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-8}
    )
    
    return result.x.reshape(-1, 2)

def _optimize_boundary(centers, r_target, maxiter=200):
    """
    Optimize the outer ring of circles to maximize the minimum distance to the wall.
    We treat the outer ring as particles connected to the rest via constraints.
    Here we simply re-optimize the whole system but with a bias towards pushing outer points out?
    Actually, a simpler approach: Identify outer points. Optimize their polar coords (theta, rho)
    to maximize min(1-rho) subject to non-overlap.
    """
    n = len(centers)
    radii = np.linalg.norm(centers, axis=1)
    
    # Identify boundary circles: those with radius > 0.8 * max_radius
    # Or simply the outermost 10%?
    # Let's use a threshold: if radius > 0.75, it's a boundary candidate.
    boundary_mask = radii > 0.75
    boundary_idx = np.where(boundary_mask)[0]
    interior_idx = np.where(~boundary_mask)[0]
    
    if len(boundary_idx) == 0:
        return centers
    
    x0 = centers[boundary_idx].flatten()
    interior_centers = centers[interior_idx]
    
    def boundary_obj(x):
        cb = x.reshape(-1, 2)
        
        # 1. Overlap with interior
        diff_bi = cb[:, np.newaxis, :] - interior_centers[np.newaxis, :, :]
        dists_bi = np.sqrt(np.sum(diff_bi ** 2, axis=-1) + 1e-12)
        overlap_bi = np.maximum(0, 2 * r_target - dists_bi)
        e_bi = np.sum(overlap_bi ** 2)
        
        # 2. Overlap with other boundary
        n_b = len(cb)
        if n_b > 1:
            diff_bb = cb[:, np.newaxis, :] - cb[np.newaxis, :, :]
            dists_bb = np.sqrt(np.sum(diff_bb ** 2, axis=-1) + 1e-12)
            mask_bb = np.triu(np.ones((n_b, n_b), dtype=bool), k=1)
            overlap_bb = np.maximum(0, 2 * r_target - dists_bb[mask_bb])
            e_bb = np.sum(overlap_bb ** 2)
        else:
            e_bb = 0.0
            
        # 3. Wall term: We want to maximize distance to wall, i.e., minimize |c_b|?
        # No, we want to push them OUT to the wall to maximize the "clearance" of the *inner* circles?
        # The score is min(min_pair_dist/2, min(1-|c_i|)).
        # If we push outer circles to the wall, their (1-|c_i|) becomes small.
        # This reduces the score if the pair distances are large enough.
        # However, pushing them out allows the inner circles to spread more, potentially increasing pair distances.
        # So we should push them out as much as possible, limited by non-overlap.
        # Objective: Minimize sum of overlaps.
        # To ensure they go to the wall, we can add a term that penalizes being INSIDE the wall?
        # No, the wall term in _overlap_energy_and_grad already penalizes going OUTSIDE.
        # It does NOT penalize being inside.
        # So L-BFGS will just minimize overlaps. It might leave them inside if that minimizes overlaps.
        # We need to encourage them to be at the wall.
        # Add a term: sum ( (1 - r_target) - |c_b| )^2 for those inside?
        # This pushes them to |c_b| = 1 - r_target.
        
        radii_b = np.linalg.norm(cb, axis=1)
        target_radius = 1.0 - r_target
        dist_to_target = target_radius - radii_b
        # Only penalize if they are inside the target radius?
        # If they are outside, the overlap energy with the wall (in the global function) would handle it,
        # but here we are only optimizing boundary points.
        # We should constrain them to be <= 1 - r_target.
        # Let's add a penalty for being less than target_radius? No, that pushes them out.
        # We want to maximize the minimum clearance.
        # If we just minimize overlaps, they will find a stable configuration.
        # To ensure they are "tight" against the wall, we can minimize the sum of their radii? No.
        # Let's just minimize overlaps. The global relaxation should have pushed them out somewhat.
        # This step is to fine-tune.
        
        e_wall = 0.0 # No wall penalty here, just overlaps
        
        total_e = e_bi + e_bb + e_wall
        
        # Gradient
        grad_x = np.zeros_like(x)
        
        # Grad e_bi
        w_bi = 2.0 * overlap_bi
        norm_diff_bi = diff_bi / (dists_bi[:, :, np.newaxis] + 1e-12)
        grad_bi = np.sum(w_bi[:, :, np.newaxis] * norm_diff_bi, axis=1)
        
        # Grad e_bb
        if n_b > 1:
            w_bb = 2.0 * overlap_bb
            W_bb = np.zeros((n_b, n_b))
            W_bb[mask_bb] = w_bb
            W_bb = W_bb + W_bb.T
            norm_diff_bb = diff_bb / (dists_bb[:, :, np.newaxis] + 1e-12)
            grad_bb = np.sum(W_bb[:, :, np.newaxis] * norm_diff_bb, axis=1)
        else:
            grad_bb = np.zeros_like(cb)
            
        grad_x = (grad_bi + grad_bb).flatten()
        
        return total_e, grad_x
        
    result = minimize(
        boundary_obj,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    new_boundary = result.x.reshape(-1, 2)
    new_centers = centers.copy()
    new_centers[boundary_idx] = new_boundary
    
    # Ensure no one went outside
    radii_new = np.linalg.norm(new_centers, axis=1)
    mask_out = radii_new > 1.0 - r_target
    if np.any(mask_out):
        new_centers[mask_out] *= ((1.0 - r_target) / radii_new[mask_out])[:, np.newaxis]
        
    return new_centers

def construct():
    """
    Construct 300 equal circles in a unit circle, maximizing the common radius.
    """
    centers = _hexagonal_seed(N)
    
    # Phase 1: Ramping Relaxation
    # Start with a small radius and increase it.
    r_start = 0.04
    r_end = 0.05315
    n_steps = 10
    
    current_x = centers.copy()
    
    for i in range(n_steps):
        r_target = r_start + (r_end - r_start) * (i / n_steps)
        # Relax with current r_target
        current_x = _relax_packing(current_x, r_target, maxiter=200)
        
    # Phase 2: Final Polish at r_end
    current_x = _relax_packing(current_x, r_end, maxiter=500)
    
    # Phase 3: Boundary Optimization
    current_x = _optimize_boundary(current_x, r_end, maxiter=200)
    
    # Phase 4: Final Full Relaxation to ensure no overlaps introduced by boundary opt
    current_x = _relax_packing(current_x, r_end, maxiter=100)
    
    final_score = _compute_score(current_x)
    initial_score = _compute_score(centers)
    
    if final_score < initial_score:
        current_x = centers
        
    # Final safety check
    for i in range(len(current_x)):
        r = np.linalg.norm(current_x[i])
        if r > 1.0 - 1e-10:
            scale = (1.0 - 1e-10) / r
            current_x[i] *= scale
            
    return current_x
# EVOLVE-BLOCK-END
