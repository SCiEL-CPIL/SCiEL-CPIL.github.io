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

def _md_relax(centers, r_target, steps=500, dt=0.001):
    """
    Molecular Dynamics style relaxation using repulsive forces.
    This helps escape local minima better than pure gradient descent.
    """
    n = len(centers)
    pos = centers.copy()
    
    # We want to enforce d_ij >= 2*r_target and |c_i| <= 1 - r_target
    # Force = -grad(energy)
    # Energy = sum max(0, 2r - d)^2 + sum max(0, r + |c| - 1)^2
    
    for step in range(steps):
        # Calculate forces
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion
        # Vectorized computation of distances and forces is O(N^2)
        # For N=300, 90k pairs, this is fast in numpy.
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
        
        # Mask for i < j to avoid double counting and self-interaction
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        
        # Overlap amount
        overlap = np.maximum(0, 2 * r_target - dists)
        
        # Force magnitude for each pair
        # F = 2 * overlap * (d_vec / d^2) ... wait, grad of (2r-d)^2 is -2(2r-d) * grad d
        # grad d_ij w.r.t c_i is (c_i - c_j) / d_ij
        # So force on i from j is 2 * (2r - d) * (c_i - c_j) / d
        # We only apply if overlap > 0
        
        # To be efficient, we compute the force matrix directly
        # W_ij = 2 * overlap_ij / dists_ij
        # F_i = sum_j W_ij * (c_i - c_j)
        
        W_mat = np.zeros((n, n))
        W_mat[mask] = 2.0 * overlap[mask] / (dists[mask] + 1e-12)
        W_mat = W_mat + W_mat.T
        
        # F_i = sum_j W_ij * (c_i - c_j)
        # = sum_j W_ij * c_i - sum_j W_ij * c_j
        # = c_i * sum_j W_ij - sum_j W_ij * c_j
        
        sum_W = np.sum(W_mat, axis=1)
        forces = pos * sum_W[:, np.newaxis] - (W_mat @ pos)
        
        # Boundary forces
        radii = np.linalg.norm(pos, axis=1)
        bound_overlap = np.maximum(0, r_target + radii - 1.0)
        # Force is inward: -bound_overlap * c_i / |c_i|
        safe_radii = np.where(radii < 1e-12, 1e-12, radii)
        forces -= (bound_overlap / safe_radii)[:, np.newaxis] * pos
        
        # Update positions with damping
        # Simple Euler integration with damping
        pos += forces * dt
        pos *= 0.95 # Damping
        
        # Project back inside container to avoid huge excursions
        radii = np.linalg.norm(pos, axis=1)
        max_r = 1.0 - r_target + 0.1 # Allow some slack
        mask_out = radii > max_r
        if np.any(mask_out):
            pos[mask_out] *= (max_r / radii[mask_out])[:, np.newaxis]
            
    return pos

def _overlap_energy(x, r_target):
    """
    Compute overlap energy and gradient for L-BFGS.
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
    
    w_pairs = 2.0 * overlap_pairs
    W_mat = np.zeros((n, n))
    W_mat[mask] = w_pairs
    W_mat = W_mat + W_mat.T
    
    normalized_diff = diff / (dists[:, :, np.newaxis] + 1e-12)
    grad_pair = W_mat[:, :, np.newaxis] * normalized_diff
    grad_pair = np.sum(grad_pair, axis=1)
    
    safe_radii = np.where(radii < 1e-12, 1e-12, radii)
    w_bounds = 2.0 * overlap_bounds
    grad_bound = w_bounds[:, np.newaxis] * centers / safe_radii[:, np.newaxis]
    
    grad = (grad_pair + grad_bound).flatten()
    
    return total_energy, grad

def _optimize_boundary(centers, r_target):
    """
    Optimize only the boundary circles to maximize clearance to the wall,
    while maintaining non-overlap with interior circles.
    """
    n = len(centers)
    radii = np.linalg.norm(centers, axis=1)
    
    # Identify boundary circles: those with radius > 0.8
    # Or simply the outermost ring.
    # Let's say if dist > 0.75, it's a boundary candidate.
    boundary_mask = radii > 0.75
    boundary_idx = np.where(boundary_mask)[0]
    interior_idx = np.where(~boundary_mask)[0]
    
    if len(boundary_idx) == 0:
        return centers
    
    # We optimize the positions of boundary circles.
    # Objective: Maximize min(1 - |c_b|) subject to |c_b - c_i| >= 2r for all b, i.
    # Equivalently, Minimize sum max(0, 2r - |c_b - c_i|)^2 + lambda * sum (|c_b| - (1-r))^2 ?
    # No, we want to push them OUT to the wall.
    # So we want to Minimize sum max(0, (1-r) - |c_b|)^2 ? No, that pushes them in if they are too far?
    # If |c_b| < 1-r, they are safe. We want to maximize |c_b|.
    # So Minimize - sum |c_b| ? No, that's not differentiable at 0.
    # Let's use: Minimize sum ( (1-r) - |c_b| )^2 for those inside?
    # Actually, the main constraint is non-overlap.
    # If we just push them out, they might overlap.
    
    # Let's define an objective for boundary circles x_b:
    # 1. Penalty for overlapping with interior: sum_{b,i} max(0, 2r - |c_b - c_i|)^2
    # 2. Penalty for being too close to wall? No, we want them AT the wall.
    # 3. Penalty for overlapping with other boundary circles: sum_{b1,b2} max(0, 2r - |c_b1 - c_b2|)^2
    # 4. Reward for being far from center? 
    #    Let's minimize: E = E_overlap + mu * sum ( (1-r) - |c_b| )^2 
    #    If |c_b| < 1-r, the term is positive, pushing it out? 
    #    d/dx ( (1-r) - |x| )^2 = -2 (1-r - |x|) * x/|x|.
    #    If |x| < 1-r, gradient is -2(positive) * unit_vec. Force is -grad = +2(positive) * unit_vec. Pushes OUT. Correct.
    #    If |x| > 1-r, gradient is -2(negative) * unit_vec = +... Force is -... Pushes IN. Correct (prevents going outside).
    
    x0 = centers[boundary_idx].flatten()
    
    def boundary_obj(x):
        cb = x.reshape(-1, 2)
        # Overlap with interior
        diff_bi = cb[:, np.newaxis, :] - centers[interior_idx][np.newaxis, :, :]
        dists_bi = np.sqrt(np.sum(diff_bi ** 2, axis=-1) + 1e-12)
        overlap_bi = np.maximum(0, 2 * r_target - dists_bi)
        e_bi = np.sum(overlap_bi ** 2)
        
        # Overlap with other boundary
        n_b = len(cb)
        if n_b > 1:
            diff_bb = cb[:, np.newaxis, :] - cb[np.newaxis, :, :]
            dists_bb = np.sqrt(np.sum(diff_bb ** 2, axis=-1) + 1e-12)
            mask_bb = np.triu(np.ones((n_b, n_b), dtype=bool), k=1)
            overlap_bb = np.maximum(0, 2 * r_target - dists_bb[mask_bb])
            e_bb = np.sum(overlap_bb ** 2)
        else:
            e_bb = 0.0
            
        # Wall term
        radii_b = np.linalg.norm(cb, axis=1)
        wall_dist = (1.0 - r_target) - radii_b
        e_wall = np.sum(wall_dist ** 2)
        
        total_e = e_bi + e_bb + 1.0 * e_wall
        
        # Gradient
        grad_x = np.zeros_like(x)
        
        # Grad e_bi
        w_bi = 2.0 * overlap_bi
        # grad w.r.t c_b: sum_i w_bi * (c_b - c_i) / d_bi
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
            
        # Grad e_wall
        # d/dx ( (1-r) - |x| )^2 = -2 (1-r - |x|) * x/|x|
        safe_radii_b = np.where(radii_b < 1e-12, 1e-12, radii_b)
        grad_wall = -2.0 * wall_dist[:, np.newaxis] * cb / safe_radii_b[:, np.newaxis]
        
        grad_x = (grad_bi + grad_bb + grad_wall).flatten()
        
        return total_e, grad_x
        
    result = minimize(
        boundary_obj,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
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
    
    # Phase 1: MD Relaxation to find a good basin
    # We ramp up r
    r_start = 0.01
    r_end = 0.0531
    n_steps = 5
    current_x = centers.copy()
    
    for i in range(n_steps):
        r_target = r_start + (r_end - r_start) * (i / n_steps)
        # Run MD
        current_x = _md_relax(current_x, r_target, steps=100, dt=0.005)
        
    # Phase 2: L-BFGS Polish on full set
    x0 = current_x.flatten()
    
    def scipy_obj(x):
        e, g = _overlap_energy(x, r_end)
        return e
        
    def scipy_grad(x):
        e, g = _overlap_energy(x, r_end)
        return g
        
    result = minimize(
        scipy_obj,
        x0,
        jac=scipy_grad,
        method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    optimized = result.x.reshape(-1, 2)
    
    # Phase 3: Boundary Optimization
    optimized = _optimize_boundary(optimized, r_end)
    
    final_score = _compute_score(optimized)
    initial_score = _compute_score(centers)
    
    if final_score < initial_score:
        optimized = centers
        
    # Final safety check
    for i in range(len(optimized)):
        r = np.linalg.norm(optimized[i])
        if r > 1.0 - 1e-10:
            scale = (1.0 - 1e-10) / r
            optimized[i] *= scale
            
    return optimized
# EVOLVE-BLOCK-END
