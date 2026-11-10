"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _compute_radius(pts):
    """Compute the max radius for given centers in [-0.5, 0.5]^2."""
    if len(pts) < 2:
        return 0.0
    # Min distance between pairs
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    min_pair = d[:, 1].min()
    # Distance to boundary
    bounds = 0.5 - np.abs(pts).max(axis=1)
    min_bound = bounds.min()
    r = min(min_pair / 2.0, min_bound)
    return r

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin
    
    # 1. Hexagonal Initialization
    # Generate a hexagonal lattice of points
    # For N=200, we need a grid of roughly 15x14 or similar
    # Hexagonal packing density is higher than square.
    # Let's generate points on a hexagonal grid and select the N closest to center.
    
    # Grid parameters
    # Side length of hexagon in lattice units
    # We want to cover [-0.5, 0.5]^2
    # Let's generate a large grid and filter
    
    # Hexagonal lattice vectors
    # v1 = (1, 0), v2 = (0.5, sqrt(3)/2)
    # Scale factor 's' determines spacing.
    # We don't know optimal 's' yet, so we generate a dense grid and normalize?
    # Better: Estimate spacing.
    # Area = 1.0. N=200. Area per point = 0.005.
    # Hexagon area per point = s^2 * sqrt(3)/2.
    # s^2 = 0.005 * 2 / sqrt(3) = 0.01 / 1.732 = 0.00577
    # s = sqrt(0.00577) = 0.076
    
    s = 0.076
    v1 = np.array([s, 0])
    v2 = np.array([s/2, s*np.sqrt(3)/2])
    
    # Generate grid points
    # Range: -1 to 1 to cover square with margin
    max_idx = int(np.ceil(1.0 / s)) + 2
    indices = np.arange(-max_idx, max_idx + 1)
    
    # Create meshgrid in lattice coordinates
    i, j = np.meshgrid(indices, indices, indexing='ij')
    
    # Convert to Cartesian
    x = i * v1[0] + j * v2[0]
    y = i * v1[1] + j * v2[1]
    
    pts_grid = np.column_stack((x.ravel(), y.ravel()))
    
    # Filter points that are inside or near the square [-0.5, 0.5]^2
    # To ensure we have enough points, we keep those within [-0.5, 0.5]
    # If we don't have enough, we expand the range or adjust s.
    # For s=0.076, width 1.0 / 0.076 ~ 13 points. 13*13 = 169. Close to 200.
    # We might need slightly smaller s or larger range.
    
    # Let's just generate a large number of candidates and pick the best N
    # Or simpler: Generate a grid of size MxM where M is large enough.
    
    # Let's try to generate exactly N points by adjusting s slightly or picking closest to center.
    # Actually, for force-directed, initialization just needs to be non-overlapping and dense.
    # Let's use a square grid for simplicity of generation, but perturb it to hex-like?
    # No, the prompt says hexagonal initialization provides improvement.
    
    # Let's generate a hex grid with s=0.07 and see how many points fit.
    # If < 200, we need smaller s.
    
    # Robust generation:
    # Try to generate a hex grid that contains at least N points in the square.
    # We can generate a large grid and then select the N points closest to the center of the square.
    
    s = 0.07
    v1 = np.array([s, 0])
    v2 = np.array([s/2, s*np.sqrt(3)/2])
    
    max_idx = 15
    indices = np.arange(-max_idx, max_idx + 1)
    i, j = np.meshgrid(indices, indices, indexing='ij')
    x = i * v1[0] + j * v2[0]
    y = i * v1[1] + j * v2[1]
    pts_grid = np.column_stack((x.ravel(), y.ravel()))
    
    # Keep points within [-0.5, 0.5]^2
    mask = (np.abs(pts_grid[:, 0]) <= 0.5) & (np.abs(pts_grid[:, 1]) <= 0.5)
    pts_inside = pts_grid[mask]
    
    if len(pts_inside) < N:
        # If not enough, reduce s
        s = 0.06
        v1 = np.array([s, 0])
        v2 = np.array([s/2, s*np.sqrt(3)/2])
        i, j = np.meshgrid(indices, indices, indexing='ij')
        x = i * v1[0] + j * v2[0]
        y = i * v1[1] + j * v2[1]
        pts_grid = np.column_stack((x.ravel(), y.ravel()))
        mask = (np.abs(pts_grid[:, 0]) <= 0.5) & (np.abs(pts_grid[:, 1]) <= 0.5)
        pts_inside = pts_grid[mask]
        
    # If still not enough, or too many, select N closest to center (0,0)
    dists_to_center = np.sum(pts_inside**2, axis=1)
    idx_sorted = np.argsort(dists_to_center)
    pts = pts_inside[idx_sorted[:N]].copy()
    
    # Add small random jitter to break symmetry
    np.random.seed(42)
    pts += np.random.randn(N, 2) * 0.001
    pts = np.clip(pts, -0.499, 0.499)
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # 2. Force-Directed Optimization
    # We simulate repulsive forces between circles and attractive force to center?
    # No, just repulsive forces to maximize min distance, and boundary constraints.
    # Objective: Maximize min distance.
    # Gradient of min distance is 0 for non-min pairs.
    # Force-directed: For each pair (i,j), force F_ij = k * (d_ij - d_target) if d_ij < d_target?
    # Or simply repulsive force F_ij = 1 / d_ij^2 along vector d_ij.
    # This pushes points apart.
    # We need to balance this with keeping them in the box.
    # If we just push apart, they will hit the walls.
    # So we add a force pushing them back if they are too close to walls.
    
    # Let's use a temperature-based cooling schedule.
    # At high T, allow large moves. At low T, fine tune.
    
    current_pts = pts.copy()
    
    # Parameters
    num_steps = 2000
    learning_rate = 0.01
    decay = 0.999
    
    for step in range(num_steps):
        if time.time() > deadline:
            break
            
        # Compute forces
        # Use KDTree to find neighbors within a cutoff distance to save time?
        # With N=200, full N^2 is 40k ops, which is fast.
        # Let's do full pairwise for simplicity and robustness, vectorized.
        
        # Diff matrix
        diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dists[dists == 0] = 1e-6
        
        # Force magnitude: Repulsion
        # F = 1 / dist^2
        # We only want to push if they are "too close". 
        # But for maximizing min distance, we want to push the closest pairs the most.
        # A common potential is WCA (Weeks-Chandler-Andersen) or just inverse square.
        # Let's use a soft repulsion: F = max(0, 1/dist - 1/r0)^2 ?
        # Simpler: Gradient of sum of 1/dist?
        # Let's use the gradient of the "min distance" proxy.
        # Actually, standard force-directed layout uses F = k * (d - L) or similar.
        # For packing, we want to increase the smallest distance.
        
        # Let's use a potential energy E = sum_{i<j} 1/d_ij^2.
        # Minimizing E maximizes distances on average.
        # Force on i from j: F_ij = 2 * (r_i - r_j) / d_ij^4
        
        # Vectorized calculation
        # dists is NxN
        # We need to mask diagonal
        np.fill_diagonal(dists, 1.0)
        
        # Force vectors
        # F_ij = k * diff_ij / dists_ij^3  (for 1/d^2 potential, grad is -2/d^3)
        # Let's use k=1.
        inv_dist3 = 1.0 / (dists**3)
        inv_dist3[np.eye(N, dtype=bool)] = 0.0
        
        # Forces = sum over j of (r_i - r_j) / |r_i - r_j|^3
        # diff is r_i - r_j
        # So force contribution is diff * inv_dist3
        forces = np.sum(diff * inv_dist3[:, :, np.newaxis], axis=1)
        
        # Boundary forces
        # Push back if close to boundary
        # x in [-0.5, 0.5]
        # If x > 0.5 - margin, push left.
        margin = 0.01
        # x_force
        x_bound = 0.5 - margin
        y_bound = 0.5 - margin
        
        # Smooth boundary force
        # F_x = -k_b * (x - x_bound) if x > x_bound
        # F_x = k_b * (-x_bound - x) if x < -x_bound
        
        # Let's use a linear penalty
        k_b = 10.0
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Right boundary
        mask_r = current_pts[:, 0] > x_bound
        fx[mask_r] -= k_b * (current_pts[mask_r, 0] - x_bound)
        
        # Left boundary
        mask_l = current_pts[:, 0] < -x_bound
        fx[mask_l] += k_b * (-x_bound - current_pts[mask_l, 0])
        
        # Top boundary
        mask_t = current_pts[:, 1] > y_bound
        fy[mask_t] -= k_b * (current_pts[mask_t, 1] - y_bound)
        
        # Bottom boundary
        mask_b = current_pts[:, 1] < -y_bound
        fy[mask_b] += k_b * (-y_bound - current_pts[mask_b, 1])
        
        forces[:, 0] += fx
        forces[:, 1] += fy
        
        # Update positions
        # Adaptive step size?
        # Normalize forces to avoid huge jumps
        norm_forces = np.linalg.norm(forces, axis=1, keepdims=True)
        norm_forces[norm_forces == 0] = 1.0
        # Cap force magnitude
        max_force = 0.1
        forces = forces * (max_force / norm_forces)
        
        new_pts = current_pts + learning_rate * forces
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        # Check if radius improved
        new_r = _compute_radius(new_pts)
        
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts.copy()
            
        # Accept move if it doesn't decrease radius too much (Simulated Annealing style?)
        # Or just always accept if it improves local energy?
        # For now, let's only accept if it improves the global min radius or is equal.
        # This might get stuck.
        # Better: Accept if new_r >= current_r?
        # We track current_r separately.
        
        # Let's use a simple hill-climbing on the radius.
        # If new_r > current_r, accept.
        # If new_r < current_r, reject (and maybe backtrack).
        
        # We need to track current_r.
        # Initially current_r = best_r.
        if step == 0:
            current_r = best_r
            
        if new_r >= current_r:
            current_pts = new_pts
            current_r = new_r
        else:
            # Reject, try smaller step?
            # For simplicity, just reject.
            pass
            
        # Decay learning rate
        learning_rate *= decay
        
        # Periodically check time
        if step % 100 == 0:
             pass
            
    # Final refinement: Local search on the best points
    # The force-directed method might leave some points slightly suboptimal.
    # Do a few rounds of "push closest pair" optimization.
    
    current_pts = best_pts.copy()
    current_r = best_r
    
    for refine_iter in range(500):
        if time.time() > deadline:
            break
            
        # Find closest pair
        tree = cKDTree(current_pts)
        d, idx_pairs = tree.query(current_pts, k=2)
        min_d = d[:, 1].min()
        min_idx_pair = np.argmin(d[:, 1])
        
        # Also check boundary
        bounds = 0.5 - np.abs(current_pts).max(axis=1)
        min_bound = bounds.min()
        min_bound_idx = np.argmin(bounds)
        
        if min_d / 2.0 < min_bound:
            # Closest pair is the bottleneck
            i = min_idx_pair
            j = idx_pairs[min_idx_pair, 1]
            
            # Move i and j apart
            vec = current_pts[i] - current_pts[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                vec = np.random.randn(2)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            # Move i by +vec * delta, j by -vec * delta
            # Optimal delta?
            # We want to increase min_d.
            # Try a few steps
            for delta in [0.001, 0.005, 0.01, 0.02]:
                new_i = current_pts[i] + vec * delta
                new_j = current_pts[j] - vec * delta
                
                # Check bounds
                if np.any(np.abs(new_i) > 0.499) or np.any(np.abs(new_j) > 0.499):
                    continue
                    
                temp_pts = current_pts.copy()
                temp_pts[i] = new_i
                temp_pts[j] = new_j
                
                temp_r = _compute_radius(temp_pts)
                
                if temp_r > current_r:
                    current_pts = temp_pts
                    current_r = temp_r
                    if current_r > best_r:
                        best_r = current_r
                        best_pts = current_pts.copy()
                    break
        else:
            # Boundary is bottleneck
            i = min_bound_idx
            # Move i towards center
            vec = -current_pts[i]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                vec = np.random.randn(2)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for delta in [0.001, 0.005, 0.01, 0.02]:
                new_i = current_pts[i] + vec * delta
                if np.any(np.abs(new_i) > 0.499):
                    continue
                
                temp_pts = current_pts.copy()
                temp_pts[i] = new_i
                
                temp_r = _compute_radius(temp_pts)
                
                if temp_r > current_r:
                    current_pts = temp_pts
                    current_r = temp_r
                    if current_r > best_r:
                        best_r = current_r
                        best_pts = current_pts.copy()
                    break
                    
    return best_pts
# EVOLVE-BLOCK-END
