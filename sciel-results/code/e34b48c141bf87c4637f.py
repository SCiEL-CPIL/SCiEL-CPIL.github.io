"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # --- Stage 1: Hexagonal Lattice Initialization ---
    # A hexagonal lattice provides a highly structured, near-optimal starting point
    # for dense packing, which is significantly better than spiral or random initialization.
    # We initialize with a spacing slightly larger than the expected optimal radius
    # to ensure a valid (non-overlapping) start, then relax.
    
    points = np.zeros((N, 2))
    idx = 0
    
    # We want to fit N points in a unit circle.
    # Area of hexagon with side s is (3*sqrt(3)/2)*s^2.
    # Approximate area per point in hex lattice is (sqrt(3)/2) * (2r)^2 = 2*sqrt(3)*r^2.
    # Total area ~ pi. So N * 2*sqrt(3)*r^2 ~ pi => r ~ sqrt(pi / (2*sqrt(3)*N)).
    # For N=300, r ~ sqrt(3.14 / (3.46*300)) ~ sqrt(3.14/1038) ~ sqrt(0.003) ~ 0.055.
    # Let's start with a slightly larger spacing to be safe, say r_init = 0.055.
    # Spacing d = 2 * r_init.
    
    r_init = 0.055
    d = 2 * r_init
    h = d * np.sqrt(3) / 2.0
    
    # Generate hex lattice points
    # Max radius in lattice: we need to cover the unit circle.
    # Number of rings needed: approx sqrt(N/3). For N=300, ~10 rings.
    # Let's generate points in a hex grid centered at origin.
    
    # We'll generate points in a square grid of hex lattice indices (i, j)
    # and keep those within radius < 1.
    # x = i * d + j * d/2
    # y = j * h
    
    # Estimate grid size
    max_i = int(np.ceil(1.0 / d)) + 2
    candidates = []
    for i in range(-max_i, max_i + 1):
        for j in range(-max_i, max_i + 1):
            x = i * d + j * d / 2.0
            y = j * h
            # Check if inside unit circle with a small margin for the circle radius
            # We want the centers to be inside 1 - r_init.
            if np.sqrt(x*x + y*y) <= 1.0 - r_init:
                candidates.append([x, y])
    
    # We have more candidates than N. We need to select N of them.
    # To maximize symmetry and density, we should select the N closest to the origin?
    # Actually, the hex lattice is uniform. If we take the N closest to origin,
    # we get a roughly circular patch of the lattice.
    # Let's sort by distance from origin and take the first N.
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    sorted_indices = np.argsort(dists)
    points = candidates[sorted_indices[:N]]
    
    # --- Stage 2: Force-Based Relaxation ---
    # Objective: Maximize the minimum distance between any two points and from boundary.
    # We use a repulsive force model.
    # Force between i and j: F_ij = k * (1/d_ij^3) * (p_i - p_j) / d_ij  (if d_ij < 2r_target)
    # Actually, simpler: just repel all pairs with a strong potential.
    # To maximize min distance, we can iterate:
    # 1. Compute forces based on current distances.
    # 2. Move points along forces.
    # 3. Project back to unit disk.
    # 4. Repeat.
    
    # We will use a "soft" constraint approach where we push points apart if they are too close.
    # Let's define a target radius r_target that we try to achieve.
    # Start with r_current = 0.052 (conservative) and increase it as we relax.
    
    r_current = 0.052
    points = points.copy()
    
    # Precompute neighbors? No, N=300 is small enough for O(N^2) per step if vectorized.
    # But we want to be fast.
    
    # We will run a loop for a fixed number of iterations or until time runs out.
    # Each iteration:
    # 1. Compute pairwise distances.
    # 2. Compute repulsive forces for pairs closer than 2 * r_target.
    # 3. Compute boundary forces for points closer than 1 - r_target.
    # 4. Update positions.
    
    # Parameters
    # Learning rate / step size
    step_size = 0.001
    # Number of iterations
    max_iters = 5000
    
    for iter in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        # Compute pairwise distances
        # dists[i, j] = |p_i - p_j|
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        # We want to push points apart if dist < 2 * r_target
        # Force magnitude proportional to 1/d^2 or 1/d^3 to be stiff near contact.
        # Let's use F = (2*r_target - d) / d^3 * (p_i - p_j)
        # This creates a strong repulsion when d is close to 2*r_target.
        
        # Mask for pairs that are too close
        # We only care about pairs where d < 2 * r_target + margin
        margin = 0.01
        mask = dists < (2 * r_current + margin)
        # Exclude diagonal
        mask = mask & ~np.eye(N, dtype=bool)
        
        if np.any(mask):
            # For these pairs, compute force
            # d_masked = dists[mask]
            # diff_masked = diff[mask]
            # This is tricky to vectorize directly into forces array without loops or advanced indexing.
            # Let's use a loop for the non-zero forces? No, N=300, 45k pairs. Loop is too slow in Python.
            # Vectorized approach:
            # F_i = sum_{j} w_ij * (p_i - p_j)
            # w_ij = (2*r - d_ij) / d_ij^3 if d_ij < 2*r else 0
            
            # Let's compute weights matrix
            w = np.zeros((N, N))
            # Avoid division by zero
            safe_dists = np.where(mask, dists, 1.0)
            w_vals = np.where(mask, (2 * r_current - safe_dists) / (safe_dists**3), 0.0)
            w = w_vals
            
            # Force on i is sum_j w_ij * (p_i - p_j)
            # = (sum_j w_ij) * p_i - sum_j w_ij * p_j
            sum_w = np.sum(w, axis=1)
            weighted_points = w @ points
            forces += sum_w[:, np.newaxis] * points - weighted_points
            
        # Boundary force
        # Push points inward if |p_i| > 1 - r_current
        norms = np.linalg.norm(points, axis=1)
        boundary_mask = norms > (1.0 - r_current - margin)
        if np.any(boundary_mask):
            # Force is inward: -p_i / |p_i| * strength
            # Strength proportional to violation
            violations = np.where(boundary_mask, norms - (1.0 - r_current), 0.0)
            # Normalize points for direction
            safe_norms = np.where(norms > 1e-9, norms, 1.0)
            directions = points / safe_norms[:, np.newaxis]
            # Add boundary force
            forces[boundary_mask] -= directions[boundary_mask] * (violations[boundary_mask] / (safe_norms[boundary_mask] + 1e-9))[:, np.newaxis] * 10.0 # Scale boundary force
            
        # Update points
        points += step_size * forces
        
        # Project to unit disk (hard constraint)
        norms = np.linalg.norm(points, axis=1)
        max_norm = 1.0 - r_current - 1e-6
        out_of_bounds = norms > max_norm
        if np.any(out_of_bounds):
            points[out_of_bounds] *= (max_norm / norms[out_of_bounds])[:, np.newaxis]
        
        # Try to increase r_current if configuration is valid
        # Check min distance and min boundary distance
        if iter % 100 == 0:
            min_pair_dist = np.min(dists)
            min_bound_dist = 1.0 - np.max(norms)
            current_min_r = min(min_pair_dist, min_bound_dist) / 2.0
            
            if current_min_r > r_current + 0.0001:
                r_current = current_min_r
                # Reset step size or keep it? Keep it.
                
    # Final polish: Ensure validity
    # The score is min(min_pair/2, min_bound).
    # We must ensure points are strictly inside.
    norms = np.linalg.norm(points, axis=1)
    # If any point is too close to edge, pull it in.
    # We want 1 - |p_i| >= r_final.
    # Let's calculate the actual best r we can support.
    
    dists = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    min_bound = 1.0 - np.max(norms)
    
    r_final = min(min_pair, min_bound)
    
    # If r_final is very small, something went wrong.
    # But we initialized well and relaxed, so it should be good.
    
    # Safety: ensure no point is outside
    norms = np.linalg.norm(points, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        points[mask] *= 0.999999
        
    return points

# EVOLVE-BLOCK-END
