import numpy as np
import time

N = 50

def _compute_radius(centers):
    """
    Compute the radius of the packing.
    """
    # Pairwise distances
    # centers shape (N, 3)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    # Cube is [-0.5, 0.5]^3
    # Distance to boundary for coordinate k is 0.5 - abs(center_k)
    abs_c = np.abs(centers)
    boundary_dists = 0.5 - abs_c
    min_boundary_dist = np.min(boundary_dists)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def _local_refine(centers, max_iters=1000, step_size=0.01):
    """
    Local refinement using subgradient ascent on the min-distance objective.
    """
    centers = centers.copy()
    n = len(centers)
    
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    best_r = _compute_radius(centers)
    no_improve_count = 0
    patience = 100
    
    for it in range(max_iters):
        # Compute pairwise distances
        diff = centers[i_idx] - centers[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        
        # Compute boundary distances
        abs_c = np.abs(centers)
        boundary_dists = 0.5 - abs_c
        min_boundary_dist = np.min(boundary_dists)
        
        current_r = min(np.min(dists) / 2.0, min_boundary_dist)
        
        if current_r > best_r:
            best_r = current_r
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > patience:
                break
        
        # Identify active constraints
        # We want to increase the minimum distance.
        # Let's use a threshold slightly above the current min to include all "tight" constraints.
        threshold = current_r + 0.001
        
        grad = np.zeros_like(centers)
        
        # Pair constraints
        active_pairs = dists < 2.0 * threshold
        if np.any(active_pairs):
            # For each active pair (i, j), push i away from j and j away from i
            # Direction is diff = c_i - c_j
            # Normalize direction
            active_diffs = diff[active_pairs]
            active_dists = dists[active_pairs]
            
            # Avoid division by zero
            norms = np.linalg.norm(active_diffs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-6
            unit_vecs = active_diffs / norms
            
            # Weight by how "tight" they are? Or just uniform?
            # Uniform is simpler.
            # Scatter add to grad
            for k in range(len(active_diffs)):
                i = i_idx[active_pairs][k]
                j = j_idx[active_pairs][k]
                grad[i] += unit_vecs[k]
                grad[j] -= unit_vecs[k]
        
        # Boundary constraints
        # For each point i and dimension k, if 0.5 - abs(c_ik) < threshold
        active_boundaries = boundary_dists < threshold
        if np.any(active_boundaries):
            # Create a force vector for each point
            # If c_ik > 0, push in -k direction? No, push away from boundary.
            # Boundary is at 0.5. If c_ik is close to 0.5, push towards 0.
            # Boundary is at -0.5. If c_ik is close to -0.5, push towards 0.
            # So force direction is -sign(c_ik) * e_k
            force = np.zeros_like(centers)
            # active_boundaries is a boolean array of shape (N, 3)
            # We want to add -sign(centers) for those active
            # sign(centers) is 1 if positive, -1 if negative.
            # Pushing away from boundary means pushing towards center.
            # So if c > 0, push left (-x). If c < 0, push right (+x).
            # Direction is -sign(c).
            signs = np.sign(centers)
            # Only apply where active
            active_forces = -signs * active_boundaries
            force += active_forces
            grad += force
            
        # Normalize gradient
        grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norms[grad_norms == 0] = 1.0
        grad = grad / grad_norms
        
        # Update
        # Adaptive step size?
        # Start with small step, decrease if it oscillates or doesn't improve.
        # For now, fixed small step.
        step = step_size
        centers += step * grad
        
        # Project back to cube?
        # The force should keep them inside, but numerical errors might push them out.
        # Clipping to [-0.5, 0.5] is safe but might stick to boundary.
        # Better to ensure they don't cross.
        # If a point is pushed out, the boundary force will pull it back next iteration.
        # However, if it crosses, the distance calculation becomes invalid for the "inside" constraint.
        # We should clip to [-0.5 + eps, 0.5 - eps] to maintain validity?
        # The score calculation handles boundary distance. If it goes outside, min_boundary_dist becomes negative, which is bad.
        # So we MUST ensure centers stay inside.
        centers = np.clip(centers, -0.5, 0.5)
        
    return centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_centers = None
    best_r = 0.0
    
    # Initial configuration: 4x4x4 grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_centers = grid[:N].copy()
    
    # Run optimization on the initial grid
    # We can try multiple restarts with different noise levels or initial configs
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        # Add noise to break symmetry
        noise_scale = 0.01 * (1 + 0.5 * np.sin(restarts)) 
        centers = init_centers + np.random.normal(0, noise_scale, init_centers.shape)
        centers = np.clip(centers, -0.5, 0.5)
        
        # Refine
        refined_centers, r = _local_refine(centers, max_iters=500, step_size=0.02)
        
        if r > best_r:
            best_r = r
            best_centers = refined_centers
            
        # Check time
        if time.time() > deadline:
            break
            
        # Optional: try a different initial config if time permits
        # e.g. random init
        if restarts % 5 == 0:
             random_centers = np.random.uniform(-0.4, 0.4, (N, 3))
             refined_centers, r = _local_refine(random_centers, max_iters=500, step_size=0.02)
             if r > best_r:
                 best_r = r
                 best_centers = refined_centers
                 if time.time() > deadline:
                     break

    if best_centers is None:
        return init_centers
    return best_centers
