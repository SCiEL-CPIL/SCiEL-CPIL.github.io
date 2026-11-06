"""
Pack 200 equal circles in the unit square [-1/2, 1/2]^2 to maximize common radius.
Uses a vectorized repulsion-based local search with multiple restarts and 
hexagonal grid initialization.
"""
import numpy as np
import time
from scipy.spatial.distance import pdist

N = 200
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distances: min_i (0.5 - |c_i|)
    # For each point, distance to closest wall
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(dx, dy)
    min_wall = np.min(wall_dists)
    
    if min_wall < 0:
        return 0.0
        
    if N < 2:
        return float(min_wall)
        
    # Pairwise distances
    d = pdist(centers)
    min_pair = np.min(d)
    pair_term = min_pair / 2.0
    
    return min(float(min_wall), float(pair_term))


def _hexagonal_grid(n_circles):
    """
    Create a hexagonal packing arrangement for n_circles circles.
    Returns an array of shape (n_circles, 2).
    """
    best_centers = None
    best_r = 0.0

    # Try different numbers of rows
    # For N=200, rows around 10-20 are good.
    for m in range(10, 25):
        base_per_row = N // m
        extra = N % m

        row_counts = []
        for i in range(m):
            if i < extra:
                row_counts.append(base_per_row + 1)
            else:
                row_counts.append(base_per_row)

        max_row_count = max(row_counts)

        # Estimate radius
        # Width constraint: max_row_count * 2r <= 1  => r <= 1/(2*max_row_count)
        # Height constraint: (m-1)*sqrt(3)*r + 2r <= 1 => r <= 1/((m-1)*sqrt(3) + 2)
        r_width = 1.0 / (2.0 * max_row_count)
        r_height = 1.0 / ((m - 1) * np.sqrt(3) + 2)
        r_est = min(r_width, r_height)

        if r_est <= best_r and best_centers is not None:
            continue

        dx = 2.0 * r_est
        dy = np.sqrt(3) * r_est

        centers = []
        y_offset = -0.5 + r_est

        for row_idx, row_count in enumerate(row_counts):
            x_offset = 0.0
            if row_idx % 2 == 1:
                x_offset = dx / 2.0

            if row_count > 0:
                row_width = (row_count - 1) * dx
                start_x = -row_width / 2.0 + x_offset
            else:
                start_x = 0.0

            for col_idx in range(row_count):
                x = start_x + col_idx * dx
                y = y_offset + row_idx * dy
                centers.append([x, y])

        centers = np.array(centers)

        # Center the arrangement in the square
        centers[:, 0] -= np.mean(centers[:, 0])
        centers[:, 1] -= np.mean(centers[:, 1])

        # Clip to be within bounds
        centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)

        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()

    return best_centers if best_centers is not None else np.random.uniform(-0.5, 0.5, (N, 2))


def _refine_repulsion_vectorized(centers, max_iters=5000):
    """
    Refine the packing using a vectorized repulsion-based optimizer.
    Moves all overlapping/close circles simultaneously.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Initial step size
    step_size = 0.05
    no_improve_count = 0
    stagnation_threshold = 50
    
    for it in range(max_iters):
        if _remaining_time() < 2.0:
            break
            
        # Compute pairwise distances
        d = pdist(centers)
        
        # Find the minimum pair distance to set a threshold for "close" pairs
        # We want to resolve all conflicts, not just the minimum one.
        # Let's use a cutoff of 1.1 * min_dist to handle clusters.
        min_d = np.min(d)
        if min_d < 1e-6:
            # If circles are coincident, add random jitter
            centers += np.random.randn(*centers.shape) * 0.01
            centers = np.clip(centers, -0.5, 0.5)
            continue
            
        # Threshold for repulsion
        # We want to push apart any pair that is closer than, say, the target radius * 2 
        # But we don't know the target radius. 
        # Let's use a fixed small threshold relative to the current min distance?
        # Or better: calculate forces for ALL pairs with d < 2 * r_current?
        # r_current is roughly min_d/2.
        # So d < min_d is the only overlapping pair if we assume perfect packing.
        # But in a packing, many pairs are at distance ~min_d.
        # Let's apply force to all pairs with d < min_d * 1.5.
        
        threshold = min_d * 1.5
        
        # Get indices of pairs below threshold
        # pdist is condensed. We need to map back to i, j.
        # For N=200, number of pairs is ~20k. Sorting/filtering is fast.
        
        # Instead of filtering indices, we can compute forces for all pairs and mask.
        # But vectorizing the force calculation for all pairs is O(N^2) memory if we store matrices.
        # N=200, N^2 = 40k. This is small.
        
        # Compute distance matrix (N, N)
        # We already have d (condensed). Let's reconstruct or just use condensed.
        # Using condensed is tricky for vectorized force summation without index mapping.
        # Let's use the full distance matrix for clarity and speed (N=200 is small).
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        # Mask out self-interactions
        np.fill_diagonal(dists, np.inf)
        
        # Identify active pairs
        active_mask = dists < threshold
        
        if not np.any(active_mask):
            # No conflicts? Unlikely in a dense packing, but if so, we are done or stuck.
            # Actually, if no pairs are closer than threshold, the min distance is > threshold.
            # This means we have expanded? 
            # Break if we are stable.
            if min_d > best_r * 2 * 0.99: # Approx check
                 break
            # Otherwise, just continue with smaller step?
            pass

        # Compute forces
        # F_ij = (1/d) * (d - r_target) ? No, we want to maximize min distance.
        # Standard repulsion: F = k * (1/d^2) or linear.
        # Let's use a simple linear repulsion for d < threshold.
        # Force magnitude proportional to (threshold - d).
        # Direction: away from each other.
        
        # Normalize diff
        with np.errstate(divide='ignore', invalid='ignore'):
            norms = dists
            # Avoid division by zero
            norms[norms == 0] = 1.0
            dir_vecs = diff / norms[:, :, np.newaxis]
        
        # Force magnitude
        # We want to push apart if d < threshold.
        # strength = (threshold - dists)
        strength = np.where(active_mask, threshold - dists, 0.0)
        
        # Force vector for each pair i->j is dir_vecs[i,j] * strength[i,j]
        # But force on i is sum of forces from j.
        # F_i = sum_j (dir_vecs[i,j] * strength[i,j])
        # Note: dir_vecs[i,j] = (c_i - c_j) / d.
        # If d < threshold, strength > 0.
        # Force on i is in direction (c_i - c_j), i.e., away from j. Correct.
        
        forces = np.sum(dir_vecs * strength[:, :, np.newaxis], axis=1)
        
        # Add wall constraints
        # If a point is too close to a wall, push it in.
        # Wall force is much stronger.
        dx = 0.5 - np.abs(centers[:, 0])
        dy = 0.5 - np.abs(centers[:, 1])
        
        # If x < -0.5 + r_min, push right. If x > 0.5 - r_min, push left.
        # We don't know r_min exactly, but we can use a small epsilon or the current min_wall.
        min_wall = np.min(np.minimum(dx, dy))
        wall_margin = max(0.01, min_wall * 0.5) # Small margin
        
        # Left wall: x < -0.5 + wall_margin
        # Right wall: x > 0.5 - wall_margin
        # Bottom: y < -0.5 + wall_margin
        # Top: y > 0.5 - wall_margin
        
        wall_force_x = np.zeros(N)
        wall_force_y = np.zeros(N)
        
        # Left
        mask_l = centers[:, 0] < -0.5 + wall_margin
        wall_force_x[mask_l] += 1.0
        # Right
        mask_r = centers[:, 0] > 0.5 - wall_margin
        wall_force_x[mask_r] -= 1.0
        # Bottom
        mask_b = centers[:, 1] < -0.5 + wall_margin
        wall_force_y[mask_b] += 1.0
        # Top
        mask_t = centers[:, 1] > 0.5 - wall_margin
        wall_force_y[mask_t] -= 1.0
        
        # Scale wall forces to be comparable to repulsion forces
        # Repulsion force magnitude is roughly (threshold - d). 
        # Let's scale wall forces by a factor.
        wall_scale = 10.0
        forces[:, 0] += wall_force_x * wall_scale
        forces[:, 1] += wall_force_y * wall_scale
        
        # Normalize forces to prevent huge jumps
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_norms)
        if max_force > 1e-6:
            # Limit step size
            # We want step_size * force to be small.
            # Let's normalize forces to unit length and multiply by step_size?
            # Or just scale down if too large.
            pass
            
        # Update positions
        centers += step_size * forces
        
        # Clip to square
        centers = np.clip(centers, -0.5, 0.5)
        
        # Evaluate
        new_r = _compute_radius(centers)
        
        if new_r > best_r + 1e-9:
            best_r = new_r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
            # Adaptive step size
            if no_improve_count > stagnation_threshold:
                # Random perturbation to escape local minima
                # Move a few random points
                num_perturb = 5
                idxs = np.random.choice(N, num_perturb, replace=False)
                centers[idxs] += np.random.randn(num_perturb, 2) * 0.05
                centers = np.clip(centers, -0.5, 0.5)
                no_improve_count = 0
                step_size *= 0.8
            elif no_improve_count > 20:
                step_size *= 0.95
                
        # Decay step size slowly
        step_size *= 0.999
        
    return best_centers, best_r


def construct():
    """Construct optimal circle packing."""
    # Start with hexagonal grid
    centers = _hexagonal_grid(N)
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Refine with vectorized repulsion
    if _remaining_time() > 5.0:
        try:
            refined_centers, refined_radius = _refine_repulsion_vectorized(centers)
            if refined_radius > best_radius:
                best_radius = refined_radius
                best_centers = refined_centers.copy()
        except Exception:
            pass
            
    # If we have time, try a second pass with a different initialization or restart
    if _remaining_time() > 10.0:
        # Try a slightly perturbed version of the best solution
        perturbed = best_centers + np.random.randn(N, 2) * 0.01
        perturbed = np.clip(perturbed, -0.5, 0.5)
        try:
            refined_centers2, refined_radius2 = _refine_repulsion_vectorized(perturbed, max_iters=2000)
            if refined_radius2 > best_radius:
                best_radius = refined_radius2
                best_centers = refined_centers2.copy()
        except Exception:
            pass

    # Final safety clip
    best_centers = np.clip(best_centers, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_centers
