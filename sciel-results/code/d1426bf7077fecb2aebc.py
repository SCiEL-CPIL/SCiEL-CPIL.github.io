"""Pack 300 equal circles in a unit disk using barrier method optimization."""
import numpy as np
import time
import math

N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _golden_spiral_init(n, radius=1.0):
    """
    Generate n points using golden angle spiral (sunflower model).
    """
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    k = np.arange(n, dtype=np.float64)
    r = radius * np.sqrt((k + 0.5) / n)
    theta = k * golden_angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _compute_min_radius(points):
    """Compute the maximum possible radius for the given points."""
    if len(points) == 0:
        return 0.0
    
    dists_to_center = np.linalg.norm(points, axis=1)
    boundary_min = np.min(1.0 - dists_to_center)
    
    if boundary_min < 0:
        return 0.0
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    
    pair_min = min_pair_dist / 2.0
    
    return min(boundary_min, pair_min)

def _local_search(points, max_iters=5000, time_limit=100.0, target_ratio=0.5):
    """
    Vectorized gradient ascent on a soft-min of pairwise distances and boundary margins.
    Biases force toward close pairs to push the minimum up.
    """
    points = points.copy().astype(np.float64)
    n = len(points)
    start_time = time.time()
    
    best_r = _compute_min_radius(points)
    best_points = points.copy()
    
    lr = 0.005
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute all pairwise distances (vectorized)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_matrix = np.sqrt(dist_sq)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Boundary margins
        dists_to_center = np.linalg.norm(points, axis=1)
        boundary_margin = 1.0 - dists_to_center
        boundary_margin = np.maximum(boundary_margin, 1e-10)
        
        # Compute current min radius
        current_r = min(np.min(boundary_margin), np.min(dist_matrix) / 2.0)
        
        # Soft-min: use weighted sum emphasizing small distances
        # Weight pairs by exp(-d / scale) so close pairs dominate
        scale = max(current_r * 2.0, 1e-4)
        
        # Pairwise weights: emphasize close pairs
        pair_weights = np.exp(-dist_matrix / scale)
        np.fill_diagonal(pair_weights, 0.0)
        
        # Gradient from pairwise repulsion with soft-min weighting
        # Force on point i from point j: w_ij * (p_i - p_j) / d_ij
        # where w_ij = exp(-d_ij / scale)
        # This is O(n^2) but fully vectorized
        
        # Compute weighted forces
        # For pair (i,j): force on i = w_ij * diff[i,j] / dist_matrix[i,j]
        # diff[i,j] = points[i] - points[j], so force pushes i away from j
        
        safe_dist = np.maximum(dist_matrix, 1e-10)
        # Weighted unit vectors: w_ij * diff_ij / d_ij = w_ij * diff_ij / d_ij
        # = w_ij * diff_ij / sqrt(dist_sq)
        # Since w_ij = exp(-d/scale), and diff/d = unit vector
        weight_factor = pair_weights / safe_dist  # shape (n, n)
        
        # Force on each point: sum over j of weight_factor[i,j] * diff[i,j]
        # diff[i,j] = points[i] - points[j]
        # Sum_j weight_factor[i,j] * (points[i] - points[j])
        # = points[i] * sum_j weight_factor[i,j] - sum_j weight_factor[i,j] * points[j]
        
        wf_sum = np.sum(weight_factor, axis=1)  # shape (n,)
        # sum_j weight_factor[i,j] * points[j]
        weighted_points = weight_factor @ points  # shape (n, 2)
        
        pair_force = points * wf_sum[:, np.newaxis] - weighted_points
        
        # Boundary force: push inward if near boundary
        # Force = -points[i] / (d_i * margin_i) * margin_weight
        # Emphasize points close to boundary
        boundary_weight = np.exp(-boundary_margin / scale)
        safe_center_dist = np.maximum(dists_to_center, 1e-10)
        boundary_force = -points * (boundary_weight / (safe_center_dist * boundary_margin))[:, np.newaxis]
        
        # Total gradient
        grad = pair_force + boundary_force
        
        # Normalize
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad = grad / grad_norm
        
        # Adaptive step
        new_points = points + lr * grad
        
        # Project to unit disk
        new_dists = np.linalg.norm(new_points, axis=1)
        outside_mask = new_dists > 0.9999
        if np.any(outside_mask):
            new_points[outside_mask] *= (0.9999 / new_dists[outside_mask, np.newaxis])
        
        new_r = _compute_min_radius(new_points)
        
        if new_r >= best_r - 1e-12:
            points = new_points
            if new_r > best_r + 1e-10:
                best_r = new_r
                best_points = points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            # Slightly increase lr
            lr = min(lr * 1.02, 0.02)
        else:
            lr *= 0.9
            no_improve_count += 1
            if lr < 0.0005:
                break
        
        # Early termination
        if no_improve_count > 200:
            break
    
    return best_points, best_r

def _simulated_annealing(points, max_iters=20000, time_limit=50.0):
    """
    Fast simulated annealing with vectorized distance computation.
    """
    points = points.copy().astype(np.float64)
    n = len(points)
    start_time = time.time()
    
    current_r = _compute_min_radius(points)
    best_r = current_r
    best_points = points.copy()
    
    T = 0.02
    T_min = 1e-7
    cooling = 0.9995
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Propose random perturbation
        perturbation = np.random.normal(0, T, size=(n, 2))
        new_points = points + perturbation
        
        # Project to unit disk
        new_dists = np.linalg.norm(new_points, axis=1)
        outside_mask = new_dists > 0.9999
        if np.any(outside_mask):
            new_points[outside_mask] *= (0.9999 / new_dists[outside_mask, np.newaxis])
        
        new_r = _compute_min_radius(new_points)
        
        if new_r >= current_r:
            points = new_points
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_points = points.copy()
        else:
            delta = current_r - new_r
            if delta < 1e-14:
                points = new_points
                current_r = new_r
            else:
                accept_prob = np.exp(-delta / (T + 1e-15))
                if np.random.random() < accept_prob:
                    points = new_points
                    current_r = new_r
        
        T = max(T * cooling, T_min)
    
    return best_points, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    np.random.seed(42)
    
    # 1. Initialize with golden spiral
    points = _golden_spiral_init(N)
    best_r = _compute_min_radius(points)
    best_points = points.copy()
    
    # 2. Phase 1: Local search (vectorized gradient ascent)
    if time.time() < deadline - 60.0:
        opt_points, opt_r = _local_search(points, max_iters=10000, time_limit=50.0)
        if opt_r > best_r:
            best_points = opt_points
            best_r = opt_r
    
    # 3. Phase 2: Simulated annealing
    if time.time() < deadline - 30.0:
        sa_points, sa_r = _simulated_annealing(best_points, max_iters=20000, time_limit=30.0)
        if sa_r > best_r:
            best_points = sa_points
            best_r = sa_r
    
    # 4. Phase 3: Final local search refinement
    if time.time() < deadline - 5.0:
        final_points, final_r = _local_search(best_points, max_iters=5000, time_limit=15.0)
        if final_r > best_r:
            best_points = final_points
            best_r = final_r
    
    # 5. One more SA + local search if time permits
    if time.time() < deadline - 10.0:
        sa2_points, sa2_r = _simulated_annealing(best_points, max_iters=10000, time_limit=10.0)
        if sa2_r > best_r:
            best_points = sa2_points
            best_r = sa2_r
        if time.time() < deadline - 5.0:
            fin2_points, fin2_r = _local_search(best_points, max_iters=3000, time_limit=5.0)
            if fin2_r > best_r:
                best_points = fin2_points
                best_r = fin2_r
    
    return best_points
# EVOLVE-BLOCK-END
