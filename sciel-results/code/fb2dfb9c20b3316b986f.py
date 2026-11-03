"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(centers):
    """Compute the radius score: min of half min-distance and min boundary distance."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: centers must be inside [-0.5, 0.5]^2
    boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary_margin)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r = min(min_dist / 2.0, min_boundary)
    return r

def _initial_hex_grid(n):
    """Generate a high-quality hexagonal grid initialization."""
    # Optimal hex packing density is ~0.9069.
    # Area per point ~ 1/200 = 0.005.
    # Hex cell area for radius r is 2*sqrt(3)*r^2.
    # So 2*sqrt(3)*r^2 ~ 0.005 => r^2 ~ 0.005/3.464 ~ 0.00144 => r ~ 0.038.
    
    # We will generate a hex lattice and then scale/shift it to fit.
    r_est = 0.038
    dx = 2 * r_est
    dy = dx * np.sqrt(3) / 2
    
    best_points = None
    best_score = 0
    
    for n_rows in range(12, 18):
        n_cols = int(np.ceil(n / n_rows))
        if n_rows * n_cols < n:
            continue
            
        points = []
        for row in range(n_rows):
            y = row * dy
            offset = 0 if row % 2 == 0 else dx / 2
            for col in range(n_cols):
                x = col * dx + offset
                points.append((x, y))
        
        points = np.array(points[:n])
        
        # Center the points
        center = points.mean(axis=0)
        points -= center
        
        # Scale to fit in [-0.5, 0.5]^2 with some margin
        max_extent = np.max(np.abs(points))
        if max_extent > 0:
            scale = 0.45 / max_extent
            points *= scale
            
        if np.all(np.abs(points) <= 0.5 - 1e-10):
            score = _score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
    if best_points is not None:
        return best_points
        
    # Fallback: Square grid
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]

def _simulated_annealing(centers, deadline):
    """
    Optimize the packing using Simulated Annealing.
    Objective: Maximize the minimum distance (radius).
    Move: Randomly select a point and move it by a small random step.
    Acceptance: Metropolis criterion based on the change in the minimum distance.
    """
    centers = centers.copy()
    current_r = _score(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    # Initial temperature
    # The change in r is small, so T should be small but allow exploration.
    T = 1e-5 
    T_min = 1e-9
    cooling_rate = 0.9995
    
    # Step size for random moves
    step_size = 0.05
    
    # We need to efficiently find the min distance.
    # For N=200, O(N^2) is 40,000 ops. 
    # If we do 100,000 iterations, that's 4e9 ops. 
    # In Python/NumPy, vectorized O(N^2) is fast.
    # Let's estimate: 100,000 iters * (1 vectorized dist calc + score) 
    # Vectorized dist calc for 200x200 is ~1ms? 
    # 100,000 * 1ms = 100s. Too slow.
    # We need to be smarter or limit iterations.
    
    # Alternative: Use a KDTree to find nearest neighbor for the moved point?
    # No, the global min might be elsewhere.
    # However, if we only move one point, only distances involving that point change.
    # The global min is min( old_min_excluding_i, new_min_involving_i ).
    # We can maintain the global min and update it incrementally?
    # It's tricky because the old min might have involved point i.
    
    # Let's try a simpler approach: Batched evaluation with lower frequency or faster scoring.
    # Actually, let's use a different optimization: 
    # "Repulsion with hard constraints" is often better done with a specific algorithm like 
    # "Densest Packing" via local search on the graph of close pairs.
    
    # Let's stick to a robust local search: 
    # 1. Find the closest pair (or boundary violation).
    # 2. Move them apart.
    # This is effectively what the previous program did, but let's make it more systematic.
    
    # New Strategy: "Gradient Ascent on the Min Distance"
    # The gradient of the min distance is non-zero only for the pairs achieving the min.
    # We can compute the gradient of the squared distance for the closest pairs.
    
    best_r = _score(centers)
    best_centers = centers.copy()
    current_r = best_r
    
    # We will use a "kick" strategy:
    # 1. Identify the "bottleneck" (closest pair or boundary).
    # 2. Apply a force to alleviate it.
    # 3. Accept if score improves.
    
    # To speed up, we'll use a KDTree for neighbor lookups to find close pairs quickly.
    from scipy.spatial import cKDTree
    
    for it in range(50000):
        if time.time() > deadline:
            break
            
        # Find closest pair
        tree = cKDTree(centers)
        # Query each point for its 2 nearest neighbors (self + 1)
        dists, idxs = tree.query(centers, k=2)
        min_dist = np.min(dists[:, 1])
        
        # Find boundary violation
        boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
        min_boundary = np.min(boundary_margin)
        
        # Determine the limiting radius
        r_candidate = min(min_dist / 2.0, min_boundary)
        
        # If the score has improved, update best
        if r_candidate > current_r:
            current_r = r_candidate
            if r_candidate > best_r:
                best_r = r_candidate
                best_centers = centers.copy()
        
        # If we are stuck (current_r == best_r for a while), we might need to break local minima.
        # But first, let's try to improve the current bottleneck.
        
        # Identify the bottleneck
        if min_dist / 2.0 < min_boundary:
            # Bottleneck is a pair
            min_dist_idx = np.argmin(dists[:, 1])
            i = min_dist_idx
            j = idxs[i, 1]
            
            # Compute direction to move i and j apart
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-10:
                # Random kick
                centers[i] += np.random.normal(0, 0.001, 2)
                centers[j] -= np.random.normal(0, 0.001, 2)
            else:
                dir_vec = diff / norm
                # Move them apart by a small amount
                # We want to increase the distance.
                # Move i by +dir_vec * eps, j by -dir_vec * eps
                eps = 0.001
                centers[i] += dir_vec * eps
                centers[j] -= dir_vec * eps
        else:
            # Bottleneck is boundary
            min_bound_idx = np.argmin(boundary_margin)
            idx = min_bound_idx
            # Move inward
            # Determine which boundary is closest
            # If x > 0, move left, else right. Same for y.
            for k in range(2):
                if centers[idx, k] > 0:
                    centers[idx, k] -= 0.001
                else:
                    centers[idx, k] += 0.001
                    
        # Clamp to square
        centers = np.clip(centers, -0.5, 0.5)
        
        # Random perturbation to escape local minima
        if it % 100 == 0:
            # Perturb a random point
            idx = np.random.randint(N)
            centers[idx] += np.random.normal(0, 0.01, 2)
            centers = np.clip(centers, -0.5, 0.5)
            
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # 1. Initialize with hex grid
    best_centers = _initial_hex_grid(N)
    best_score = _score(best_centers)
    
    # 2. Optimization
    if time.time() < deadline - 10:
        centers, score = _simulated_annealing(best_centers, deadline - 10)
        if score > best_score:
            best_score = score
            best_centers = centers
            
    return best_centers
# EVOLVE-BLOCK-END
