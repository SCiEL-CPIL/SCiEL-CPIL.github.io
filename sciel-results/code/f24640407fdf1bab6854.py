import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START

def _hex_grid_points(radius, spacing):
    """Generate hexagonal grid points within a circle of given radius."""
    points = []
    k = int(radius / spacing) + 2
    sqrt3 = np.sqrt(3.0)
    for j in range(-k, k + 1):
        y = j * spacing * sqrt3 / 2.0
        x_offset = (j % 2) * spacing / 2.0
        for i in range(-k, k + 1):
            x = (i + x_offset) * spacing
            if np.hypot(x, y) <= radius:
                points.append((x, y))
    return np.array(points)

def _discrete_packing(r_target):
    """
    Attempt to pack N circles of radius r_target using a greedy discrete search on a hex grid.
    Returns None if failed, else array of shape (N, 2).
    """
    if r_target <= 0:
        return None
    
    # Grid spacing should be smaller than 2*r_target to have enough options.
    # Let's use spacing = r_target / 2.0 to be safe.
    spacing = r_target / 2.0
    # Max radius for centers
    max_r = 1.0 - r_target
    
    # Generate candidates
    candidates = _hex_grid_points(max_r, spacing)
    
    if len(candidates) < N:
        return None
    
    # Sort candidates by distance to center (center-first packing)
    dists_to_center = np.hypot(candidates[:, 0], candidates[:, 1])
    sorted_indices = np.argsort(dists_to_center)
    candidates = candidates[sorted_indices]
    
    selected_arr = np.zeros((N, 2))
    count = 0
    min_dist = 2.0 * r_target
    
    for c in candidates:
        if count >= N:
            break
        
        if count == 0:
            selected_arr[0] = c
            count = 1
            continue
            
        # Check if c is far enough from all selected
        diff = selected_arr[:count] - c
        dists = np.sqrt(np.sum(diff**2, axis=1))
        if np.min(dists) >= min_dist:
            selected_arr[count] = c
            count += 1
            
    if count < N:
        return None
        
    return selected_arr

def _optimize_from_init(points, deadline):
    """Run force-directed optimization starting from points."""
    current_r = 0.0
    # Compute initial r
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_vec = np.hypot(points[:, 0], points[:, 1])
    min_boundary_dist = np.min(1.0 - r_vec)
    current_r = min(min_pair_dist, min_boundary_dist) / 2.0
    
    best_r = current_r
    best_points = points.copy()
    
    step_size = 0.01
    cooling = 0.999
    
    for iter in range(3000):
        if time.time() > deadline:
            break
            
        # Compute distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        r_vec = np.hypot(points[:, 0], points[:, 1])
        min_boundary_dist = np.min(1.0 - r_vec)
        
        current_r = min(min_pair_dist, min_boundary_dist) / 2.0
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            
        # Find indices of min pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx[0], min_idx[1]
        
        # Force calculation
        forces = np.zeros_like(points)
        
        # Pair force
        if dists[i, j] > 1e-9:
            v = points[i] - points[j]
            v_norm = np.linalg.norm(v)
            v_unit = v / v_norm
            forces[i] += v_unit * step_size
            forces[j] -= v_unit * step_size
            
        # Boundary forces
        for k in range(N):
            r_k = r_vec[k]
            if r_k > 0:
                dist_to_bound = 1.0 - r_k
                if dist_to_bound < 2 * current_r:
                    v_in = -points[k] / r_k
                    violation = (2 * current_r - dist_to_bound)
                    forces[k] += v_in * step_size * violation * 10
            
        # Update points
        points += forces
        
        # Cooling
        step_size *= cooling
        
        if step_size < 1e-5:
            step_size = 0.005
            points += np.random.normal(0, 0.001, points.shape)
            
    return best_points, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 130.0 # 20s margin
    
    best_overall_points = None
    best_overall_r = 0.0
    
    # 1. Discrete Search for a good starting point
    # Try a radius slightly below the known best to ensure we find a packing
    # Known best ~ 0.05315
    target_r = 0.0525
    
    discrete_points = _discrete_packing(target_r)
    
    if discrete_points is not None:
        # Polish it
        polished_points, polished_r = _optimize_from_init(discrete_points, deadline)
        if polished_r > best_overall_r:
            best_overall_r = polished_r
            best_overall_points = polished_points
    else:
        # Fallback to hex lattice if discrete fails
        s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
        k = int(1.0 / s) + 2
        rows = [((i + 0.5 * (j % 2)) * s, j * s * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - s / 2.0
        points = points[mask]
        if len(points) > N:
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind='stable')
            points = points[indices[:N]]
        elif len(points) < N:
            # Pad
            while len(points) < N:
                 p = np.random.uniform(-0.9, 0.9, 2)
                 if np.hypot(p[0], p[1]) < 0.9:
                     points = np.vstack([points, p])
                     
        if len(points) >= N:
            polished_points, polished_r = _optimize_from_init(points[:N], deadline)
            if polished_r > best_overall_r:
                best_overall_r = polished_r
                best_overall_points = polished_points

    if best_overall_points is None:
        # Last resort: random
        points = np.random.uniform(-0.9, 0.9, (N, 2))
        # Ensure inside
        r_vec = np.hypot(points[:, 0], points[:, 1])
        mask = r_vec > 0.9
        points[mask] *= 0.9 / r_vec[mask, np.newaxis]
        return points

    return best_overall_points

# EVOLVE-BLOCK-END
