"""Pack 300 equal circles in a unit disk using force-based relaxation."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(points):
    """Compute the minimum radius for a set of points."""
    n = len(points)
    if n == 0:
        return 0.0
    # Boundary constraint
    dists = np.sqrt(np.sum(points**2, axis=1))
    boundary_r = np.min(1.0 - dists)
    if boundary_r < 0:
        return 0.0
    # Pairwise distances
    # For N=300, this is 300*300 = 90000 pairs, manageable
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    pair_r = min_pair_dist / 2.0
    return min(boundary_r, pair_r)

def _hexagonal_init(n):
    """Initialize with hexagonal lattice pattern."""
    # Start with a reasonable spacing
    # For N=300 in unit disk, density ~ N*pi*r^2 / pi = N*r^2
    # r ~ sqrt(1/N) ~ 0.058
    # Let's try different spacings
    best_points = None
    best_score = 0
    
    for spacing in [0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14]:
        points = _hex_lattice(spacing)
        if len(points) >= n:
            # Keep the n closest to center
            dists = np.sqrt(np.sum(points**2, axis=1))
            idx = np.argsort(dists)[:n]
            points = points[idx]
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    if best_points is not None:
        return best_points
    
    # Fallback: random
    return np.random.randn(n, 2) * 0.8

def _hex_lattice(spacing):
    """Generate hexagonal lattice points within unit disk."""
    sqrt3 = np.sqrt(3.0)
    points = []
    # Grid range
    k = int(2.0 / spacing) + 3
    for j in range(-k, k + 1):
        y = j * spacing * sqrt3 / 2.0
        if abs(y) > 1.0:
            continue
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            if x*x + y*y <= 1.0:
                points.append([x, y])
    return np.array(points)

def _force_relaxation(points, max_iter=5000, deadline=None):
    """Iteratively push overlapping circles apart and boundary-violating circles inward."""
    points = points.copy().astype(np.float64)
    n = len(points)
    
    # Adaptive step size
    step_size = 0.01
    min_step = 1e-8
    cooling = 0.9995
    
    best_score = _compute_score(points)
    best_points = points.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion for overlapping or close circles
        # Only consider pairs within a cutoff distance for efficiency
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Find close pairs (within some threshold)
        # Use the current minimum distance as a guide
        current_min = np.min(dist_matrix)
        threshold = max(2.5 * current_min, 0.2)
        
        mask = dist_matrix < threshold
        np.fill_diagonal(mask, False)
        
        # For each close pair, compute repulsion force
        for i in range(n):
            for j in np.where(mask[i])[0]:
                if i < j:
                    d = dist_matrix[i, j]
                    if d > 1e-10:
                        # Repulsion: stronger when closer
                        # Force magnitude proportional to 1/d^2 or similar
                        force_mag = 1.0 / (d * d + 1e-6)
                        direction = (points[i] - points[j]) / d
                        forces[i] += force_mag * direction
                        forces[j] -= force_mag * direction
        
        # Boundary constraint: push inward if too close to boundary
        dists = np.sqrt(np.sum(points**2, axis=1))
        boundary_margin = 0.05  # Target margin from boundary
        for i in range(n):
            if dists[i] > 1.0 - boundary_margin:
                # Push inward
                if dists[i] > 1e-10:
                    direction = -points[i] / dists[i]
                    excess = dists[i] - (1.0 - boundary_margin)
                    forces[i] += 2.0 * excess * direction
        
        # Normalize forces to prevent large jumps
        force_mags = np.sqrt(np.sum(forces**2, axis=1))
        max_force = np.max(force_mags) if len(force_mags) > 0 else 0
        if max_force > 1.0:
            forces /= max_force
        
        # Update positions
        points += step_size * forces
        
        # Keep points within the disk (hard constraint)
        dists = np.sqrt(np.sum(points**2, axis=1))
        for i in range(n):
            if dists[i] > 1.0:
                points[i] *= (1.0 / dists[i]) * 0.999
        
        # Compute score
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Cool down
        step_size *= cooling
        if step_size < min_step:
            step_size = min_step
    
    return best_points, best_score

def _local_optimization(points, max_iter=1000, deadline=None):
    """Local optimization using gradient-like approach."""
    points = points.copy().astype(np.float64)
    n = len(points)
    
    # Use a simpler iterative improvement
    best_score = _compute_score(points)
    best_points = points.copy()
    
    lr = 0.001
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        
        # Compute score and identify constraints
        dists = np.sqrt(np.sum(points**2, axis=1))
        boundary_r = np.min(1.0 - dists)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        pair_r = min_pair / 2.0
        
        score = min(boundary_r, pair_r)
        
        # Find which constraint is active
        if boundary_r < pair_r:
            # Boundary is limiting: move points inward
            for i in range(n):
                if dists[i] > 0.5:  # Only move outer points
                    if dists[i] > 1e-10:
                        direction = -points[i] / dists[i]
                        points[i] += lr * direction
        else:
            # Pairwise is limiting: push apart
            # Find the minimum pair
            i_min, j_min = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            d = dist_matrix[i_min, j_min]
            if d > 1e-10:
                direction = (points[i_min] - points[j_min]) / d
                points[i_min] += lr * direction
                points[j_min] -= lr * direction
            # Also push other close pairs
            close_mask = dist_matrix < 1.2 * min_pair
            np.fill_diagonal(close_mask, False)
            for i in range(n):
                for j in np.where(close_mask[i])[0]:
                    if i < j:
                        d = dist_matrix[i, j]
                        if d > 1e-10:
                            direction = (points[i] - points[j]) / d
                            points[i] += lr * 0.5 * direction
                            points[j] -= lr * 0.5 * direction
        
        # Keep within disk
        dists = np.sqrt(np.sum(points**2, axis=1))
        for i in range(n):
            if dists[i] > 1.0:
                points[i] *= (1.0 / dists[i]) * 0.999
        
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Decay learning rate
        lr *= 0.999
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Step 1: Initialize with hexagonal lattice
    points = _hexagonal_init(N)
    score = _compute_score(points)
    
    # Step 2: Force-based relaxation
    points, score = _force_relaxation(points, max_iter=3000, deadline=deadline)
    
    # Step 3: Local optimization polish
    if time.time() < deadline - 10.0:
        points, score = _local_optimization(points, max_iter=500, deadline=deadline)
    
    # Final check
    final_score = _compute_score(points)
    if final_score > score:
        score = final_score
    
    return points
# EVOLVE-BLOCK-END
