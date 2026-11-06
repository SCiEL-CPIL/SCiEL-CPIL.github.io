import numpy as np
import time
from scipy.spatial import cKDTree

N = 200
DEADLINE = 110.0  # Safety margin before 120s limit

def _compute_radius(points):
    """Compute the maximum common radius for given circle centers."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points are inside the square
    if np.any(points[:, 0] < -0.5) or np.any(points[:, 0] > 0.5):
        return 0.0
    if np.any(points[:, 1] < -0.5) or np.any(points[:, 1] > 0.5):
        return 0.0
    
    # Center-center distances
    tree = cKDTree(points)
    pairs = tree.query_pairs(1.0)  # Only need pairs within distance 1.0
    if len(pairs) == 0:
        min_center_dist = 1.0
    else:
        min_center_dist = float('inf')
        for i, j in pairs:
            d = np.linalg.norm(points[i] - points[j])
            if d < min_center_dist:
                min_center_dist = d
    
    # Wall distances
    dx = 0.5 - np.abs(points[:, 0])
    dy = 0.5 - np.abs(points[:, 1])
    min_wall_dist = np.min(np.concatenate([dx, dy]))
    
    r = min(min_center_dist, min_wall_dist) / 2.0
    return r

def _generate_hex_grid(n):
    """Generate a hexagonal grid of points inside the unit square."""
    # Optimal hex packing: spacing s in x, s*sqrt(3)/2 in y
    # For N points in a 1x1 square, area per point = 1/N
    # Hex cell area = s^2 * sqrt(3)/2
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    cols = int(np.ceil(1.0 / s))
    rows = int(np.ceil(1.0 / (s * np.sqrt(3) / 2)))
    
    points = []
    for r in range(rows):
        for c in range(cols):
            x = -0.5 + (c + 0.5) * s
            y = -0.5 + (r + 0.5) * s * np.sqrt(3) / 2
            if r % 2 != 0:
                x += s / 2.0
            points.append([x, y])
    
    points = np.array(points)
    
    # Filter points inside the square
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) > N:
        # Select N points with maximum minimum distance (greedy)
        selected = [0]
        remaining = list(range(1, len(points)))
        while len(selected) < N and remaining:
            best_idx = -1
            best_min_dist = -1
            for idx in remaining:
                min_d = min(np.linalg.norm(points[idx] - points[s]) for s in selected)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break
        points = points[selected]
    elif len(points) < N:
        missing = N - len(points)
        random_pts = np.random.uniform(-0.49, 0.49, (missing, 2))
        points = np.vstack([points, random_pts])
        
    return points

def _push_apart(points, r_target, n_iters=200, start_time=None, deadline=None):
    """Push overlapping circles apart to achieve target radius r_target."""
    if start_time is None:
        start_time = time.time()
    if deadline is None:
        deadline = DEADLINE
    
    points = points.copy()
    step_size = 0.01
    decay = 0.995
    
    for it in range(n_iters):
        if time.time() > deadline - 5:
            break
        
        # Build KD-tree for efficient neighbor queries
        tree = cKDTree(points)
        # Query neighbors within 2*r_target + small margin
        query_r = 2 * r_target + 0.01
        
        forces = np.zeros_like(points)
        pairs = tree.query_pairs(query_r)
        
        for i, j in pairs:
            d = points[i] - points[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                # Random perturbation for coincident points
                d = np.random.randn(2) * 1e-6
                dist = np.linalg.norm(d)
            
            if dist < 2 * r_target:
                # Overlapping: push apart
                overlap = 2 * r_target - dist
                direction = d / dist
                force = overlap * direction
                forces[i] += force
                forces[j] -= force
        
        # Wall forces: push points away from walls if too close
        for i in range(N):
            # Left wall
            if points[i, 0] < r_target:
                forces[i, 0] += (r_target - points[i, 0])
            # Right wall
            if points[i, 0] > 0.5 - r_target:
                forces[i, 0] -= (points[i, 0] - (0.5 - r_target))
            # Bottom wall
            if points[i, 1] < r_target:
                forces[i, 1] += (r_target - points[i, 1])
            # Top wall
            if points[i, 1] > 0.5 - r_target:
                forces[i, 1] -= (points[i, 1] - (0.5 - r_target))
        
        # Apply forces with step size
        points += step_size * forces
        
        # Clip to stay inside the square (with margin for radius)
        points[:, 0] = np.clip(points[:, 0], -0.5 + r_target * 0.5, 0.5 - r_target * 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5 + r_target * 0.5, 0.5 - r_target * 0.5)
        
        step_size *= decay
    
    return points

def _polish(points, r_target, n_iters=500, start_time=None, deadline=None):
    """Fine-tune positions to maximize the minimum distance at fixed radius."""
    if start_time is None:
        start_time = time.time()
    if deadline is None:
        deadline = DEADLINE
    
    points = points.copy()
    step_size = 0.005
    decay = 0.998
    
    for it in range(n_iters):
        if time.time() > deadline - 5:
            break
        
        tree = cKDTree(points)
        query_r = 2 * r_target + 0.005
        
        forces = np.zeros_like(points)
        pairs = tree.query_pairs(query_r)
        
        for i, j in pairs:
            d = points[i] - points[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.random.randn(2) * 1e-7
                dist = np.linalg.norm(d)
            
            if dist < 2 * r_target:
                overlap = 2 * r_target - dist
                direction = d / dist
                force = overlap * direction
                forces[i] += force
                forces[j] -= force
        
        # Wall forces
        for i in range(N):
            if points[i, 0] < r_target:
                forces[i, 0] += (r_target - points[i, 0])
            if points[i, 0] > 0.5 - r_target:
                forces[i, 0] -= (points[i, 0] - (0.5 - r_target))
            if points[i, 1] < r_target:
                forces[i, 1] += (r_target - points[i, 1])
            if points[i, 1] > 0.5 - r_target:
                forces[i, 1] -= (points[i, 1] - (0.5 - r_target))
        
        points += step_size * forces
        points[:, 0] = np.clip(points[:, 0], -0.5 + r_target * 0.1, 0.5 - r_target * 0.1)
        points[:, 1] = np.clip(points[:, 1], -0.5 + r_target * 0.1, 0.5 - r_target * 0.1)
        
        step_size *= decay
    
    return points

def construct():
    start_time = time.time()
    
    # Generate initial hex grid
    init_points = _generate_hex_grid(N)
    init_points = np.clip(init_points, -0.49, 0.49)
    
    best_points = init_points
    best_score = _compute_radius(init_points)
    
    # Multi-scale push-apart: gradually increase target radius
    # Start from a conservative radius and increase
    # Known best is ~0.0366, so we aim for that range
    
    # Estimate initial radius from grid
    tree = cKDTree(init_points)
    pairs = tree.query_pairs(1.0)
    if len(pairs) > 0:
        min_d = min(np.linalg.norm(init_points[i] - init_points[j]) for i, j in pairs)
    else:
        min_d = 0.1
    wall_d = min(0.5 - np.abs(init_points[:, 0]).min(), 0.5 - np.abs(init_points[:, 1]).min())
    r_start = min(min_d, wall_d) / 2.0 * 0.5  # Conservative start
    
    # Gradually increase radius
    r_targets = []
    r = r_start
    while r < 0.037:
        r_targets.append(r)
        r *= 1.05  # 5% increase per step
    r_targets.append(0.0366)  # Aim for known best
    r_targets.append(0.0367)  # Slightly above
    
    for r_target in r_targets:
        if time.time() > DEADLINE - 30:
            break
        
        # Push apart to achieve this radius
        points = _push_apart(best_points, r_target, n_iters=300, 
                            start_time=start_time, deadline=DEADLINE)
        
        # Check if we achieved the target radius
        actual_r = _compute_radius(points)
        
        if actual_r >= r_target * 0.95:  # Close enough to target
            # Polish to extract precision
            points = _polish(points, actual_r, n_iters=200,
                           start_time=start_time, deadline=DEADLINE)
            actual_r = _compute_radius(points)
        
        if actual_r > best_score:
            best_score = actual_r
            best_points = points.copy()
    
    # Final polish at the best radius found
    if time.time() < DEADLINE - 10:
        final_points = _polish(best_points, best_score, n_iters=500,
                              start_time=start_time, deadline=DEADLINE)
        final_score = _compute_radius(final_points)
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
    
    # Ensure all points are strictly inside
    best_points[:, 0] = np.clip(best_points[:, 0], -0.4999, 0.4999)
    best_points[:, 1] = np.clip(best_points[:, 1], -0.4999, 0.4999)
    
    return best_points
