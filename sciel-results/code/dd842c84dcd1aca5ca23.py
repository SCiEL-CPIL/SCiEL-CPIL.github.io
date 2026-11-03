import numpy as np
import time

N = 50

def _poisson_disk_init(n, min_dist, bounds):
    points = []
    # Add first point
    p = np.random.uniform(bounds[0], bounds[1], 3)
    points.append(p)
    
    attempts = 0
    max_attempts = 10000
    while len(points) < n and attempts < max_attempts:
        p = np.random.uniform(bounds[0], bounds[1], 3)
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        if valid:
            points.append(p)
        attempts += 1
        
    if len(points) < n:
        # Fallback to random
        while len(points) < n:
            points.append(np.random.uniform(bounds[0], bounds[1], 3))
            
    return np.array(points)

def _optimize(points, time_limit):
    start_time = time.time()
    step_size = 0.05
    cooling_rate = 0.9995
    
    best_radius = 0.0
    best_points = points.copy()
    
    while time.time() < start_time + time_limit:
        # Calculate forces
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(dist_sq)
        
        eps = 1e-9
        dist[dist < eps] = eps
        dist_sq[dist_sq < eps*eps] = eps*eps
        
        inv_dist_cubed = 1.0 / (dist_sq * dist)
        np.fill_diagonal(inv_dist_cubed, 0.0)
        
        forces = np.sum(diff * inv_dist_cubed[:, :, np.newaxis], axis=1)
        
        # Update
        points += step_size * forces
        points = np.clip(points, -0.5, 0.5)
        
        # Score
        min_pair_dist = np.min(dist[~np.eye(N, dtype=bool)])
        min_wall_dist = 0.5 - np.max(np.abs(points))
        current_radius = min(min_pair_dist / 2.0, min_wall_dist)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        step_size *= cooling_rate
        
        if step_size < 1e-6:
            break
            
    return best_points, best_radius

def construct():
    global_start = time.time()
    total_deadline = global_start + 110.0
    
    best_points = None
    best_radius = 0.0
    
    num_restarts = 5
    
    for i in range(num_restarts):
        remaining_time = total_deadline - time.time()
        if remaining_time < 10:
            break
            
        time_per_restart = remaining_time / (num_restarts - i)
        
        np.random.seed(i)
        
        # Initialize with Poisson Disk to ensure good spread
        # Min dist 0.1 is safe for 50 spheres in unit cube (max radius ~0.13)
        points = _poisson_disk_init(N, 0.1, (-0.45, 0.45))
        
        pts, rad = _optimize(points, time_per_restart)
        
        if rad > best_radius:
            best_radius = rad
            best_points = pts
            
    return best_points
