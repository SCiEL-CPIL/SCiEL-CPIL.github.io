import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization: Hexagonal Lattice
    def generate_hex_lattice():
        best_points = None
        best_r = -1
        
        # Spacing s determines radius r ~ s/2
        # Target r ~ 0.053 => s ~ 0.106
        for s in [0.100, 0.105, 0.108, 0.110, 0.112, 0.115, 0.120]:
            # Generate lattice points
            # Hex lattice basis: (s, 0) and (s/2, s*sqrt(3)/2)
            # We need to cover the unit circle.
            # Range: -1.2 to 1.2
            xs = np.arange(-1.2, 1.2, s)
            ys = np.arange(-1.2, 1.2, s * np.sqrt(3) / 2)
            
            points = []
            for i, x in enumerate(xs):
                # Offset every other row
                offset = 0 if i % 2 == 0 else s / 2
                for y in ys:
                    points.append([x + offset, y])
            
            points = np.array(points)
            # Filter points inside unit circle (with margin)
            radii = np.linalg.norm(points, axis=1)
            mask = radii <= 1.0
            points = points[mask]
            
            if len(points) < N:
                continue
            
            # Pick N points closest to origin
            idx = np.argsort(radii)
            points = points[idx[:N]]
            
            # Calculate initial radius
            dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            min_boundary_dist = 1.0 - np.min(np.linalg.norm(points, axis=1))
            r = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if r > best_r:
                best_r = r
                best_points = points.copy()
        
        return best_points, best_r

    points, r_best = generate_hex_lattice()
    
    if points is None:
        # Fallback: random
        points = np.random.uniform(-0.9, 0.9, (N, 2))
        # Ensure valid
        # Just return random if init fails, though it shouldn't
        r_best = 0.0

    # 2. Optimization Loop
    pos = points.copy()
    
    step = 0.005  # Start with smaller step for stability
    min_step = 0.00001
    decay = 0.9995
    
    last_improvement_time = start_time
    no_improvement_count = 0
    
    while time.time() < deadline:
        # Calculate pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # Find closest neighbor for each point
        min_dist, min_idx = np.min(dist, axis=1), np.argmin(dist, axis=1)
        
        # Boundary distances
        boundary_dist = 1.0 - np.linalg.norm(pos, axis=1)
        
        # Determine move direction
        move = np.zeros_like(pos)
        
        # Mask for points constrained by boundary
        is_boundary_constrained = boundary_dist < min_dist
        
        # Points NOT boundary constrained: move away from closest neighbor
        if not np.all(is_boundary_constrained):
            idx_not_boundary = np.where(~is_boundary_constrained)[0]
            neighbor_pos = pos[min_idx[idx_not_boundary]]
            vec = pos[idx_not_boundary] - neighbor_pos
            norm_vec = np.linalg.norm(vec, axis=1, keepdims=True)
            norm_vec[norm_vec == 0] = 1.0
            move[idx_not_boundary] = vec / norm_vec
        
        # Points boundary constrained: move towards origin
        if not np.all(~is_boundary_constrained):
            idx_boundary = np.where(is_boundary_constrained)[0]
            vec = -pos[idx_boundary]
            norm_vec = np.linalg.norm(vec, axis=1, keepdims=True)
            norm_vec[norm_vec == 0] = 1.0
            move[idx_boundary] = vec / norm_vec
        
        # Update positions
        pos += step * move
        
        # Calculate new radius
        dists = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        min_boundary_dist = 1.0 - np.min(np.linalg.norm(pos, axis=1))
        r_curr = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if r_curr > r_best:
            r_best = r_curr
            points = pos.copy()
            last_improvement_time = time.time()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # Decay step size
        step *= decay
        if step < min_step:
            step = min_step
            
        # If stuck, kick
        if no_improvement_count > 3000:
            pos += np.random.normal(0, 0.005, pos.shape)
            step = 0.005
            no_improvement_count = 0
            
    return points
