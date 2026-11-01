"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initialize with concentric rings
    def ring_init():
        points = np.zeros((N, 2))
        # Center
        points[0] = [0, 0]
        remaining = N - 1
        ring_idx = 1
        idx = 1
        while remaining > 0 and ring_idx < 10:
            # Number of points on this ring
            if ring_idx == 1:
                n_ring = 6
            else:
                n_ring = 6 * ring_idx
            n_ring = min(n_ring, remaining)
            if n_ring == 0:
                break
            angles = np.linspace(0, 2*np.pi, n_ring, endpoint=False)
            # Radius for this ring - approximate
            r_ring = ring_idx * 0.15
            points[idx:idx+n_ring, 0] = r_ring * np.cos(angles)
            points[idx:idx+n_ring, 1] = r_ring * np.sin(angles)
            idx += n_ring
            remaining -= n_ring
            ring_idx += 1
        return points
    
    def hex_init():
        """Hexagonal lattice initialization"""
        points_list = []
        s = 0.22
        for j in range(-5, 6):
            for i in range(-5, 6):
                x = i * s + (j % 2) * s * 0.5
                y = j * s * np.sqrt(3) / 2
                if x*x + y*y <= 0.85**2:
                    points_list.append([x, y])
        points = np.array(points_list[:N])
        if len(points) < N:
            # Fill remaining with ring init
            ring_pts = ring_init()
            points = np.vstack([points, ring_pts[len(points):]])
        return points
    
    # Try multiple initializations and pick the best
    best_points = None
    best_score = -1
    
    inits = [ring_init(), hex_init()]
    
    # Add some random perturbations to ring init
    for trial in range(3):
        rng = np.random.RandomState(42 + trial)
        pts = ring_init().copy()
        pts += rng.randn(N, 2) * 0.05
        inits.append(pts)
    
    for init_pts in inits:
        # Quick evaluation
        pts = init_pts.copy()
        # Ensure inside
        dists = np.hypot(pts[:, 0], pts[:, 1])
        mask = dists < 0.98
        if mask.sum() < N:
            continue
        # Score
        diff = pts[:, None, :] - pts[None, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        min_pair = np.min(dist_mat)
        min_boundary = 1.0 - np.max(dists)
        score = min(min_pair / 2, min_boundary)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    if best_points is None:
        best_points = ring_init()
        best_score = 0.1
    
    # Optimization: gradient-free force-directed approach
    points = best_points.copy()
    best_r = best_score
    
    # Parameters
    lr = 0.001
    momentum = 0.9
    vel = np.zeros_like(points)
    
    while time.time() < deadline - 5.0:
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion (soft)
        diff = points[:, None, :] - points[None, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, 1e10)
        
        # Only consider pairs that are too close
        threshold = best_r * 2.0 + 0.05
        close_mask = dist_mat < threshold
        np.fill_diagonal(close_mask, False)
        
        for i in range(N):
            for j in range(i+1, N):
                if close_mask[i, j]:
                    d = dist_mat[i, j]
                    if d < 1e-10:
                        # Random push
                        angle = np.random.uniform(0, 2*np.pi)
                        fx = np.cos(angle) * 0.1
                        fy = np.sin(angle) * 0.1
                    else:
                        fx = (points[i, 0] - points[j, 0]) / d
                        fy = (points[i, 1] - points[j, 1]) / d
                    # Force magnitude increases as they get closer
                    f_mag = (threshold - d) * 0.5
                    forces[i, 0] += fx * f_mag
                    forces[i, 1] += fy * f_mag
                    forces[j, 0] -= fx * f_mag
                    forces[j, 1] -= fy * f_mag
        
        # Boundary constraint: push inward if too close to edge
        dists = np.hypot(points[:, 0], points[:, 1])
        for i in range(N):
            if dists[i] > 1.0 - best_r - 0.02:
                if dists[i] > 1e-10:
                    forces[i, 0] -= points[i, 0] / dists[i] * 0.3
                    forces[i, 1] -= points[i, 1] / dists[i] * 0.3
        
        # Update with momentum
        vel = momentum * vel + lr * forces
        points += vel
        
        # Clamp to inside
        dists = np.hypot(points[:, 0], points[:, 1])
        for i in range(N):
            if dists[i] > 1.0 - best_r * 0.5:
                scale = (1.0 - best_r * 0.5) / dists[i]
                points[i] *= scale
        
        # Evaluate
        diff = points[:, None, :] - points[None, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        min_pair = np.min(dist_mat)
        dists = np.hypot(points[:, 0], points[:, 1])
        min_boundary = 1.0 - np.max(dists)
        current_r = min(min_pair / 2, min_boundary)
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
        
        # Adaptive learning rate
        if current_r > best_score * 1.001:
            lr = max(lr * 0.99, 0.0001)
        else:
            lr = min(lr * 1.01, 0.005)
    
    # Final polish: local refinement
    if time.time() < deadline - 2.0:
        points = best_points.copy()
        lr = 0.0001
        vel = np.zeros_like(points)
        momentum = 0.8
        
        for _ in range(200):
            if time.time() > deadline - 2.0:
                break
            forces = np.zeros_like(points)
            diff = points[:, None, :] - points[None, :, :]
            dist_mat = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist_mat, 1e10)
            
            for i in range(N):
                for j in range(i+1, N):
                    d = dist_mat[i, j]
                    if d < 1e-10:
                        angle = np.random.uniform(0, 2*np.pi)
                        fx = np.cos(angle) * 0.01
                        fy = np.sin(angle) * 0.01
                    else:
                        fx = (points[i, 0] - points[j, 0]) / d
                        fy = (points[i, 1] - points[j, 1]) / d
                    # Equalize distances
                    avg_d = best_r * 2.0
                    f_mag = (avg_d - d) * 0.1
                    forces[i, 0] += fx * f_mag
                    forces[i, 1] += fy * f_mag
                    forces[j, 0] -= fx * f_mag
                    forces[j, 1] -= fy * f_mag
            
            dists = np.hypot(points[:, 0], points[:, 1])
            for i in range(N):
                if dists[i] > 1.0 - best_r:
                    if dists[i] > 1e-10:
                        forces[i, 0] -= points[i, 0] / dists[i] * 0.2
                        forces[i, 1] -= points[i, 1] / dists[i] * 0.2
            
            vel = momentum * vel + lr * forces
            points += vel
            
            dists = np.hypot(points[:, 0], points[:, 1])
            for i in range(N):
                if dists[i] > 1.0 - best_r * 0.5:
                    scale = (1.0 - best_r * 0.5) / dists[i]
                    points[i] *= scale
            
            diff = points[:, None, :] - points[None, :, :]
            dist_mat = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist_mat, np.inf)
            min_pair = np.min(dist_mat)
            dists = np.hypot(points[:, 0], points[:, 1])
            min_boundary = 1.0 - np.max(dists)
            current_r = min(min_pair / 2, min_boundary)
            
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
    
    # Final validation
    points = best_points
    dists = np.hypot(points[:, 0], points[:, 1])
    # Ensure all inside
    for i in range(N):
        if dists[i] >= 1.0:
            if dists[i] > 1e-10:
                points[i] *= (1.0 - 1e-6) / dists[i]
            else:
                points[i] = [0, 0]
    
    return points
# EVOLVE-BLOCK-END
