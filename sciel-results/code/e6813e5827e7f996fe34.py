"""Improved packing: ring-based initialization + force-directed relaxation."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 120s limit, 2s safety margin

    def generate_ring_initialization():
        """Generate points on concentric rings with hexagonal-like spacing."""
        # Target spacing based on area argument
        # For N points in unit disk, typical spacing ~ sqrt(pi/N)
        s_target = np.sqrt(np.pi / N)
        
        points = [(0.0, 0.0)]  # Center point
        
        # Generate rings outward
        r = s_target
        ring_idx = 0
        while r < 1.0 and len(points) < N + 50:
            # Number of points on this ring: approx 2*pi*r / s
            n_pts = max(1, int(2 * np.pi * r / s_target))
            # Adjust to be a multiple that gives good coverage
            angle_step = 2 * np.pi / n_pts
            # Offset alternate rings for hexagonal packing
            angle_offset = (ring_idx % 2) * (np.pi / n_pts)
            
            for k in range(n_pts):
                angle = angle_offset + k * angle_step
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                if np.hypot(x, y) <= 0.99:  # Keep well inside
                    points.append((x, y))
            
            r += s_target * 0.866  # sqrt(3)/2 for hexagonal
            ring_idx += 1
        
        points = np.array(points)
        
        # If we have more than N, randomly select N
        if len(points) > N:
            indices = np.random.choice(len(points), N, replace=False)
            points = points[indices]
        elif len(points) < N:
            # Pad with random points
            pad = np.random.uniform(-0.8, 0.8, (N - len(points), 2))
            points = np.vstack([points, pad])
        
        return points

    def calc_min_radius(pts):
        """Compute the maximum common radius for given centers."""
        if len(pts) < 2:
            return 0.0
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d)
        min_bound = 1.0 - np.max(np.linalg.norm(pts, axis=1))
        return min(min_pair / 2.0, min_bound)

    # Initialize with ring-based layout
    points = generate_ring_initialization()
    r_current = calc_min_radius(points)
    
    # Force-directed relaxation to maximize minimum radius
    # Use a smooth surrogate: minimize sum of soft violations
    r_target = r_current * 1.01
    
    lr = 0.008
    momentum = 0.9
    v = np.zeros_like(points)
    best_score = r_current
    best_points = points.copy()
    
    # Main optimization loop
    while time.time() < deadline - 5.0:
        # Compute pairwise distances efficiently
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        
        # Pairwise repulsion for close pairs
        mask = dists < 2.0 * r_target
        dists_safe = np.where(dists > 1e-8, dists, 1.0)
        viol = np.where(mask, 2.0 * r_target - dists, 0.0)
        weights = viol / dists_safe
        weights[~mask] = 0.0
        np.fill_diagonal(weights, 0.0)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary constraint
        radii = np.linalg.norm(points, axis=1)
        b_mask = radii > 1.0 - r_target
        b_viol = np.where(b_mask, radii - (1.0 - r_target), 0.0)
        b_forces = -b_viol[:, np.newaxis] * (points / np.maximum(radii, 1e-8)[:, np.newaxis])
        
        total_forces = forces + b_forces
        
        # Update with momentum
        v = momentum * v + lr * total_forces
        points += v
        
        # Enforce boundary constraint
        new_radii = np.linalg.norm(points, axis=1)
        max_r = 1.0 - r_target
        outside = new_radii > max_r
        if np.any(outside):
            points[outside] *= (max_r / new_radii[outside])[:, np.newaxis]
            v[outside] = 0.0
        
        # Periodically evaluate and adjust target
        if int((time.time() - start_time) * 10) % 10 == 0:  # ~every 0.1s
            current_score = calc_min_radius(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                r_current = current_score
                r_target = r_current * 1.003
                lr = max(0.001, lr * 0.995)
            else:
                r_target = r_current * 0.998
                if r_target < best_score * 0.97:
                    points = best_points.copy()
                    v = np.zeros_like(points)
                    r_target = best_score * 1.001
                    lr = 0.003

    # Final polish with small steps
    if time.time() < deadline - 1.0:
        points = best_points.copy()
        r_target = best_score * 1.0001
        lr = 0.0003
        v = np.zeros_like(points)
        
        for _ in range(100):
            if time.time() > deadline - 0.5:
                break
            dists = cdist(points, points)
            np.fill_diagonal(dists, np.inf)
            mask = dists < 2.0 * r_target
            dists_safe = np.where(dists > 1e-8, dists, 1.0)
            viol = np.where(mask, 2.0 * r_target - dists, 0.0)
            weights = viol / dists_safe
            weights[~mask] = 0.0
            np.fill_diagonal(weights, 0.0)
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
            
            radii = np.linalg.norm(points, axis=1)
            b_mask = radii > 1.0 - r_target
            b_viol = np.where(b_mask, radii - (1.0 - r_target), 0.0)
            b_forces = -b_viol[:, np.newaxis] * (points / np.maximum(radii, 1e-8)[:, np.newaxis])
            
            total_forces = forces + b_forces
            v = momentum * v + lr * total_forces
            points += v
            
            new_radii = np.linalg.norm(points, axis=1)
            max_r = 1.0 - r_target
            outside = new_radii > max_r
            if np.any(outside):
                points[outside] *= (max_r / new_radii[outside])[:, np.newaxis]
                v[outside] = 0.0
        
        final_score = calc_min_radius(points)
        if final_score > best_score:
            best_points = points

    return best_points
# EVOLVE-BLOCK-END
