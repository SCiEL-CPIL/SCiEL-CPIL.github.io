import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Initialization: Hexagonal Lattice
    # We want to pack N circles.
    # Let's generate a grid of points and select the N closest to the center.
    
    def generate_hex_lattice(r):
        d = 2.0 * r
        if d <= 0:
            return np.array([])
        
        k = int(1.0 / d) + 2
        points = []
        sqrt3 = np.sqrt(3.0)
        
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * sqrt3 / 2.0
                dist = np.hypot(x, y)
                if dist + r <= 1.0:
                    points.append([x, y])
        
        if len(points) == 0:
            return np.array([])
            
        points = np.array(points)
        if len(points) >= N:
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind="stable")[:N]
            return points[indices]
        else:
            return points

    # Find a good initial radius
    r_low = 0.01
    r_high = 0.08
    
    # Ensure r_low gives enough points
    while True:
        pts = generate_hex_lattice(r_low)
        if len(pts) >= N:
            break
        r_low *= 0.8
        
    # Binary search for max r in hex lattice
    for _ in range(20):
        r_mid = (r_low + r_high) / 2.0
        pts = generate_hex_lattice(r_mid)
        if len(pts) >= N:
            r_low = r_mid
        else:
            r_high = r_mid
            
    points = generate_hex_lattice(r_low)
    
    if len(points) < N:
        # Fallback random
        angles = np.random.rand(N) * 2 * np.pi
        radii = np.sqrt(np.random.rand(N))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        points = points * 0.5 # Scale down

    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]

    best_points = points.copy()
    best_r = 0.0
    
    def compute_radius(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        # Efficient pairwise distance
        # Use broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_pair, min_boundary)

    best_r = compute_radius(best_points)
    
    # Optimization Loop
    # Strategy: Force-based relaxation with adaptive step size
    # We try to push points apart to increase the minimum distance.
    
    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 50
    
    for it in range(100000):
        if time.time() > deadline:
            break
            
        # Compute current radius
        current_r = compute_radius(points)
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturb to escape local minimum
                perturb_scale = 0.01
                points += np.random.normal(0, perturb_scale, (N, 2))
                # Ensure validity
                radii = np.hypot(points[:, 0], points[:, 1])
                outside = radii > 1.0
                if np.any(outside):
                    points[outside] = points[outside] / radii[outside, np.newaxis]
                
                no_improve_count = 0
                step_size *= 0.9 # Reduce step size after perturbation
        
        # Compute Forces
        # We want to maximize the minimum distance.
        # Force from pairs: If d < 2 * r_target, push apart.
        # Force from boundary: If 1 - |c| < r_target, push in.
        
        # Let's use r_target = best_r * 1.01
        r_target = best_r * 1.01
        
        forces = np.zeros_like(points)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        
        # Vectorized pairwise force calculation
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Mask for active pairs (where distance is less than 2 * r_target)
        mask = dists < 2.0 * r_target
        
        # Compute force magnitudes: 1/d^2 for active pairs, 0 otherwise
        # Avoid division by zero by using np.where
        safe_dists = np.where(mask, np.maximum(dists, 1e-6), 1.0)
        force_mags = np.where(mask, 1.0 / (safe_dists**2), 0.0)
        
        # Scale down
        force_mags *= 0.005
        
        # Compute forces: sum over j of (p_i - p_j) * force_mag[i,j]
        # diff[i,j] = p_i - p_j, so we multiply diff by force_mags and sum over j
        forces += np.sum(diff * force_mags[:, :, np.newaxis], axis=1)
        
        # Boundary forces (vectorized)
        # If a point is too close to the boundary, push it towards the center
        bound_violation = np.maximum(0, radii - (1.0 - r_target))
        # Force direction: -points / radii (towards center)
        # Avoid division by zero
        safe_radii = np.maximum(radii, 1e-6)
        boundary_forces = -(points / safe_radii[:, np.newaxis]) * bound_violation[:, np.newaxis] * 5.0
        forces += boundary_forces
        
        # Update
        points += forces * step_size
        
        # Enforce hard constraint: inside unit circle
        new_radii = np.hypot(points[:, 0], points[:, 1])
        outside = new_radii > 1.0
        if np.any(outside):
            points[outside] = points[outside] / new_radii[outside, np.newaxis]
            
        # Decay step size slowly
        step_size *= 0.9995
        
        # Optional: If best_r is very high, reduce search scope
        if best_r > 0.0535:
             break

    # Final polish pass with tighter tolerances
    if time.time() < deadline - 10.0:
        polish_deadline = deadline - 5.0
        points = best_points.copy()
        polish_r = best_r
        polish_step = 0.001
        polish_no_improve = 0
        polish_max_no_improve = 30
        
        for it in range(50000):
            if time.time() > polish_deadline:
                break
            
            current_r = compute_radius(points)
            if current_r > polish_r:
                polish_r = current_r
                best_r = polish_r
                best_points = points.copy()
                polish_no_improve = 0
            else:
                polish_no_improve += 1
                if polish_no_improve > polish_max_no_improve:
                    break
            
            r_target = polish_r * 1.005
            forces = np.zeros_like(points)
            
            radii = np.hypot(points[:, 0], points[:, 1])
            
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            mask = dists < 2.0 * r_target
            safe_dists = np.where(mask, np.maximum(dists, 1e-6), 1.0)
            force_mags = np.where(mask, 1.0 / (safe_dists**2), 0.0)
            force_mags *= 0.002
            
            forces += np.sum(diff * force_mags[:, :, np.newaxis], axis=1)
            
            bound_violation = np.maximum(0, radii - (1.0 - r_target))
            safe_radii = np.maximum(radii, 1e-6)
            boundary_forces = -(points / safe_radii[:, np.newaxis]) * bound_violation[:, np.newaxis] * 5.0
            forces += boundary_forces
            
            points += forces * polish_step
            
            new_radii = np.hypot(points[:, 0], points[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / new_radii[outside, np.newaxis]
            
            polish_step *= 0.999

    return best_points
