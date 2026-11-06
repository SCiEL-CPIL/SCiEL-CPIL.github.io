"""Combined circle packing for N=300 in unit circle using hex lattice init and force-based relaxation."""
import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    # Safety margin: return before 150s wall clock limit
    deadline = start_time + 145.0 

    # --- Initialization: Hexagonal Lattice via Binary Search ---
    
    def generate_hex_lattice(d):
        """
        Generate a hexagonal lattice with spacing d.
        Returns points that are strictly inside the unit circle 
        (distance from origin + d/2 <= 1).
        """
        if d <= 0:
            return np.array([])
        
        # Estimate range of indices needed
        # Max extent is approx 1.0 / d
        k = int(1.0 / d) + 2
        points = []
        sqrt3 = np.sqrt(3.0)
        
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * sqrt3 / 2.0
                dist = np.hypot(x, y)
                # Ensure the circle of radius d/2 centered at (x,y) is inside unit circle
                if dist + d/2.0 <= 1.0:
                    points.append([x, y])
        
        if len(points) == 0:
            return np.array([])
            
        points = np.array(points)
        if len(points) >= N:
            # If we have more than N points, select the N closest to the origin
            # to keep the packing dense and centered.
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind="stable")[:N]
            return points[indices]
        else:
            return points

    # Binary search for the largest d (smallest spacing is wrong, we want max r -> min d? 
    # Wait. r = d/2. Maximize r -> Maximize d.
    # But we need to fit N points. 
    # If d is too large, we fit fewer points.
    # If d is small, we fit many points.
    # We want the LARGEST d such that we can still fit at least N points.
    # This gives the largest initial radius r = d/2.
    
    d_low = 0.05  # Small spacing, many points
    d_high = 0.15 # Large spacing, few points
    
    # Ensure d_low is valid (enough points)
    while True:
        pts = generate_hex_lattice(d_low)
        if len(pts) >= N:
            break
        d_low *= 1.1
        if d_low > 0.5: break # Fallback if something goes wrong

    # Ensure d_high is invalid (not enough points)
    while True:
        pts = generate_hex_lattice(d_high)
        if len(pts) < N:
            break
        d_high *= 0.9
        if d_high < 0.01: break

    for _ in range(40):
        d_mid = (d_low + d_high) / 2.0
        pts = generate_hex_lattice(d_mid)
        if len(pts) >= N:
            d_low = d_mid
        else:
            d_high = d_mid
            
    points = generate_hex_lattice(d_low)
    
    if len(points) < N:
        # Fallback: Random uniform in disk
        angles = np.random.rand(N) * 2 * np.pi
        radii = np.sqrt(np.random.rand(N)) * 0.9
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]

    # Ensure points is exactly N x 2
    if points.shape[0] != N:
        # Pad or trim if necessary (should not happen with fallback)
        if points.shape[0] < N:
            pad = np.random.uniform(-0.9, 0.9, (N - points.shape[0], 2))
            points = np.vstack([points, pad])
        else:
            points = points[:N]

    # --- Optimization: Force-based Relaxation ---
    
    def compute_radius(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(radii >= 1.0):
            return 0.0
        min_boundary = np.min(1.0 - radii)
        
        # Efficient pairwise distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_pair, min_boundary)

    best_points = points.copy()
    best_r = compute_radius(best_points)
    
    # Optimization parameters
    step_size = 0.005
    no_improve_count = 0
    max_no_improve = 50
    total_iterations = 0
    max_iterations = 200000 # Hard cap, but time is the real limit

    for it in range(max_iterations):
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
                # Scale perturbation by current step size or a fixed small value
                perturb_scale = 0.005
                points += np.random.normal(0, perturb_scale, (N, 2))
                
                # Ensure validity
                radii = np.hypot(points[:, 0], points[:, 1])
                outside = radii > 1.0
                if np.any(outside):
                    points[outside] = points[outside] / radii[outside, np.newaxis] * 0.999
                
                no_improve_count = 0
                # Reduce step size after perturbation to settle down
                step_size *= 0.8
                # Also decay step size over time to converge
                step_size = max(step_size, 0.0005)

        # Compute Forces
        # Target radius: slightly larger than best_r to push for improvement?
        # Using best_r is safer. Using best_r * 1.001 might help escape plateaus.
        r_target = best_r * 1.001
        
        forces = np.zeros_like(points)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        
        # Boundary forces
        # If 1 - |c| < r_target, push in.
        # Violation = r_target - (1 - |c|) = r_target + |c| - 1
        # If violation > 0, force = -k * violation * c/|c|
        boundary_violation = np.maximum(0.0, r_target + radii - 1.0)
        # Force direction is inward: -c/|c|
        # Avoid division by zero
        radii_safe = np.where(radii > 1e-10, radii, 1.0)
        boundary_forces = (-points / radii_safe[:, np.newaxis]) * boundary_violation[:, np.newaxis]
        forces += boundary_forces
        
        # Pairwise forces
        # If d < 2*r_target, push apart.
        # Violation = 2*r_target - d
        # Force magnitude proportional to violation / d^2?
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        if r_target > 1e-6:
            d_limit = 2.0 * r_target
            # Compute factor: max(0, 1 - d/d_limit)
            # If d >= d_limit, force is 0.
            # If d < d_limit, force is positive.
            # We want to push apart, so force on i is in direction (p_i - p_j) = diff[i,j]
            # Wait, diff[i,j] = p_i - p_j.
            # Force on i should be in direction of diff[i,j] if they are too close.
            
            # Calculate violation factor
            # factor = max(0, 1 - dists / d_limit)
            factor = np.maximum(0.0, 1.0 - dists / d_limit)
            
            # Scale by 1/d^2 to make it strong at close range
            # Avoid division by zero (diagonal is inf, so 1/inf = 0, which is fine)
            # But for very small d, 1/d^2 is huge.
            # Let's cap the force magnitude or use a softer potential.
            # V = 1/d^2 is standard.
            
            dists_safe = np.where(dists > 1e-10, dists, 1.0)
            force_mag = factor / (dists_safe ** 2)
            
            # Force vector = force_mag * direction
            # direction = diff / dists
            # So force = force_mag * diff / dists = factor / dists^3 * diff
            
            # Let's use: force_ij = factor * diff / dists^3
            # This is equivalent to gradient of (factor * 1/d) ? No.
            # Let's just use: force_ij = (1/d^2) * (1 - d/d_limit) * (diff/d)
            # = (1/d^3) * (1 - d/d_limit) * diff
            
            force_vectors = (force_mag / dists_safe) * diff
            # Sum forces from all other points
            pairwise_forces = np.sum(force_vectors, axis=1)
            forces += pairwise_forces
        
        # Update positions
        points += forces * step_size
        
        # Project inside unit circle
        new_radii = np.hypot(points[:, 0], points[:, 1])
        outside = new_radii > 1.0
        if np.any(outside):
            points[outside] = points[outside] / new_radii[outside, np.newaxis] * 0.9999
        
        # Decay step size slowly
        if it % 100 == 0:
            step_size *= 0.999
            step_size = max(step_size, 0.0005)

    # Final validation and cleanup
    final_r = compute_radius(best_points)
    if final_r > best_r:
        best_r = final_r
        best_points = best_points.copy() # Just in case

    # Ensure validity
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_points)):
        # Scale down to fit
        if np.max(radii) > 0:
            scale = 0.999 / np.max(radii)
            best_points = best_points * scale
    
    return best_points
