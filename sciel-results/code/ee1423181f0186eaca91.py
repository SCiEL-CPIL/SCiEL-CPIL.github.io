import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Initialization: Hexagonal Lattice
    def generate_hex_lattice(d):
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
                if dist + d/2.0 <= 1.0:
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

    # Binary search for optimal lattice spacing to get N points
    d_low = 0.05
    d_high = 0.15
    
    # Ensure d_low is valid
    while True:
        pts = generate_hex_lattice(d_low)
        if len(pts) >= N:
            break
        d_low *= 1.1
        
    # Ensure d_high is invalid
    while True:
        pts = generate_hex_lattice(d_high)
        if len(pts) < N:
            break
        d_high *= 0.9

    for _ in range(30):
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
        radii = np.sqrt(np.random.rand(N))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        points = points * 0.9

    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]

    best_points = points.copy()
    
    def compute_radius(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        # Efficient pairwise distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_pair, min_boundary)

    best_r = compute_radius(best_points)
    
    # 2. Optimization: Force-based relaxation with adaptive target radius
    # We try to increase the radius r_target and relax the system.
    # If the system settles (energy -> 0), the new r_target is feasible.
    
    # Estimate a starting r_target slightly above current best_r
    r_target = best_r * 1.01
    step_size = 0.005
    relax_steps = 500
    no_improve_count = 0
    max_no_improve = 5
    
    # We will run a loop that attempts to increase r_target
    # and performs relaxation.
    
    current_r = best_r
    
    while time.time() < deadline:
        # Try to increase r_target
        # If we recently failed to satisfy constraints for a higher r, we might need to back off
        # For simplicity, we just try a small increment.
        proposed_r = current_r * 1.005
        
        # Relaxation for proposed_r
        pts = best_points.copy()
        
        # Run relaxation
        for step in range(relax_steps):
            if time.time() > deadline:
                break
                
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Pairwise forces
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Mask for overlaps: dists < 2 * proposed_r
            # We want to push apart.
            # Force magnitude proportional to overlap depth.
            # F = k * (2*r - d) * u
            # To avoid division by zero, we use safe division.
            # We only apply force if dists < 2*r.
            
            # Vectorized calculation:
            # overlap = np.maximum(0, 2 * proposed_r - dists)
            # factor = overlap / dists
            # forces += np.sum(diff * factor[:, :, np.newaxis], axis=1)
            # Note: diff[i,j] = pts[i] - pts[j]. Force on i is away from j, so direction is (pts[i]-pts[j])/d.
            # So adding sum(diff * factor) is correct for repulsion.
            
            if proposed_r > 0:
                overlap = np.maximum(0.0, 2.0 * proposed_r - dists)
                # Avoid division by zero. If dists is inf, overlap is 0.
                # If dists is 0, overlap is 2r, factor is inf. But dists=0 only on diagonal (inf) or coincident points.
                # If coincident, we should add a small random jitter.
                # Let's add jitter if any dists < 1e-6 (excluding diagonal)
                coincident = (dists < 1e-6) & (dists != np.inf)
                if np.any(coincident):
                    pts += np.random.normal(0, 1e-4, (N, 2))
                    # Recompute dists for this step? Or just continue? 
                    # Jittering is better. Let's recompute dists for accuracy in force calc.
                    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                    dists = np.sqrt(np.sum(diff**2, axis=2))
                    np.fill_diagonal(dists, np.inf)
                    overlap = np.maximum(0.0, 2.0 * proposed_r - dists)
                
                factor = np.zeros_like(dists)
                mask = overlap > 0
                factor[mask] = overlap[mask] / dists[mask]
                
                forces += np.sum(diff * factor[:, :, np.newaxis], axis=1)
            
            # Boundary forces
            radii = np.hypot(pts[:, 0], pts[:, 1])
            boundary_overlap = np.maximum(0.0, radii + proposed_r - 1.0)
            # Force direction: inward (-c/|c|)
            # Magnitude: boundary_overlap
            # If radius is 0, direction is undefined. Jitter?
            zero_radii = radii < 1e-6
            if np.any(zero_radii):
                pts[zero_radii] += np.random.normal(0, 1e-4, (np.sum(zero_radii), 2))
                radii = np.hypot(pts[:, 0], pts[:, 1])
                boundary_overlap = np.maximum(0.0, radii + proposed_r - 1.0)
            
            # Only apply force if overlap > 0
            b_mask = boundary_overlap > 0
            if np.any(b_mask):
                # Direction: -pts / radii
                # Factor: boundary_overlap / radii
                b_factor = np.zeros_like(radii)
                b_factor[b_mask] = boundary_overlap[b_mask] / radii[b_mask]
                forces -= pts * b_factor[:, np.newaxis]
            
            # Update positions
            pts += step_size * forces
            
            # Safety: Ensure points don't fly away
            # The force model should keep them bounded, but just in case
            max_r = np.max(np.hypot(pts[:, 0], pts[:, 1]))
            if max_r > 2.0:
                pts = pts / max_r * 1.0
        
        # Check if proposed_r is feasible
        final_r = compute_radius(pts)
        
        if final_r >= proposed_r * 0.999: # Allow small numerical error
            # Feasible! Update best
            if final_r > best_r:
                best_r = final_r
                best_points = pts.copy()
                current_r = proposed_r
                no_improve_count = 0
                # We can try to increase step_size slightly if converging fast?
                # Or keep it stable.
        else:
            # Not feasible. Try a smaller increment or back off?
            # For now, just don't update best_r.
            # We might get stuck if we always propose 1.005x.
            # Let's try a smaller increment if we fail.
            # Or just keep trying? If we fail, the relaxation might have found a better config for a smaller r.
            # Let's check if the current pts is better for the *current* best_r.
            r_for_current = compute_radius(pts)
            if r_for_current > best_r:
                best_r = r_for_current
                best_points = pts.copy()
            
            # Reduce step size to settle down?
            step_size *= 0.9
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Perturb to escape local minimum
                perturb_scale = 0.002
                pts = best_points + np.random.normal(0, perturb_scale, (N, 2))
                # Ensure validity of perturbation
                radii = np.hypot(pts[:, 0], pts[:, 1])
                outside = radii > 1.0
                if np.any(outside):
                    pts[outside] = pts[outside] / radii[outside, np.newaxis] * 0.999
                best_points = pts
                no_improve_count = 0
                step_size = 0.005 # Reset step size
                current_r = best_r # Reset target based on new best

    # Final polish
    # Do a few more iterations with very small step size to squeeze out max r
    # We can try to maximize r directly by local search on the best configuration
    # But the force relaxation is already doing that.
    
    # Ensure validity
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_points)):
        scale = 0.99 / np.max(radii) if np.max(radii) > 0 else 1.0
        best_points = best_points * scale
        
    return best_points

# EVOLVE-BLOCK-END
