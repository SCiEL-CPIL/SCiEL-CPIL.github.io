import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Initialization: Hexagonal Lattice
    # Generate a hexagonal lattice of points with spacing d.
    # We want to find a configuration where N points fit inside the unit circle.
    # The radius of the circles in this lattice is r = d/2.
    # We want to maximize r, so minimize d.
    # However, the lattice points must be strictly inside (dist + r <= 1).
    # So we binary search for the smallest d such that we have at least N points.
    
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

    # Binary search for optimal lattice spacing
    d_low = 0.05  # Should yield many points
    d_high = 0.15 # Should yield few points
    
    # Ensure d_low is valid (enough points)
    while True:
        pts = generate_hex_lattice(d_low)
        if len(pts) >= N:
            break
        d_low *= 1.1
        
    # Ensure d_high is invalid (not enough points)
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
    
    # 2. Optimization: Force-based relaxation with Soft-Min Surrogate hybrid?
    # The current program uses Soft-Min. The inspiration uses Force-based.
    # Let's stick to Force-based as it was more effective (score 1.004 vs 0.5629).
    # But we can improve the force model.
    # Instead of just pushing apart when d < 2*r_target, we can use a continuous force.
    # Force = k * (1/d^2) * direction, but clipped or softened to avoid explosion.
    # Also, we should use the current best_r as the target, but maybe push towards
    # a slightly larger target to escape local minima?
    
    # Let's implement a robust force-based relaxation.
    # We will use an adaptive step size.
    # We will also use a "cooling" schedule for the step size.
    
    step_size = 0.005
    no_improve_count = 0
    max_no_improve = 100
    
    # To speed up convergence, we can use a smaller step size initially?
    # Or start with a larger one?
    # Let's start with 0.005 and decay.
    
    for it in range(200000):
        if time.time() > deadline:
            break
        
        # Compute current radius
        current_r = compute_radius(points)
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            no_improve_count = 0
            # If we improved, maybe increase step size slightly?
            # step_size = min(step_size * 1.05, 0.02)
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturb to escape local minimum
                # Use a small random perturbation
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
        
        # Compute Forces
        # Target radius: best_r * 1.001 (slightly larger to push)
        # Actually, using best_r directly is safer.
        # Let's use r_target = best_r
        r_target = best_r
        
        forces = np.zeros_like(points)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        
        # Boundary forces
        # If 1 - |c| < r_target, push in.
        # Violation = r_target - (1 - |c|) = r_target + |c| - 1
        # If violation > 0, force = -k * violation * c/|c|
        
        # Pairwise forces
        # If d < 2*r_target, push apart.
        # Violation = 2*r_target - d
        # Force = k * violation / d^2 * direction?
        # Or just constant force if violated?
        # The inspiration used 1/d^2 which is strong.
        # Let's use a softer potential: V = 1/d^2. Force = dV/dd * dir = -2/d^3 * dir.
        # But we only want to push if d < 2*r_target.
        
        # Vectorized pairwise force calculation
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Mask for active pairs: dists < 2 * r_target
        # We want to push apart if dists < 2 * r_target.
        # Let's define a force magnitude.
        # F = k * (1 - dists / (2*r_target)) / dists^2 ?
        # This goes to 0 at the boundary and is strong at small distances.
        
        # To avoid division by zero, we can use np.where or safe division.
        # Since diagonal is inf, 1/inf = 0.
        
        # Let's try: F_mag = 1.0 / dists**2
        # But only apply if dists < 2*r_target.
        # And scale by how much it violates?
        # If we just use 1/d^2, it's always pushing apart.
        # This is fine, it will settle into a minimum energy state.
        # The "target" is implicitly the equilibrium where forces balance.
        # But we want to maximize the MINIMUM distance.
        # So we should only push if d < 2*r_current?
        # If we push all pairs, we might spread them out too much, reducing boundary clearance.
        # The boundary force will counteract that.
        
        # Let's use a force that is zero if d > 2*r_target, and strong if d < 2*r_target.
        # F_ij = k * max(0, 1 - d_ij / (2*r_target)) / d_ij^2
        
        # This requires element-wise max.
        # d_min_allowed = 2 * r_target
        # If r_target is 0, this breaks. But r_target should be > 0.
        
        if r_target > 1e-6:
            d_limit = 2.0 * r_target
            # Compute factor: max(0, 1 - d/d_limit)
            # If d >= d_limit, factor is 0.
            # If d < d_limit, factor is (d_limit - d) / d_limit.
            
            # We want to avoid division by zero in 1/d^2.
            # We can use np.where(dists < d_limit, 1.0, 0.0) * (1.0 / dists**2)
            # But 1/inf is 0, so it's safe for diagonal.
            # For small d, 1/d^2 is huge.
            
            # Let's cap the force magnitude to prevent instability.
            # Max force per pair = 10.0?
            
            # factor = np.where(dists < d_limit, (d_limit - dists) / d_limit, 0.0)
            # force_mag = factor * (1.0 / (dists**2 + 1e-6))
            # force_mag = np.minimum(force_mag, 10.0)
            
            # This is getting complicated. Let's stick to the simpler 1/d^2 but masked.
            # F = k / d^2 if d < d_limit else 0.
            
            mask = dists < d_limit
            # force_mag = np.zeros_like(dists)
            # force_mag[mask] = 1.0 / (dists[mask]**2)
            # force_mag = np.minimum(force_mag, 5.0) # Cap it
            
            # Actually, let's just use the gradient of the soft-min from the current program?
            # No, the current program scored low.
            # Let's use the force-based approach from inspiration 1 but with a cap.
            
            # force_mag = np.where(mask, 1.0 / (dists**2 + 1e-6), 0.0)
            # force_mag = np.minimum(force_mag, 2.0)
            
            # Sum forces
            # F_i = sum_j F_ij * (p_i - p_j) / d_ij
            # = sum_j (force_mag_ij / d_ij) * (p_i - p_j)
            
            # We have diff = p_i - p_j.
            # We need to multiply diff by force_mag / dists.
            # weight = force_mag / dists
            # weight = np.where(mask, 1.0 / (dists**3 + 1e-6), 0.0)
            # weight = np.minimum(weight, 10.0)
            
            # Let's try this:
            weight = np.zeros_like(dists)
            # Only compute for active pairs to save time?
            # N=300, 300x300 = 90k operations. Fast enough.
            
            # Use safe division
            dists_safe = np.where(dists > 1e-6, dists, 1.0)
            
            # Calculate weight for active pairs
            # w = (1/d^3) if d < d_limit else 0
            # This corresponds to force F ~ 1/d^2, direction ~ 1/d, so total ~ 1/d^3?
            # F_vec = (1/d^2) * (diff/d) = diff / d^3.
            
            w = np.where(mask, 1.0 / (dists_safe**3), 0.0)
            # Cap the weight
            w = np.minimum(w, 50.0)
            
            # F_i = sum_j w_ij * diff_ij
            forces += np.sum(w[:, :, np.newaxis] * diff, axis=1)
            
            # Boundary Force
            # Push in if 1 - |c| < r_target
            # Violation = r_target - (1 - |c|)
            # Force = -k * violation * c/|c|
            # Let's use a similar scaling.
            # If 1 - |c| < r_target, push in.
            # Strength proportional to violation?
            # Or constant?
            # Let's use: if 1 - |c| < r_target, F = -k * (r_target - (1-|c|)) * c/|c|
            
            dist_to_boundary = 1.0 - radii
            violation = r_target - dist_to_boundary
            # If violation > 0, push in.
            # Force magnitude = k * violation
            # Direction = -c/|c|
            
            # k_boundary = 5.0?
            # Let's tune this.
            # The pairwise force is strong (1/d^3).
            # Boundary force should be comparable.
            
            # Only apply if violation > 0
            # And avoid division by zero if radii is 0.
            radii_safe = np.where(radii > 1e-6, radii, 1.0)
            unit_vec = points / radii_safe[:, np.newaxis]
            
            # Force vector = -k * violation * unit_vec
            # k = 5.0
            k_bound = 5.0
            bound_force = np.zeros_like(points)
            active_bound = violation > 0
            bound_force[active_bound] = -k_bound * violation[active_bound, np.newaxis] * unit_vec[active_bound]
            
            forces += bound_force
        
        # Update
        points += forces * step_size
        
        # Enforce hard constraint: inside unit circle
        new_radii = np.hypot(points[:, 0], points[:, 1])
        outside = new_radii > 1.0
        if np.any(outside):
            points[outside] = points[outside] / new_radii[outside, np.newaxis] * 0.999
            
        # Decay step size slowly
        step_size *= 0.999
        
        # Optional: If best_r is very high, reduce search scope
        if best_r > 0.0535:
             break

    return best_points
