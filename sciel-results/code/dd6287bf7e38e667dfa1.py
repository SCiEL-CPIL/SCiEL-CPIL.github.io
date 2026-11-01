import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the centers of 200 equal circles in a unit square [-0.5, 0.5]^2
    to maximize the common radius.
    Uses a physics-based relaxation method (soft repulsion) to optimize the
    minimum distance between circles and between circles and boundaries.
    """
    # Deadline guard: 150s limit, keep 5s margin
    deadline = time.time() + 145.0
    
    # Helper to compute the score (radius) for a given configuration
    def compute_radius(points):
        # Min distance between pairs
        # Using scipy for speed if available, else numpy broadcasting (slower but robust)
        # Since N=200, O(N^2) is 40k ops, very fast in numpy.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Min distance to boundary
        # Points are in [-0.5, 0.5]
        # dist to left: x - (-0.5) = x + 0.5
        # dist to right: 0.5 - x
        # dist to bottom: y + 0.5
        # dist to top: 0.5 - y
        min_boundary_dist = np.min([
            np.min(points[:, 0]) + 0.5,
            0.5 - np.max(points[:, 0]),
            np.min(points[:, 1]) + 0.5,
            0.5 - np.max(points[:, 1])
        ])
        
        # Radius is half of the minimum clearance
        return 0.5 * min(min_pair_dist, min_boundary_dist)

    def optimize(points, max_time):
        """
        Iteratively improve the configuration using gradient ascent on a smoothed
        objective function (sum of inverse distances or similar) or simple force-directed
        relaxation.
        """
        current_r = compute_radius(points)
        best_points = points.copy()
        best_r = current_r
        
        # Initialize velocity
        vel = np.zeros_like(points)
        
        # Learning rate schedule
        # Start with a larger step, decay over time
        # We want to maximize min distance.
        # A common trick: maximize sum of 1/d^2 or similar to push all pairs apart.
        # Or simply: move each point in the direction of the sum of repulsive forces
        # from its closest neighbors and away from boundaries.
        
        t = 0
        total_steps = 20000 # Arbitrary cap, time is the real limit
        step_size = 0.01
        
        while t < total_steps and time.time() < max_time:
            # Compute forces
            forces = np.zeros_like(points)
            
            # Pairwise repulsion
            # To avoid O(N^2) bottleneck in pure python loops, use numpy.
            # But we need to be careful with the "min" logic.
            # Standard MD uses 1/r^2. This optimizes the global packing but
            # might not maximize the *minimum* distance directly.
            # However, it usually leads to good local optima.
            
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            
            # Avoid division by zero
            mask = dists > 1e-10
            inv_dists = np.zeros_like(dists)
            inv_dists[mask] = 1.0 / dists[mask]
            
            # Force is proportional to 1/d^2 in direction of separation
            # F_ij = (p_i - p_j) / |p_i - p_j|^2
            # Sum over j for each i
            
            # Using broadcasting:
            # force_i = sum_j (p_i - p_j) / |p_i - p_j|^2
            # (p_i - p_j) is diff[i,j]
            # We need to multiply diff by 1/d^2
            # inv_dists_sq = inv_dists ** 2
            # But we only want to push apart if they are close? 
            # Actually, for packing, we want to maximize the minimum distance.
            # A "soft" potential like 1/d^2 works well.
            
            # Note: diff[i,j] is p_i - p_j. 
            # Force on i due to j is along p_i - p_j.
            # So we sum (diff[i,j] * inv_dists[i,j]**2) over j.
            
            forces += np.sum(diff * (inv_dists ** 2)[:, :, np.newaxis], axis=1)
            
            # Boundary repulsion
            # Push points away from edges if they are too close.
            # We want to maximize distance to boundary.
            # Let d_left = x + 0.5. If d_left < r_target, push right.
            # Since we don't know r_target exactly, we can use a soft wall.
            # Force from left wall: if x < -0.5 + epsilon, push right.
            # Better: Use the same 1/d^2 logic for boundaries.
            # Distance to left: dl = x + 0.5
            # Distance to right: dr = 0.5 - x
            # Distance to bottom: db = y + 0.5
            # Distance to top: dt = 0.5 - y
            
            dl = points[:, 0] + 0.5
            dr = 0.5 - points[:, 0]
            db = points[:, 1] + 0.5
            dt = 0.5 - points[:, 1]
            
            # Only apply force if distance is small (e.g., < 0.1) to avoid
            # weird behavior when points are in the center.
            # Actually, for packing, we want to push them out to the boundaries
            # only if they are constrained by other circles.
            # A simple approach: Add a small constant repulsion from boundaries
            # that increases as they get closer.
            
            # Let's use a simple spring to the boundary if they cross, 
            # or a soft repulsion.
            # To maximize radius, points should be as far from boundary as possible
            # IF the internal spacing is the bottleneck.
            # If boundary is the bottleneck, they should be closer to boundary?
            # No, the radius is min(internal_spacing/2, boundary_dist).
            # So we want to maximize the minimum of these two.
            
            # The force-directed approach naturally balances this if we treat
            # the boundaries as "virtual circles" of infinite radius? No.
            # Let's just add a small repulsive force from boundaries to prevent
            # sticking, and rely on the pairwise forces to spread them out.
            
            # A better strategy for boundary:
            # If a point is closer to the boundary than to any other point,
            # the boundary is the bottleneck for that point.
            # We want to increase that distance.
            
            # Let's stick to simple pairwise repulsion + hard constraint projection.
            # If a point goes outside, project it back.
            # This ensures validity. The optimization then maximizes internal distances.
            
            # Update positions
            points += forces * step_size
            points += vel
            vel = vel * 0.9 + forces * step_size * 0.1 # Simple momentum
            
            # Project back to square [-0.5, 0.5]
            # This is a hard constraint.
            # Note: This projection might reduce the radius if points were pushed
            # out, but it ensures validity.
            points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
            points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
            
            # Compute new radius
            new_r = compute_radius(points)
            
            if new_r > best_r:
                best_r = new_r
                best_points = points.copy()
            
            # Decay step size
            step_size *= 0.9995
            
            # If step size is too small, break or restart?
            if step_size < 1e-6:
                break
                
            t += 1
            
            # Check time every 100 steps
            if t % 100 == 0 and time.time() > max_time:
                break
                
        return best_points, best_r

    # Run multiple random restarts to find a good local optimum
    best_global_r = 0
    best_global_points = None
    
    np.random.seed(42) # Reproducibility for debugging, but maybe random is better?
    # Actually, for a fixed score, deterministic is safer.
    # But we want to beat the seed. The seed is a grid.
    # Random restarts might find a better non-grid packing.
    
    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() > deadline - 10: # Reserve time for last optimization
            break
            
        # Random initialization inside the square
        # Add a small margin to ensure they start inside
        points = np.random.uniform(-0.48, 0.48, size=(N, 2))
        
        # Run optimization
        opt_points, opt_r = optimize(points, deadline - 10)
        
        if opt_r > best_global_r:
            best_global_r = opt_r
            best_global_points = opt_points
    
    # If no optimization was run (time out immediately), return a grid
    if best_global_points is None:
        # Fallback to a simple grid
        k = 15
        row_lengths = [13] * 15 # 15*13=195, need 5 more
        # Just make a grid
        xs = np.linspace(-0.49, 0.49, 14)
        ys = np.linspace(-0.49, 0.49, 15)
        grid = np.array([[x, y] for y in ys for x in xs])
        # Trim to N
        return grid[:N]
        
    return best_global_points

# EVOLVE-BLOCK-END
