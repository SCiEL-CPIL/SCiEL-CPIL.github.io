import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Initialize with hexagonal lattice
    # Approximate spacing for N circles in unit circle
    # Area of unit circle = pi. Area of N circles = N * pi * r^2.
    # Max r approx sqrt(1/N). For N=300, r approx 0.0577.
    # Spacing approx 2r.
    
    # Generate points on hex grid
    points = []
    # We need to find a good spacing.
    # Let's estimate spacing s.
    # Number of points in hex grid of radius R in lattice units is approx 3R^2.
    # We want approx N points. R approx sqrt(N/3).
    # Physical radius 1.
    # Let's just generate a large grid and filter.
    
    s = 0.1 # Initial guess for spacing (2r)
    # Adjust s to get approx N points
    # Area of hex cell is s^2 * sqrt(3)/2.
    # Total area pi.
    # N * s^2 * sqrt(3)/2 approx pi => s approx sqrt(2*pi/(N*sqrt(3)))
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    # Generate grid
    k = int(1.0 / s) + 2
    rows = [((i + 0.5 * (j % 2)) * s, j * s * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    
    # Filter to unit circle
    radius = np.hypot(points[:, 0], points[:, 1])
    # We need to keep points that can fit with radius r = s/2.
    # So |c| + s/2 <= 1 => |c| <= 1 - s/2
    mask = radius <= 1.0 - s / 2.0
    points = points[mask]
    
    # If we have more than N, pick the ones closest to center (usually best for symmetry)
    if len(points) > N:
        # Sort by distance to center
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind='stable')
        points = points[indices[:N]]
    elif len(points) < N:
        # If too few, reduce spacing and try again
        s *= 0.9
        k = int(1.0 / s) + 2
        rows = [((i + 0.5 * (j % 2)) * s, j * s * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - s / 2.0
        points = points[mask]
        if len(points) > N:
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind='stable')
            points = points[indices[:N]]
        # If still not enough, just pad with random points? Unlikely for N=300.
        if len(points) < N:
             # Fallback: just take what we have and add random ones? 
             # Better to just return what we have if it's close, but task requires N.
             # For N=300, s ~ 0.05, grid will be dense enough.
             pass

    # Ensure exactly N points
    if len(points) < N:
         # This shouldn't happen with the grid above for N=300, but just in case
         # Add random points
         while len(points) < N:
             p = np.random.uniform(-0.9, 0.9, 2)
             if np.hypot(p[0], p[1]) < 0.9:
                 points = np.vstack([points, p])
    elif len(points) > N:
         points = points[:N]

    # 2. Optimization
    # We want to maximize the minimum distance.
    # We will use a simple local search: 
    # Iterate, find the pair with minimum distance, push them apart.
    # Also handle boundary.
    
    # Initial radius estimate
    # Compute current min dist
    def compute_min_dist(pts):
        # Use scipy distance matrix if available? No, let's do it manually or with numpy broadcasting
        # N=300, 300x300 matrix is 90k elements. Very fast.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        # Boundary distance
        r_vec = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary_dist = np.min(1.0 - r_vec)
        return min_pair_dist, min_boundary_dist, dists, r_vec

    min_pair_dist, min_boundary_dist, dists, r_vec = compute_min_dist(points)
    current_r = min(min_pair_dist, min_boundary_dist) / 2.0
    
    best_r = current_r
    best_points = points.copy()
    
    step_size = 0.01
    cooling = 0.999
    
    # Optimization Loop
    # We try to increase the minimum distance.
    # Heuristic: 
    # 1. Find the closest pair (i, j).
    # 2. Push i and j apart along the vector connecting them.
    # 3. If a point is too close to boundary, push it inward.
    
    for iter in range(5000):
        if time.time() > deadline:
            break
            
        # Compute distances
        # To save time, we can reuse dists if we only move a few points, but moving all is easier to code.
        # N=300, computing 300x300 dist matrix is ~1ms in numpy.
        # 5000 iterations * 1ms = 5s. Safe.
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        r_vec = np.hypot(points[:, 0], points[:, 1])
        min_boundary_dist = np.min(1.0 - r_vec)
        
        current_r = min(min_pair_dist, min_boundary_dist) / 2.0
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            
        # Find indices of min pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx[0], min_idx[1]
        
        # Force calculation
        forces = np.zeros_like(points)
        
        # Pair force
        if dists[i, j] > 1e-9:
            v = points[i] - points[j]
            v_norm = np.linalg.norm(v)
            v_unit = v / v_norm
            # Push apart
            f = step_size * (1.0 - dists[i, j] / (2 * current_r + 1e-9)) # Scale force
            # Actually, simpler: just push by a small amount in opposite directions
            # We want to increase the distance.
            # Move i by +v_unit * step, j by -v_unit * step
            forces[i] += v_unit * step_size
            forces[j] -= v_unit * step_size
            
        # Boundary forces
        # For each point, if 1 - |c| < 2*current_r, push inward
        # We can do this for all points to smooth out the boundary
        for k in range(N):
            r_k = r_vec[k]
            if r_k > 0:
                # Distance to boundary
                dist_to_bound = 1.0 - r_k
                if dist_to_bound < 2 * current_r:
                    # Push inward
                    v_in = -points[k] / r_k
                    # Strength proportional to violation
                    violation = (2 * current_r - dist_to_bound)
                    forces[k] += v_in * step_size * violation * 10 # Tuning factor
            
        # Update points
        points += forces
        
        # Project to unit circle if they go outside? 
        # If a point goes outside, it's invalid. 
        # We should clip or reflect.
        # Simple clip: if |c| > 1, set |c| = 1? 
        # But this might cause overlaps.
        # Better: if |c| > 1 - 2*current_r, it's a constraint violation.
        # The force above handles this.
        # But if step_size is too large, we might jump out.
        # Let's ensure points stay within a safe radius, say 0.99
        # Actually, the boundary force should keep them in.
        
        # Cooling
        step_size *= cooling
        
        # If step size is too small, we might be stuck.
        # We can restart with a slightly larger step or random perturbation.
        if step_size < 1e-5:
            step_size = 0.005
            # Add small random noise
            points += np.random.normal(0, 0.001, points.shape)
            
    return best_points

# EVOLVE-BLOCK-END
