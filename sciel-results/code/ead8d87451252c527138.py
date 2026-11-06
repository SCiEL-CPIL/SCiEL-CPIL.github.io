import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    # 1. Initialization: Hexagonal Lattice
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
        
    # Add small random jitter to break symmetry and avoid degenerate states
    points += np.random.normal(0, 0.001, (N, 2))
    
    best_points = points.copy()
    best_r = 0.0
    
    def compute_radius(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        # Efficient pairwise distance using chunked computation
        n = len(pts)
        min_pair_sq = np.inf
        chunk_size = 50
        for i in range(0, n, chunk_size):
            j = min(i + chunk_size, n)
            diff = pts[i:j, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            if i > 0:
                # Only consider pairs where both indices are in range
                # For chunk starting at i, we want pairs (a,b) where i<=a<j and a<b
                # But we need all pairs, so let's just compute all and take min
                pass
            # Set diagonal to inf to exclude self-pairs
            if i == 0:
                np.fill_diagonal(dists_sq, np.inf)
            else:
                # For subsequent chunks, the diagonal in this sub-matrix corresponds to
                # actual diagonal entries only if the chunk aligns with the global diagonal
                # This is tricky - let's use a different approach
                pass
            
            # For simplicity, let's compute full pairwise for each chunk row
            # Actually, let's just do it properly with a mask
            row_indices = np.arange(i, j)
            col_indices = np.arange(n)
            # We want to exclude pairs where row == col
            mask = (row_indices[:, np.newaxis] != col_indices[np.newaxis, :])
            dists_sq[~mask] = np.inf
            min_pair_sq = min(min_pair_sq, np.min(dists_sq))
        
        min_pair = np.sqrt(min_pair_sq) / 2.0
        
        return min(min_pair, min_boundary)

    def compute_radius_fast(pts):
        """Faster radius computation using only boundary and approximate pair check"""
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        if min_boundary <= 0:
            return 0.0
        # Use a quick check: if boundary is the limiting factor, return it
        # Otherwise, we need pair distances
        # For speed, sample a subset of pairs
        n = len(pts)
        if n <= 100:
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, np.inf)
            min_pair_sq = np.min(dists_sq)
            min_pair = np.sqrt(min_pair_sq) / 2.0
            return min(min_pair, min_boundary)
        else:
            # Sample pairs for speed
            np.random.seed(42)
            sample_size = 5000
            idx = np.random.randint(0, n, (sample_size, 2))
            a, b = idx[:, 0], idx[:, 1]
            valid = a != b
            a, b = a[valid], b[valid]
            if len(a) == 0:
                return min_boundary
            diffs = pts[a] - pts[b]
            dists_sq = np.sum(diffs**2, axis=1)
            min_pair_sq = np.min(dists_sq)
            min_pair = np.sqrt(min_pair_sq) / 2.0
            return min(min_pair, min_boundary)

    best_r = compute_radius(best_points)
    
    # Optimization Loop: Vectorized Gradient Descent on Potential Energy
    # Objective: Minimize E = sum_{i<j} 1/d_ij^2 + lambda * sum_i 1/(1-|c_i|)^2
    # This pushes points apart and away from the boundary.
    
    step_size = 0.001
    no_improve_count = 0
    max_no_improve = 50
    lambda_bound = 5.0
    compute_radius_every = 10  # Compute full radius every 10 iterations
    
    for it in range(200000):
        if time.time() > deadline:
            break
            
        # Compute current radius periodically
        if it % compute_radius_every == 0:
            current_r = compute_radius(points)
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Targeted perturbation: move the most constrained points
                    radii = np.hypot(points[:, 0], points[:, 1])
                    # Find points closest to boundary or with close neighbors
                    boundary_proximity = 1.0 - radii
                    # Perturb points with small boundary proximity
                    close_to_boundary = boundary_proximity < np.percentile(boundary_proximity, 20)
                    points[close_to_boundary] += np.random.normal(0, 0.005, (np.sum(close_to_boundary), 2))
                    # Also perturb some random points
                    random_idx = np.random.choice(N, size=10, replace=False)
                    points[random_idx] += np.random.normal(0, 0.003, (10, 2))
                    
                    # Ensure validity
                    new_radii = np.hypot(points[:, 0], points[:, 1])
                    outside = new_radii > 1.0
                    if np.any(outside):
                        points[outside] = points[outside] / new_radii[outside, np.newaxis]
                    
                    no_improve_count = 0
                    step_size *= 0.8

        # Compute Forces (Gradients) - only every 2 iterations for speed
        if it % 2 == 0:
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, 1.0) # Avoid div by zero
            
            # Use soft-min approximation: focus force on closest pairs
            # inv_dists_sq = 1.0 / dists_sq
            # For better gradient, use inverse distance (not squared) for smoother behavior
            dists = np.sqrt(dists_sq)
            np.fill_diagonal(dists, 1.0)
            
            # Force from pairs: gradient of 1/d is -1/d^2 * (p_i - p_j)/d = -(p_i - p_j)/d^3
            # We want to push apart, so force on i is sum_j (p_i - p_j)/d^3 * strength
            # Use a softened version to avoid singularities
            eps = 0.01
            dists_safe = np.maximum(dists, eps)
            factors = 1.0 / (dists_safe ** 3)
            mask = np.ones((N, N), dtype=bool)
            np.fill_diagonal(mask, False)
            factors[~mask] = 0.0
            
            # F_i = sum_j factors[i,j] * diff[i,j]
            pair_forces = np.einsum('ij,ijk->ik', factors, diff)
            
            # Boundary Force: gradient of 1/(1-|p|)^2 is 2*p/(|p|*(1-|p|)^3)
            # We want to push away from boundary, so force = -lambda * gradient
            radii = np.hypot(points[:, 0], points[:, 1])
            radii_safe = np.where(radii > 1e-6, radii, 1.0)
            radii_clipped = np.minimum(radii_safe, 1.0 - 1e-4)
            denom = radii_clipped * (1.0 - radii_clipped)**3
            denom = np.where(denom > 1e-8, denom, 1.0)
            
            bound_forces = -lambda_bound * points / denom[:, np.newaxis]
            
            total_forces = pair_forces + bound_forces
            
            # Update
            points += total_forces * step_size
            
            # Enforce hard constraint: inside unit circle
            new_radii = np.hypot(points[:, 0], points[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / new_radii[outside, np.newaxis]
            
            # Decay step size slowly
            step_size *= 0.99995
        
        # Optional: If best_r is very high, reduce search scope
        if best_r > 0.0534:
             break

    # Final polish phase: use smaller steps and more frequent radius computation
    if time.time() < deadline - 5.0:
        polish_deadline = deadline - 2.0
        polish_step = step_size * 0.3
        for it in range(50000):
            if time.time() > polish_deadline:
                break
            
            current_r = compute_radius(points)
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
            
            # Compute forces
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, 1.0)
            dists = np.sqrt(dists_sq)
            np.fill_diagonal(dists, 1.0)
            dists_safe = np.maximum(dists, 0.005)
            factors = 1.0 / (dists_safe ** 3)
            mask = np.ones((N, N), dtype=bool)
            np.fill_diagonal(mask, False)
            factors[~mask] = 0.0
            pair_forces = np.einsum('ij,ijk->ik', factors, diff)
            
            radii = np.hypot(points[:, 0], points[:, 1])
            radii_safe = np.where(radii > 1e-6, radii, 1.0)
            radii_clipped = np.minimum(radii_safe, 1.0 - 1e-4)
            denom = radii_clipped * (1.0 - radii_clipped)**3
            denom = np.where(denom > 1e-8, denom, 1.0)
            bound_forces = -lambda_bound * points / denom[:, np.newaxis]
            
            total_forces = pair_forces + bound_forces
            points += total_forces * polish_step
            
            new_radii = np.hypot(points[:, 0], points[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / new_radii[outside, np.newaxis]
            
            polish_step *= 0.99999

    return best_points
