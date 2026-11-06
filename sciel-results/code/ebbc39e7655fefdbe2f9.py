import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

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

    # Binary search for optimal lattice spacing
    d_low = 0.05
    d_high = 0.15
    
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

    # Add small jitter to break symmetry
    points += np.random.normal(0, 0.0005, (N, 2))

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
    
    # 2. Optimization: Force-based relaxation
    # Using a smooth potential that pushes points apart and away from the boundary.
    
    step_size = 0.005
    no_improve_count = 0
    max_no_improve = 50
    
    for it in range(200000):
        if time.time() > deadline:
            break
        
        # Compute current radius periodically to save time
        if it % 10 == 0:
            current_r = compute_radius(points)
            
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturb to escape local minimum
                    # Focus on points with smallest clearance
                    radii = np.hypot(points[:, 0], points[:, 1])
                    clearances = 1.0 - radii
                    # Also consider local density (inverse of min distance to neighbors)
                    # For simplicity, just perturb a subset of points with low clearance
                    low_clearance_idx = np.where(clearances < np.percentile(clearances, 20))[0]
                    if len(low_clearance_idx) > 0:
                        points[low_clearance_idx] += np.random.normal(0, 0.005, (len(low_clearance_idx), 2))
                    
                    # Also perturb some random points
                    random_idx = np.random.choice(N, size=10, replace=False)
                    points[random_idx] += np.random.normal(0, 0.003, (10, 2))
                    
                    # Ensure validity
                    new_radii = np.hypot(points[:, 0], points[:, 1])
                    outside = new_radii > 1.0
                    if np.any(outside):
                        points[outside] = points[outside] / new_radii[outside, np.newaxis] * 0.999
                    
                    no_improve_count = 0
                    step_size *= 0.8

        # Compute Forces
        # Pairwise repulsion: V = 1/d^2, Force = -grad V = 2/d^3 * direction
        # Boundary repulsion: V = 1/(1-r)^2, Force = -grad V = 2*r/(1-r)^3 * direction
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1.0)  # Avoid division by zero
        
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, 1.0)
        
        # Soften to avoid singularities
        eps = 0.01
        dists_safe = np.maximum(dists, eps)
        
        # Pairwise forces: F_i = sum_j (2/d_ij^3) * (p_i - p_j)
        # Factor for each pair: 2 / d^3
        factors = 2.0 / (dists_safe ** 3)
        mask = np.ones((N, N), dtype=bool)
        np.fill_diagonal(mask, False)
        factors[~mask] = 0.0
        
        pair_forces = np.einsum('ij,ijk->ik', factors, diff)
        
        # Boundary forces
        radii = np.hypot(points[:, 0], points[:, 1])
        radii_safe = np.where(radii > 1e-6, radii, 1.0)
        radii_clipped = np.minimum(radii_safe, 1.0 - 1e-4)
        denom = radii_clipped * (1.0 - radii_clipped)**3
        denom = np.where(denom > 1e-8, denom, 1.0)
        
        # Force pointing inward: -lambda * p / denom
        # We want to push away from boundary, so if close to boundary, force is inward
        lambda_bound = 5.0
        bound_forces = -lambda_bound * points / denom[:, np.newaxis]
        
        total_forces = pair_forces + bound_forces
        
        # Update
        points += total_forces * step_size
        
        # Enforce hard constraint: inside unit circle
        new_radii = np.hypot(points[:, 0], points[:, 1])
        outside = new_radii > 1.0
        if np.any(outside):
            points[outside] = points[outside] / new_radii[outside, np.newaxis]
        
        # Decay step size
        step_size *= 0.99995

    # Final polish phase
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
            factors = 2.0 / (dists_safe ** 3)
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
