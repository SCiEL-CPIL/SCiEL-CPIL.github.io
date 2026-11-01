"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Places 50 spheres on the surface of an inscribed sphere within the unit cube.
    Uses a Fibonacci sphere initialization followed by local optimization to maximize
    the minimum chord distance between points on the unit sphere.
    Then determines the optimal sphere radius R that balances the wall constraint
    and the inter-sphere distance constraint.
    """
    # 1. Initialize with Fibonacci sphere on the unit sphere
    points = np.zeros((N, 3))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(N):
        z = 1.0 - (i / (N - 1)) * 2.0
        r = np.sqrt(1.0 - z**2)
        theta = golden_angle * i
        points[i, 0] = np.cos(theta) * r
        points[i, 1] = np.sin(theta) * r
        points[i, 2] = z

    # 2. Local optimization to maximize the minimum distance between points on the sphere
    # We use a simple force-based relaxation projected onto the tangent plane.
    
    def calculate_min_dist(pts):
        # Calculate pairwise distances
        # pts shape (N, 3)
        # dists shape (N, N)
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        return np.min(dists)

    current_min_dist = calculate_min_dist(points)
    
    # Relaxation loop
    # We limit iterations to ensure speed.
    max_iters = 20
    for _ in range(max_iters):
        improved = False
        # Calculate forces for all points
        # Force on i is sum over j != i of (p_i - p_j) / ||p_i - p_j||^2
        # This is O(N^2)
        
        forces = np.zeros_like(points)
        # Vectorized force calculation
        # diff[i, j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0) # Avoid division by zero
        
        # Inverse square law repulsion
        # We want to push points apart
        # Force vector = diff / dist^2
        # But we only want the component tangent to the sphere
        inv_dist_sq = 1.0 / (dists**2)
        inv_dist_sq[np.diag_indices_from(inv_dist_sq)] = 0.0
        
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project forces to tangent plane and apply step
        # normal[i] = points[i] / ||points[i]||
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        normals = points / norms
        
        # Tangent force = force - dot(force, normal) * normal
        dot_forces = np.sum(forces * normals, axis=1, keepdims=True)
        tangent_forces = forces - dot_forces * normals
        
        # Update points
        step_size = 0.05
        new_points = points + step_size * tangent_forces
        
        # Normalize to sphere
        new_norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / new_norms
        
        # Check if improvement
        new_min_dist = calculate_min_dist(new_points)
        
        if new_min_dist > current_min_dist:
            points = new_points
            current_min_dist = new_min_dist
            improved = True
        else:
            # If no improvement, reduce step size or break
            # For simplicity, we break if no improvement in a full pass
            break

    d_min_unit = current_min_dist

    # 3. Determine optimal radius R
    # Constraint 1: Wall distance r <= 0.5 - R
    # Constraint 2: Inter-sphere distance r <= 0.5 * R * d_min_unit
    # Maximize r = min(0.5 - R, 0.5 * R * d_min_unit)
    # Optimal R is where 0.5 - R = 0.5 * R * d_min_unit
    # 0.5 = R * (1 + 0.5 * d_min_unit)
    R_opt = 0.5 / (1.0 + 0.5 * d_min_unit)

    # 4. Scale points to optimal radius
    centers = points * R_opt

    return centers
# EVOLVE-BLOCK-END
