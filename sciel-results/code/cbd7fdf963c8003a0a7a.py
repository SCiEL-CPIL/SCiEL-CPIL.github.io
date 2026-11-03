"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct centers of 300 equal circles in the unit disk using a hexagonal grid 
    initialization followed by iterative relaxation to maximize minimum distance.
    """
    # Step 1: Generate hexagonal grid points inside the unit circle
    d = 0.11
    sqrt3 = np.sqrt(3)
    vert_spacing = d * sqrt3 / 2
    
    points = []
    max_row = int(1.0 / vert_spacing) + 2
    max_col = int(1.0 / d) + 2
    
    for row in range(-max_row, max_row + 1):
        y = row * vert_spacing
        x_offset = (d / 2) if (row % 2 != 0) else 0
        for col in range(-max_col, max_col + 1):
            x = col * d + x_offset
            if x**2 + y**2 <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # Ensure we have exactly N points
    if len(points) > N:
        # Select N points
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        points = points[indices]
    elif len(points) < N:
        # Add random points to fill up to N
        np.random.seed(42)
        while len(points) < N:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.sqrt(np.random.uniform(0, 1))
            points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
    
    # Step 2: Iterative relaxation
    n_relax_iterations = 200
    
    for it in range(n_relax_iterations):
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        min_pair_dist = np.min(dist)
        center_dists = np.linalg.norm(points, axis=1)
        min_boundary = 1.0 - np.max(center_dists)
        current_min = min(min_pair_dist, min_boundary)
        
        # Target: increase current_min by 5%
        target = current_min * 1.05
        
        # Find pairs with distance < target
        mask = dist < target
        np.fill_diagonal(mask, False)
        
        # If no pairs to fix and boundary is okay, break
        if not np.any(mask) and min_boundary >= target:
            break
        
        # Compute displacement for each point
        displacements = np.zeros_like(points)
        
        for i in range(N):
            for j in range(i + 1, N):
                if mask[i, j]:
                    d_ij = dist[i, j]
                    if d_ij < 1e-9:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = 0.001
                        displacements[i, 0] += push * np.cos(angle)
                        displacements[i, 1] += push * np.sin(angle)
                        displacements[j, 0] -= push * np.cos(angle)
                        displacements[j, 1] -= push * np.sin(angle)
                    else:
                        direction = (points[i] - points[j]) / d_ij
                        push_amount = (target - d_ij) / 2
                        displacements[i] += push_amount * direction
                        displacements[j] -= push_amount * direction
        
        # Apply displacements
        points += displacements
        
        # Handle boundary: pull points back inside if they exceed 1 - current_min
        center_dists = np.linalg.norm(points, axis=1)
        limit = 1.0 - current_min
        overflow = np.maximum(0, center_dists - limit)
        safe_dists = np.where(center_dists > 1e-9, center_dists, 1e-9)
        points -= (overflow / safe_dists)[:, np.newaxis] * points
    
    # Final: ensure all points are strictly inside unit circle
    center_dists = np.linalg.norm(points, axis=1)
    for i in range(N):
        if center_dists[i] > 1.0:
            points[i] *= (1.0 - 1e-6) / center_dists[i]
    
    return points
# EVOLVE-BLOCK-END
