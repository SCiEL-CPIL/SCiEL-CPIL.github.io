import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Initialize with a hexagonal lattice
    # Estimate radius to fit N circles
    # Area per circle ~ 1/N, pi*r^2 ~ 1/N => r ~ sqrt(1/(pi*N))
    # But hexagonal packing is denser.
    # Let's just create a grid that is roughly hexagonal.
    
    # We want to place N points.
    # Let's try to fit a hexagonal grid.
    # Side of square is 1.
    # Let r be the radius.
    # Horizontal spacing: 2r
    # Vertical spacing: r * sqrt(3)
    
    # We can adjust r to fit N points.
    # Let's start with a reasonable guess for r, e.g., 0.035
    r_init = 0.035
    
    points = []
    y = -0.5 + r_init
    row_idx = 0
    while len(points) < N and y <= 0.5 - r_init:
        if row_idx % 2 == 0:
            x = -0.5 + r_init
        else:
            x = -0.5 + 2*r_init # Offset by one diameter? No, hex offset is r.
            # Hex lattice:
            # Row 0: x = -0.5 + r, -0.5 + r + 2r, ...
            # Row 1: x = -0.5 + 2r, -0.5 + 2r + 2r, ...
            # Vertical distance: r * sqrt(3)
            pass
            
        # Correct hex lattice generation:
        # Let dx = 2*r_init
        # Let dy = r_init * np.sqrt(3)
        # Row 0: x = -0.5 + r_init + i*dx
        # Row 1: x = -0.5 + 2*r_init + i*dx
        # Row 2: x = -0.5 + r_init + i*dx
        # ...
        
        dx = 2 * r_init
        dy = r_init * np.sqrt(3)
        
        if row_idx % 2 == 0:
            x_start = -0.5 + r_init
        else:
            x_start = -0.5 + 2 * r_init
            
        x = x_start
        while x <= 0.5 - r_init and len(points) < N:
            points.append([x, y])
            x += dx
        y += dy
        row_idx += 1
        
    # If we have fewer than N points, we need to add more or adjust.
    # If we have more, we truncate.
    points = np.array(points[:N])
    
    if len(points) < N:
        # This shouldn't happen with r_init=0.035, but just in case
        # Fill remaining with random points inside square
        remaining = N - len(points)
        new_points = np.random.uniform(-0.5 + r_init, 0.5 - r_init, size=(remaining, 2))
        points = np.vstack([points, new_points])

    # 2. Define objective function: minimize negative of min radius
    # We want to maximize r = min(min_pairwise_dist/2, min_boundary_dist)
    
    def compute_min_radius(pts):
        # Pairwise distances
        # pts is (N, 2)
        # Use broadcasting to compute distance matrix
        # dist[i, j] = || pts[i] - pts[j] ||
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Zero out diagonal
        np.fill_diagonal(dists, np.inf)
        
        min_pairwise = np.min(dists) / 2.0
        
        # Boundary distances
        # Distance to x = -0.5 is pts[:, 0] + 0.5
        # Distance to x = 0.5 is 0.5 - pts[:, 0]
        # Distance to y = -0.5 is pts[:, 1] + 0.5
        # Distance to y = 0.5 is 0.5 - pts[:, 1]
        
        dist_x_min = pts[:, 0] + 0.5
        dist_x_max = 0.5 - pts[:, 0]
        dist_y_min = pts[:, 1] + 0.5
        dist_y_max = 0.5 - pts[:, 1]
        
        min_boundary = np.min(np.concatenate([dist_x_min, dist_x_max, dist_y_min, dist_y_max]))
        
        return min(min_pairwise, min_boundary)
    
    current_r = compute_min_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    # 3. Local Hill Climbing
    step_size = 0.01
    iterations = 0
    max_iterations = 50000 # Safety cap, but time is the real limit
    
    while time.time() - start_time < deadline and iterations < max_iterations:
        iterations += 1
        
        # Pick a random point
        idx = np.random.randint(N)
        
        # Save original position
        orig_pos = points[idx].copy()
        
        # Generate perturbation
        # Scale step size down slowly to allow fine tuning
        perturbation = np.random.normal(0, step_size, size=2)
        new_pos = orig_pos + perturbation
        
        # Check if new position is inside the square (with a small margin to avoid invalid states)
        # Actually, the radius calculation handles boundaries, but if a point goes outside, 
        # the "radius" becomes negative or very small, which is bad.
        # We can just check if it's strictly inside [-0.5, 0.5]
        if new_pos[0] < -0.5 or new_pos[0] > 0.5 or new_pos[1] < -0.5 or new_pos[1] > 0.5:
            continue
            
        points[idx] = new_pos
        new_r = compute_min_radius(points)
        
        if new_r > best_r:
            best_r = new_r
            best_points = points.copy()
        else:
            # Revert if not better than current best (or current state)
            # Standard hill climbing: accept if better than current, reject otherwise.
            # But since we track best, we can just revert to orig_pos if it didn't improve current_r?
            # Let's stick to strict improvement on current state to avoid oscillation.
            points[idx] = orig_pos
            
        # Reduce step size slightly to converge
        if iterations % 1000 == 0:
            step_size *= 0.95
            
    return best_points

# EVOLVE-BLOCK-END
