"""Improved packing: hexagonal lattice initialization + repulsive force relaxation."""
import numpy as np
from scipy.optimize import minimize

N = 200


def construct():
    # Initialize on a hexagonal lattice clipped to the unit square [-0.5, 0.5]^2
    # Then perform repulsive-force-based relaxation to maximize minimum distance
    
    # Hexagonal lattice parameters
    # For a hexagonal lattice with spacing s, the rows are offset
    # Row spacing: s * sqrt(3)/2
    # Horizontal spacing: s
    # Alternate rows shifted by s/2
    
    # Choose initial spacing roughly based on N
    # Area per circle ~ 1/N, so spacing ~ 1/sqrt(N)
    # For hexagonal, area per point = s * s * sqrt(3)/2
    # So s ~ sqrt(2/(N * sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    # Generate hexagonal lattice points
    # Number of rows: enough to cover the square
    row_spacing = s * np.sqrt(3.0) / 2.0
    n_rows = int(np.ceil(1.0 / row_spacing)) + 2
    
    points = []
    for i in range(n_rows):
        y = -0.5 + (i + 0.5) * row_spacing
        if y > 0.5:
            break
        x_offset = (s / 2.0) if (i % 2 == 1) else 0.0
        n_cols = int(np.ceil(1.0 / s)) + 2
        for j in range(n_cols):
            x = -0.5 + (j + 0.5) * s + x_offset
            if x > 0.5:
                break
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than N points, take a subset; if fewer, adjust spacing
    if len(points) > N:
        # Take every k-th point to get N points, or just take first N
        # Better: randomly sample N points from the lattice
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # Reduce spacing to get more points
        s = s * np.sqrt(len(points) / N)
        row_spacing = s * np.sqrt(3.0) / 2.0
        n_rows = int(np.ceil(1.0 / row_spacing)) + 2
        points = []
        for i in range(n_rows):
            y = -0.5 + (i + 0.5) * row_spacing
            if y > 0.5:
                break
            x_offset = (s / 2.0) if (i % 2 == 1) else 0.0
            n_cols = int(np.ceil(1.0 / s)) + 2
            for j in range(n_cols):
                x = -0.5 + (j + 0.5) * s + x_offset
                if x > 0.5:
                    break
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
        points = np.array(points)
        if len(points) > N:
            idx = np.random.choice(len(points), N, replace=False)
            points = points[idx]
        elif len(points) < N:
            # Pad with random points
            while len(points) < N:
                pt = np.random.uniform(-0.5, 0.5, 2)
                points = np.vstack([points, pt.reshape(1, 2)])
    
    points = points[:N]
    
    # Now perform repulsive force-based relaxation
    # We want to maximize the minimum pairwise distance and distance to boundaries
    # Use a soft-max approach: maximize sum of log(distances) or similar
    
    def objective(x):
        """Negative of a soft measure of minimum distance."""
        x = x.reshape(-1, 2)
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Distance to boundaries
        boundary_dists = np.minimum(
            np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
            np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
        )
        
        # Combine all distances
        all_dists = np.concatenate([dists[np.triu_indices(N, k=1)], boundary_dists])
        
        # Soft-min: -log(sum(exp(-beta * d))) / beta approximates min
        beta = 20.0
        soft_min = -np.log(np.sum(np.exp(-beta * all_dists))) / beta
        return -soft_min
    
    def objective_grad(x):
        """Gradient of the objective."""
        x = x.reshape(-1, 2)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2)) + 1e-12
        np.fill_diagonal(dists, np.inf)
        
        # Distance to boundaries
        boundary_dists = np.minimum(
            np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
            np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
        )
        
        all_dists = np.concatenate([dists[np.triu_indices(N, k=1)], boundary_dists])
        beta = 20.0
        weights = np.exp(-beta * all_dists)
        weights /= np.sum(weights)
        
        grad = np.zeros_like(x)
        
        # Pairwise contributions
        idx_i, idx_j = np.triu_indices(N, k=1)
        pair_dists = dists[idx_i, idx_j]
        pair_weights = weights[:len(idx_i)]
        
        for k in range(len(idx_i)):
            i, j = idx_i[k], idx_j[k]
            w = pair_weights[k]
            d = pair_dists[k]
            if d > 1e-10:
                # d/dx_i = -(x_i - x_j) / d * (-beta * w * exp(-beta*d)) / sum
                # Actually let's just use numerical or a simpler repulsive force
                pass
        
        # Use a simpler repulsive force approach instead
        return None
    
    # Use a simpler iterative repulsive force method
    # Each iteration: compute forces, move points
    
    n_iterations = 500
    step_size = 0.01
    
    for it in range(n_iterations):
        x = points.reshape(-1, 2)
        forces = np.zeros_like(x)
        
        # Pairwise repulsive forces
        for i in range(N):
            for j in range(i + 1, N):
                diff = x[i] - x[j]
                d = np.sqrt(np.sum(diff ** 2)) + 1e-10
                if d < 0.1:  # Only interact with nearby points for efficiency
                    force = diff / (d ** 2)
                    forces[i] += force
                    forces[j] -= force
        
        # Boundary repulsive forces
        for i in range(N):
            # Left boundary
            d_left = x[i, 0] + 0.5
            if d_left < 0.1:
                forces[i, 0] += 1.0 / (d_left + 1e-10)
            # Right boundary
            d_right = 0.5 - x[i, 0]
            if d_right < 0.1:
                forces[i, 0] -= 1.0 / (d_right + 1e-10)
            # Bottom boundary
            d_bottom = x[i, 1] + 0.5
            if d_bottom < 0.1:
                forces[i, 1] += 1.0 / (d_bottom + 1e-10)
            # Top boundary
            d_top = 0.5 - x[i, 1]
            if d_top < 0.1:
                forces[i, 1] -= 1.0 / (d_top + 1e-10)
        
        # Update positions
        points += step_size * forces
        
        # Clamp to square
        points[:, 0] = np.clip(points[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        points[:, 1] = np.clip(points[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Decrease step size
        step_size *= 0.99
    
    # Final: ensure all points are strictly inside
    points[:, 0] = np.clip(points[:, 0], -0.5 + 1e-12, 0.5 - 1e-12)
    points[:, 1] = np.clip(points[:, 1], -0.5 + 1e-12, 0.5 - 1e-12)
    
    return points
