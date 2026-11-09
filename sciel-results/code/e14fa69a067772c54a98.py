"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs the centers of 300 circles inside a unit circle.
    Uses Hexagonal Grid Initialization followed by a Targeted Local Search 
    to maximize the minimum distance (packing radius).
    """
    
    def generate_hex_grid(n_points):
        # Estimate spacing s for hexagonal lattice
        # Area per point in hex packing ~ (3*sqrt(3)/2) * s^2
        # Total area pi
        # n * (3*sqrt(3)/2) * s^2 = pi  =>  s = sqrt(2*pi / (3*sqrt(3)*n))
        s = np.sqrt(2 * np.pi / (3 * np.sqrt(3) * n_points))
        
        # Generate grid points
        max_i = int(np.ceil(1.5 / s)) + 2
        max_j = int(np.ceil(1.5 / s)) + 2
        
        points = []
        for j in range(-max_j, max_j + 1):
            for i in range(-max_i, max_i + 1):
                x = s * (i + 0.5 * j)
                y = s * (np.sqrt(3) / 2) * j
                if x*x + y*y <= 1.0:
                    points.append((x, y))
        
        points = np.array(points)
        
        # If we don't have enough points, increase density
        if len(points) < n_points:
            s = s * 0.95
            max_i = int(np.ceil(1.5 / s)) + 2
            max_j = int(np.ceil(1.5 / s)) + 2
            points = []
            for j in range(-max_j, max_j + 1):
                for i in range(-max_i, max_i + 1):
                    x = s * (i + 0.5 * j)
                    y = s * (np.sqrt(3) / 2) * j
                    if x*x + y*y <= 1.0:
                        points.append((x, y))
            points = np.array(points)

        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        
        if len(points) > n_points:
            indices = indices[:n_points]
        else:
            # Pad with random points if somehow too few
            extra = np.random.uniform(-0.5, 0.5, (n_points - len(points), 2))
            points = np.vstack([points, extra])
            indices = np.arange(n_points)

        return points[indices]

    pos = generate_hex_grid(N)
    
    # Targeted Local Search
    max_iterations = 2000
    step_size = 0.005
    epsilon = 1e-6
    
    for it in range(max_iterations):
        # Calculate pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_dist_sq = np.min(dists_sq)
        min_pair_dist = np.sqrt(min_pair_dist_sq)
        
        # Boundary distances
        radii = np.linalg.norm(pos, axis=1)
        boundary_dists = 1.0 - radii
        min_boundary_dist = np.min(boundary_dists)
        
        # Current radius r
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        # Determine active constraint
        if min_pair_dist / 2.0 <= min_boundary_dist:
            # Push pairs apart
            threshold_sq = (min_pair_dist * 1.1)**2
            close_mask = dists_sq < threshold_sq
            np.fill_diagonal(close_mask, False)
            
            sum_vecs = np.sum(
                np.where(close_mask[:, :, np.newaxis], diff, 0.0), 
                axis=1
            )
            forces = sum_vecs
            norms = np.linalg.norm(forces, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            forces = forces / norms
            
            pos += forces * step_size
        else:
            # Push points inward
            boundary_mask = boundary_dists < (min_boundary_dist + 1e-4)
            radii_norm = np.linalg.norm(pos, axis=1, keepdims=True)
            radii_norm[radii_norm == 0] = 1.0
            unit_inward = -pos / radii_norm
            forces = np.where(boundary_mask[:, np.newaxis], unit_inward, 0.0)
            
            pos += forces * step_size

        # Project outside points back
        radii = np.linalg.norm(pos, axis=1)
        outside_mask = radii > 1.0
        if np.any(outside_mask):
            pos[outside_mask] = pos[outside_mask] / radii[outside_mask, np.newaxis]

    return pos

# EVOLVE-BLOCK-END
