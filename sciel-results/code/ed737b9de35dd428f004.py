"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    """
    Construct centres of 200 equal circles inside the unit square [-0.5, 0.5]^2.
    
    Strategy:
    1. Start from a dense hexagonal lattice pattern (good initial packing).
    2. Use a projected gradient ascent method to maximize the minimum distance
       between circles and from boundaries.
    3. The objective is to maximize r = min(min_pairwise_dist/2, min_boundary_dist).
    """
    np.random.seed(42)
    
    # Step 1: Create initial hexagonal lattice arrangement
    # For 200 circles, we need roughly a 14x15 grid area
    # Hexagonal packing has density pi/(2*sqrt(3)) ≈ 0.9069
    # Square grid has density 1.0 but less efficient for circle packing
    
    # Let's start with a hexagonal lattice
    # In hexagonal packing, rows are offset
    # Distance between adjacent circles in hex lattice: d
    # Row spacing: d * sqrt(3)/2
    # Column spacing: d
    # Offset rows: d/2
    
    # We want to fit ~200 circles in [-0.5, 0.5]^2
    # Let's try different numbers of rows/columns
    
    def hex_lattice_init(n_rows, n_cols, offset=True):
        """Generate hexagonal lattice points."""
        points = []
        # Estimate spacing
        # We want to fit in [-0.5, 0.5]
        # x range: (n_cols - 1) * d + (0.5 if offset else 0) * d <= 1.0
        # y range: (n_rows - 1) * d * sqrt(3)/2 <= 1.0
        
        # Try to find good d
        d = min(1.0 / max(n_cols - 1, 1), 1.0 / max((n_rows - 1) * np.sqrt(3) / 2, 1))
        
        for i in range(n_rows):
            for j in range(n_cols):
                x = (j - (n_cols - 1) / 2.0) * d
                if offset and i % 2 == 1:
                    x += d / 2.0
                y = (i - (n_rows - 1) / 2.0) * d * np.sqrt(3) / 2.0
                points.append([x, y])
        
        points = np.array(points)
        return points[:N]
    
    # Try several configurations and pick the best initial one
    best_init = None
    best_init_score = 0
    
    for n_rows in range(12, 18):
        for n_cols in range(12, 18):
            if n_rows * n_cols < N:
                continue
            pts = hex_lattice_init(n_rows, n_cols)
            if len(pts) < N:
                continue
            
            # Compute score
            score = compute_score(pts)
            if score > best_init_score:
                best_init_score = score
                best_init = pts.copy()
    
    if best_init is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_init = grid[:N].copy()
    
    # Step 2: Projected gradient ascent to maximize minimum distance
    # We'll iteratively move points to increase the minimum constraint
    
    points = best_init.copy()
    
    # Precompute which pairs are close (within some threshold)
    def compute_score(pts):
        """Compute the score: min of pairwise distances/2 and boundary distances."""
        if len(pts) < 2:
            return 0.0
        
        # Boundary distances
        boundary_dists = np.min([
            0.5 - pts[:, 0],
            0.5 + pts[:, 0],
            0.5 - pts[:, 1],
            0.5 + pts[:, 1]
        ], axis=1)
        min_boundary = np.min(boundary_dists)
        
        # Pairwise distances - use efficient computation
        # For N=200, we can do O(N^2)
        dists = np.zeros((len(pts), len(pts)))
        for i in range(len(pts)):
            diff = pts[i] - pts[i+1:]
            dists[i, i+1:] = np.sqrt(np.sum(diff**2, axis=1))
        
        # Get minimum pairwise distance (excluding diagonal)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        
        return min(min_boundary, min_pairwise / 2.0)
    
    # Gradient ascent with projection
    # At each step, find the binding constraints and move points to increase them
    
    max_iter = 2000
    step_size = 0.01
    
    for iteration in range(max_iter):
        # Compute current score
        current_score = compute_score(points)
        
        # Find the minimum pairwise distance and boundary distances
        boundary_dists = np.min([
            0.5 - points[:, 0],
            0.5 + points[:, 0],
            0.5 - points[:, 1],
            0.5 + points[:, 1]
        ], axis=1)
        
        # Find all pairs with distance close to minimum
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        
        # Find pairs that are at or near the minimum distance
        # Use a tolerance
        tol = 1e-6
        close_pairs = np.argwhere(dists <= min_dist + tol)
        close_pairs = close_pairs[close_pairs[:, 0] < close_pairs[:, 1]]
        
        # Find points with minimum boundary distance
        min_boundary = np.min(boundary_dists)
        close_boundary = np.argwhere(boundary_dists <= min_boundary + tol)
        close_boundary = close_boundary.flatten()
        
        # Compute gradient direction for each point
        # For each point, we want to move it away from its closest neighbors
        # and away from boundaries
        
        gradient = np.zeros_like(points)
        
        # Handle pairwise constraints
        for i, j in close_pairs:
            if i == j:
                continue
            diff_ij = points[i] - points[j]
            norm_ij = np.linalg.norm(diff_ij)
            if norm_ij < 1e-10:
                # Random small perturbation
                diff_ij = np.random.randn(2) * 1e-8
                norm_ij = np.linalg.norm(diff_ij)
            
            # Move i away from j, j away from i
            direction = diff_ij / norm_ij
            gradient[i] += direction
            gradient[j] -= direction
        
        # Handle boundary constraints
        for idx in close_boundary:
            x, y = points[idx]
            # Distance to each boundary
            d_left = 0.5 + x
            d_right = 0.5 - x
            d_bottom = 0.5 + y
            d_top = 0.5 - y
            
            min_b = min(d_left, d_right, d_bottom, d_top)
            
            if min_b == d_left:
                gradient[idx, 0] += 1.0  # Move right
            elif min_b == d_right:
                gradient[idx, 0] -= 1.0  # Move left
            elif min_b == d_bottom:
                gradient[idx, 1] += 1.0  # Move up
            elif min_b == d_top:
                gradient[idx, 1] -= 1.0  # Move down
        
        # Normalize gradient
        grad_norm = np.linalg.norm(gradient)
        if grad_norm < 1e-10:
            break
        
        gradient /= grad_norm
        
        # Move points
        new_points = points + step_size * gradient
        
        # Project back into the square
        new_points[:, 0] = np.clip(new_points[:, 0], -0.5, 0.5)
        new_points[:, 1] = np.clip(new_points[:, 1], -0.5, 0.5)
        
        # Check if score improved
        new_score = compute_score(new_points)
        
        if new_score > current_score:
            points = new_points
            # Can increase step size slightly
            step_size = min(step_size * 1.05, 0.05)
        else:
            # Decrease step size
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    # Final cleanup: ensure points are strictly inside the square
    points[:, 0] = np.clip(points[:, 0], -0.4999, 0.4999)
    points[:, 1] = np.clip(points[:, 1], -0.4999, 0.4999)
    
    return points


# EVOLVE-BLOCK-END
