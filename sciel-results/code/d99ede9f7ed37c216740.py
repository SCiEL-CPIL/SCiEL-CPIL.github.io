"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import itertools

def construct():
    # We want to pack 21 circles in a rectangle with perimeter 4.
    # Perimeter = 2*(w+h) = 4 => w+h = 2.
    # We maximize sum of radii.
    
    best_sum = -1
    best_result = None
    
    # Try different aspect ratios
    # w + h = 2
    # Let's sample w from 0.5 to 1.5
    aspect_ratios = np.linspace(0.4, 1.6, 13) # w values
    
    # Grid resolution for candidate positions
    # High resolution helps find better local optima
    grid_res = 40
    
    for w in aspect_ratios:
        h = 2.0 - w
        if w <= 0 or h <= 0:
            continue
            
        # Create grid of candidate positions
        # We need to ensure positions are strictly inside so radius > 0
        # Margin 0.01
        margin = 0.01
        xs = np.linspace(margin, w - margin, grid_res)
        ys = np.linspace(margin, h - margin, grid_res)
        
        # Flatten grid to list of (x, y)
        grid_points = np.array([(x, y) for x in xs for y in ys])
        M = len(grid_points)
        
        # Initialize positions: Randomly select N unique grid points
        # Or spread them out initially
        # Let's try a few random initializations
        for init_iter in range(5):
            # Initial positions: Random sample from grid
            indices = np.random.choice(M, N, replace=False)
            positions = grid_points[indices].copy()
            
            # Iterate to improve
            for _ in range(20): # Max iterations
                # Calculate cost matrix C (N x M)
                # C[i, j] = -radius_if_circle_i_at_grid_j
                C = np.zeros((N, M))
                
                for i in range(N):
                    # Radius determined by:
                    # 1. Distance to boundary
                    # 2. Distance to other circles (current positions)
                    
                    # Dist to boundary for all grid points
                    dist_boundary = np.minimum(
                        np.minimum(grid_points[:, 0], w - grid_points[:, 0]),
                        np.minimum(grid_points[:, 1], h - grid_points[:, 1])
                    )
                    
                    # Dist to other circles
                    # For each grid point j, dist to pos_k for all k != i
                    # We can vectorize this:
                    # grid_points shape (M, 2)
                    # positions shape (N, 2)
                    
                    # Calculate distances from each grid point to all other circles
                    # diff shape (M, N, 2)
                    # dist shape (M, N)
                    dists_to_others = np.linalg.norm(grid_points[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
                    
                    # Exclude self
                    dists_to_others[:, i] = np.inf
                    
                    # Min dist to any other circle
                    min_dist_to_others = np.min(dists_to_others, axis=1)
                    
                    # Feasible radius at grid point j for circle i
                    radii = np.minimum(dist_boundary, min_dist_to_others / 2.0)
                    
                    # Cost is negative radius
                    C[i, :] = -radii
                
                # Solve Assignment Problem
                # Minimize total cost (maximize total radius)
                row_ind, col_ind = linear_sum_assignment(C)
                
                new_positions = grid_points[col_ind]
                
                # Check if positions changed
                if np.allclose(positions, new_positions):
                    break
                    
                positions = new_positions
                
            # Final evaluation of this configuration
            # Calculate the actual max sum of radii
            # The radii are not fixed, we can scale them up?
            # No, the positions are fixed on the grid.
            # The "radius" for circle i is min(dist to boundary, min(dist to others)/2)
            # Sum of these radii is the value.
            
            radii = np.zeros(N)
            for i in range(N):
                dist_b = min(positions[i][0], w - positions[i][0], positions[i][1], h - positions[i][1])
                min_dist_others = np.inf
                for j in range(N):
                    if i != j:
                        d = np.linalg.norm(positions[i] - positions[j])
                        if d < min_dist_others:
                            min_dist_others = d
                radii[i] = min(dist_b, min_dist_others / 2.0)
            
            total_sum = np.sum(radii)
            
            if total_sum > best_sum:
                best_sum = total_sum
                result = np.zeros((N, 3))
                result[:, 0] = positions[:, 0]
                result[:, 1] = positions[:, 1]
                result[:, 2] = radii
                best_result = result

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
