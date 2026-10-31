"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def construct():
    # We want to pack 21 circles in a rectangle with perimeter 4.
    # Perimeter = 2*(w+h) = 4 => w+h = 2.
    # We maximize sum of radii.
    
    # Strategy:
    # 1. Iterate over aspect ratios.
    # 2. For each aspect ratio, use a high-resolution grid.
    # 3. Use Hungarian algorithm to select N cells that maximize distance to boundary.
    # 4. Refine positions using local optimization to maximize the minimum radius.
    
    best_result = None
    best_sum = -1
    
    # Aspect ratios w / (w+h). Since w+h=2, w = 2*frac.
    # Try a range of aspect ratios
    w_fracs = np.linspace(0.25, 0.75, 11) # 0.25 to 0.75
    
    # Grid resolutions to try (resolution in x and y)
    # We want a grid fine enough to allow good placement, but not too large for Hungarian
    grid_resolutions = [
        (20, 20),
        (30, 30),
        (40, 40),
        (50, 50)
    ]
    
    for w_frac in w_fracs:
        w = 2 * w_frac
        h = 2 - w
        
        for res_x, res_y in grid_resolutions:
            # Generate grid points
            # We want points strictly inside the rectangle to allow for radius
            # Leave a small margin? No, we will handle boundary in optimization.
            # But for Hungarian, we want to avoid points too close to boundary if possible?
            # Actually, the cost function will handle it.
            
            xs = np.linspace(0, w, res_x + 2)[1:-1] # Exclude edges
            ys = np.linspace(0, h, res_y + 2)[1:-1]
            
            grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
            grid_x = grid_x.flatten()
            grid_y = grid_y.flatten()
            num_cells = len(grid_x)
            
            if num_cells < N:
                continue
                
            # Cost matrix for Hungarian
            # We want to maximize the distance to the boundary for each selected cell.
            # Cost = -distance_to_boundary
            # Distance to boundary for a point (x,y) is min(x, w-x, y, h-y)
            dists_to_boundary = np.minimum(np.minimum(grid_x, w - grid_x), 
                                          np.minimum(grid_y, h - grid_y))
            
            # We assign N circles to num_cells cells.
            # Create a cost matrix of shape (N, num_cells)
            # Since all circles are identical, the cost for any circle i to cell j is the same.
            # Cost[i, j] = -dists_to_boundary[j]
            # To use linear_sum_assignment, we need a square matrix or we can use the rectangular version.
            # scipy.optimize.linear_sum_assignment handles rectangular matrices.
            
            # However, if all rows are identical, Hungarian will just pick the N cells with the lowest cost (highest boundary dist).
            # This is equivalent to sorting.
            # Let's add a small random noise or tie-breaker to ensure diversity?
            # Or just take the top N by boundary distance.
            # The directive says "use Hungarian". We will use it formally.
            
            # Construct cost matrix
            # To save memory/time, we can just sort if we recognize the structure, but let's follow the directive.
            # For N=21 and ~1000 cells, a 21x1000 matrix is small.
            
            # We want to maximize sum of dists_to_boundary.
            # So minimize sum of -dists_to_boundary.
            cost_matrix = -np.tile(dists_to_boundary, (N, 1))
            
            # Add a small penalty for being close to other cells? 
            # Hungarian is static. We cannot easily do this.
            # We will rely on the local optimization to fix overlaps.
            
            try:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                # row_ind will be 0..N-1, col_ind will be the indices of the selected cells
                selected_indices = col_ind
                positions = np.column_stack((grid_x[selected_indices], grid_y[selected_indices]))
            except Exception:
                continue
            
            # Calculate initial radius
            if N > 1:
                dists = cdist(positions, positions)
                np.fill_diagonal(dists, np.inf)
                min_pair_dist = np.min(dists)
            else:
                min_pair_dist = np.inf
            
            min_boundary_dist = np.min(dists_to_boundary[selected_indices])
            
            r = min(min_pair_dist, min_boundary_dist) / 2 - 1e-6
            
            if r <= 0:
                continue
                
            # Local Optimization
            # We will try to increase r by moving positions.
            # We use a simple repulsion-based local search.
            # We want to maximize the minimum distance.
            
            pos = positions.copy()
            current_r = r
            
            # Optimization loop
            for _ in range(200):
                # Compute pairwise distances
                dists = cdist(pos, pos)
                np.fill_diagonal(dists, np.inf)
                
                # Find the minimum distance
                min_d = np.min(dists)
                # Find the pair (i, j) with min distance
                i, j = np.unravel_index(np.argmin(dists), dists.shape)
                
                # Also check boundary constraints
                # d_boundary[i] = min(pos[i,0], w-pos[i,0], pos[i,1], h-pos[i,1])
                d_boundaries = np.minimum(np.minimum(pos[:, 0], w - pos[:, 0]), 
                                         np.minimum(pos[:, 1], h - pos[:, 1]))
                min_b = np.min(d_boundaries)
                k = np.argmin(d_boundaries)
                
                # Determine the bottleneck
                if min_d < min_b:
                    # Bottleneck is between i and j
                    # Push i and j apart
                    delta = pos[i] - pos[j]
                    norm = np.linalg.norm(delta)
                    if norm < 1e-6:
                        # Randomly perturb
                        delta = np.random.randn(2) * 1e-3
                        norm = np.linalg.norm(delta)
                    unit_vec = delta / norm
                    # Move i in direction of unit_vec, j in opposite
                    step = 0.5 * (2 * current_r - min_d) # We want distance to be 2*current_r
                    # Actually, we want to increase the minimum distance.
                    # Let's just move them apart by a small amount
                    step = 1e-3
                    pos[i] += unit_vec * step
                    pos[j] -= unit_vec * step
                else:
                    # Bottleneck is boundary for k
                    # Push k away from boundary
                    # Determine which boundary is closest
                    x, y = pos[k]
                    dx_left = x
                    dx_right = w - x
                    dy_bottom = y
                    dy_top = h - y
                    
                    min_dist = min(dx_left, dx_right, dy_bottom, dy_top)
                    
                    if min_dist == dx_left:
                        pos[k, 0] += 1e-3
                    elif min_dist == dx_right:
                        pos[k, 0] -= 1e-3
                    elif min_dist == dy_bottom:
                        pos[k, 1] += 1e-3
                    else:
                        pos[k, 1] -= 1e-3
                        
                # Ensure positions stay within bounds
                pos[:, 0] = np.clip(pos[:, 0], 1e-6, w - 1e-6)
                pos[:, 1] = np.clip(pos[:, 1], 1e-6, h - 1e-6)
                
                # Update radius
                dists_new = cdist(pos, pos)
                np.fill_diagonal(dists_new, np.inf)
                new_min_pair = np.min(dists_new)
                
                d_boundaries_new = np.minimum(np.minimum(pos[:, 0], w - pos[:, 0]), 
                                             np.minimum(pos[:, 1], h - pos[:, 1]))
                new_min_bound = np.min(d_boundaries_new)
                
                new_r = min(new_min_pair, new_min_bound) / 2 - 1e-7
                
                if new_r > current_r:
                    current_r = new_r
                else:
                    # If we didn't improve, we might be stuck. 
                    # We can try a larger step or random restart, but for now, break if no improvement in several steps?
                    # Let's just continue for a few iterations to see if it recovers.
                    pass
            
            total_sum = N * current_r
            if total_sum > best_sum:
                best_sum = total_sum
                result = np.zeros((N, 3))
                result[:, 0] = pos[:, 0]
                result[:, 1] = pos[:, 1]
                result[:, 2] = current_r
                best_result = result

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
