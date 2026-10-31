"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def construct():
    """
    Pack 21 circles in a rectangle with perimeter 4 (w+h=2) maximizing sum of radii.
    
    Approach: Formulate as minimum-weight matching problem.
    - Create a high-resolution grid of candidate positions
    - Match circles to grid cells using Hungarian algorithm
    - Weight = cost of placing circle at that position (based on boundary constraints)
    - Iteratively optimize grid resolution and aspect ratio
    """
    
    best_result = None
    best_sum = -1
    
    # Try different aspect ratios
    aspect_ratios = np.linspace(0.25, 0.75, 11)  # w/2 ranges from 0.25 to 0.75
    
    for w_frac in aspect_ratios:
        w = 2 * w_frac
        h = 2 - w
        
        if w <= 0 or h <= 0:
            continue
        
        # Try different grid resolutions
        for grid_res in [8, 10, 12, 15]:
            # Create grid of candidate positions
            # We need at least N positions, so make sure grid has enough cells
            nx = max(N, int(np.ceil(grid_res * w / max(w, h))))
            ny = max(N, int(np.ceil(grid_res * h / max(w, h))))
            
            # Actually, let's use a more sensible grid
            # Grid should cover the rectangle with fine resolution
            n_cells_x = grid_res
            n_cells_y = int(grid_res * h / w) if w > 0 else grid_res
            n_cells_y = max(1, min(n_cells_y, 50))  # Cap to avoid too many
            
            total_cells = n_cells_x * n_cells_y
            if total_cells < N:
                continue
            
            # Create grid positions
            xs = np.linspace(w / (2 * n_cells_x), w - w / (2 * n_cells_x), n_cells_x)
            ys = np.linspace(h / (2 * n_cells_y), h - h / (2 * n_cells_y), n_cells_y)
            
            # Generate all grid positions
            grid_positions = []
            for i in range(n_cells_x):
                for j in range(n_cells_y):
                    grid_positions.append([xs[i], ys[j]])
            
            grid_positions = np.array(grid_positions)
            
            # For each grid position, compute the maximum radius a circle could have
            # based on boundary constraints
            max_radii = np.array([
                min(px, w - px, py, h - py)
                for px, py in grid_positions
            ])
            
            # We want to select N positions that maximize the minimum radius
            # This is equivalent to minimizing the "cost" where cost = -max_radius
            # But we also need to ensure non-overlap, which is tricky with just matching
            
            # Better approach: Use matching to select positions, then optimize
            # For now, let's use a greedy approach with matching as a guide
            
            # Create cost matrix: cost[i, j] = -max_radius[j] (we want to maximize radius)
            # But we need to handle the non-overlap constraint
            
            # Alternative: Use iterative approach
            # 1. Start with all positions available
            # 2. Match circles to positions using Hungarian algorithm
            # 3. Compute minimum feasible radius
            # 4. Remove positions that are too close to selected ones
            # 5. Repeat
            
            # Let's try a simpler approach first:
            # Sort positions by max_radius descending, greedily select non-overlapping ones
            
            # Compute pairwise distances between grid positions
            if len(grid_positions) > 1:
                dists = cdist(grid_positions, grid_positions)
            else:
                dists = np.array([[0]])
            
            # Greedy selection with radius optimization
            selected_indices = []
            selected_positions = []
            available = np.ones(len(grid_positions), dtype=bool)
            
            # Sort by max_radius descending
            order = np.argsort(-max_radii)
            
            for idx in order:
                if len(selected_indices) >= N:
                    break
                if not available[idx]:
                    continue
                
                # Check if this position is far enough from all selected positions
                can_select = True
                for sel_idx in selected_indices:
                    if dists[idx, sel_idx] < 2 * max_radii[idx] * 0.95:  # 5% margin
                        can_select = False
                        break
                
                if can_select:
                    selected_indices.append(idx)
                    selected_positions.append(grid_positions[idx])
                    
                    # Mark nearby positions as unavailable
                    for other_idx in range(len(grid_positions)):
                        if other_idx == idx:
                            continue
                        if dists[idx, other_idx] < 2 * max_radii[idx]:
                            available[other_idx] = False
            
            if len(selected_indices) < N:
                continue
            
            selected_positions = np.array(selected_positions[:N])
            
            # Compute the minimum feasible radius
            # This is limited by:
            # 1. Boundary constraints: min(x, w-x, y, h-y) for each circle
            # 2. Pairwise distances: min distance between any two circles / 2
            
            boundary_limits = np.array([
                min(px, w - px, py, h - py)
                for px, py in selected_positions
            ])
            
            if N > 1:
                pair_dists = cdist(selected_positions, selected_positions)
                np.fill_diagonal(pair_dists, np.inf)
                min_pair_dist = np.min(pair_dists)
            else:
                min_pair_dist = np.inf
            
            r = min(np.min(boundary_limits), min_pair_dist / 2) - 1e-7
            
            if r <= 0:
                continue
            
            # Local optimization: try to improve positions
            # Use simple gradient-based optimization
            positions = selected_positions.copy()
            
            for iteration in range(30):
                # Compute current minimum radius
                boundary_limits = np.array([
                    min(px, w - px, py, h - py)
                    for px, py in positions
                ])
                
                if N > 1:
                    pair_dists = cdist(positions, positions)
                    np.fill_diagonal(pair_dists, np.inf)
                    min_pair_dist = np.min(pair_dists)
                else:
                    min_pair_dist = np.inf
                
                current_r = min(np.min(boundary_limits), min_pair_dist / 2)
                
                # Try to push circles apart
                # For each circle, compute force from nearest neighbors
                improved = False
                for i in range(N):
                    # Find nearest neighbor
                    if N > 1:
                        dists_to_others = np.linalg.norm(positions - positions[i], axis=1)
                        dists_to_others[i] = np.inf
                        nearest_j = np.argmin(dists_to_others)
                        nearest_dist = dists_to_others[nearest_j]
                        
                        # Push apart if too close
                        if nearest_dist < 2 * current_r + 0.01:
                            direction = positions[i] - positions[nearest_j]
                            norm = np.linalg.norm(direction)
                            if norm > 1e-10:
                                direction = direction / norm
                                # Small step
                                step = 0.001
                                new_pos = positions[i] + direction * step
                                # Check bounds
                                if (new_pos[0] > 0 and new_pos[0] < w and 
                                    new_pos[1] > 0 and new_pos[1] < h):
                                    positions[i] = new_pos
                                    improved = True
                
                if not improved:
                    break
            
            # Recompute final radius
            boundary_limits = np.array([
                min(px, w - px, py, h - py)
                for px, py in positions
            ])
            
            if N > 1:
                pair_dists = cdist(positions, positions)
                np.fill_diagonal(pair_dists, np.inf)
                min_pair_dist = np.min(pair_dists)
            else:
                min_pair_dist = np.inf
            
            r = min(np.min(boundary_limits), min_pair_dist / 2) - 1e-7
            
            if r <= 0:
                continue
            
            total_sum = N * r
            if total_sum > best_sum:
                best_sum = total_sum
                result = np.zeros((N, 3))
                result[:, 0] = positions[:, 0]
                result[:, 1] = positions[:, 1]
                result[:, 2] = r
                best_result = result
    
    if best_result is None:
        # Fallback to seed-like solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
