import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    d = 2
    start_time = time.time()
    max_time = 600.0
    safety_margin = 10.0
    
    def get_score(points):
        # Calculate pairwise distances
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def fast_local_search(points, max_iters=5000):
        """
        Faster local search using coordinate-wise optimization.
        For each point, try moving it in x and y directions with adaptive step sizes.
        """
        best_points = points.copy()
        best_score = get_score(best_points)
        step_x = 0.05
        step_y = 0.05
        stagnant = 0
        
        for i in range(max_iters):
            if time.time() - start_time > max_time - safety_margin:
                break
                
            improved = False
            # Try to improve each point by moving in x or y direction
            for j in range(n):
                # Try x-direction moves
                for dx in [-step_x, step_x]:
                    candidate = best_points.copy()
                    candidate[j, 0] += dx
                    new_score = get_score(candidate)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = candidate
                        improved = True
                
                # Try y-direction moves
                for dy in [-step_y, step_y]:
                    candidate = best_points.copy()
                    candidate[j, 1] += dy
                    new_score = get_score(candidate)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = candidate
                        improved = True
            
            if not improved:
                stagnant += 1
                if stagnant > 5:
                    # Reduce step size
                    step_x *= 0.7
                    step_y *= 0.7
                    stagnant = 0
                    if step_x < 1e-6 and step_y < 1e-6:
                        break
            else:
                stagnant = 0
                # Slightly increase step size if improving
                step_x *= 1.05
                step_y *= 1.05
                # Cap step size
                step_x = min(step_x, 0.5)
                step_y = min(step_y, 0.5)
                
        return best_points, best_score

    # Initial configurations - use known good starting points
    
    configs = []
    
    # 1. 4x4 Grid (standard)
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_points = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    configs.append(grid_points)
    
    # 2. Regular 16-gon
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    poly_points = np.column_stack((np.cos(angles), np.sin(angles)))
    configs.append(poly_points)
    
    # 3. Two concentric squares (8+8)
    # Outer square: 4 corners + 4 midpoints of edges
    # Inner square: 4 corners + 4 midpoints of edges, rotated 45 degrees
    outer = np.array([
        [-1, -1], [1, -1], [1, 1], [-1, 1],  # corners
        [0, -1], [1, 0], [0, 1], [-1, 0]     # midpoints
    ])
    inner_scale = 0.5
    inner = outer * inner_scale
    # Rotate inner by 45 degrees
    theta = np.pi / 4
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    inner = inner @ R.T
    two_squares = np.vstack([outer, inner])
    configs.append(two_squares)
    
    # 4. 4x4 grid with corners pulled in
    grid2 = grid_points.copy()
    # Pull corners inward
    corners = [0, 3, 12, 15]  # indices of corner points in 4x4 grid
    for idx in corners:
        grid2[idx] *= 0.8
    configs.append(grid2)
    
    # 5. Random configuration with good spread
    np.random.seed(42)
    rand_pts = np.random.randn(n, d) * 2
    configs.append(rand_pts)
    
    # 6. Another symmetric config: 4x4 grid rotated 45 degrees
    grid_rot = grid_points.copy()
    theta = np.pi / 4
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    grid_rot = grid_rot @ R.T
    configs.append(grid_rot)
    
    # 7. Hexagonal-like packing (approximate for 16 points)
    # Use a honeycomb pattern
    hex_pts = []
    for row in range(4):
        for col in range(4):
            x = col * 1.0 + (0.5 if row % 2 == 1 else 0)
            y = row * 0.866
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Center it
    hex_pts -= hex_pts.mean(axis=0)
    configs.append(hex_pts)
    
    best_points_global = configs[0]
    best_score_global = 0.0
    
    # Run local search from each configuration
    for config_idx, config in enumerate(configs):
        if time.time() - start_time > max_time - safety_margin:
            break
        
        # Run multiple times with different random seeds for perturbation
        for run in range(2):
            if time.time() - start_time > max_time - safety_margin:
                break
            
            np.random.seed(100 + run + config_idx * 10)
            pts, score = fast_local_search(config.copy())
            
            if score > best_score_global:
                best_score_global = score
                best_points_global = pts.copy()
    
    return best_points_global

# EVOLVE-BLOCK-END
