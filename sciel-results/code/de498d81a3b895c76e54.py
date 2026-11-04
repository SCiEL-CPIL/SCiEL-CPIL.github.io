"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4, maximizing sum of radii.
    
    Approach: Use a heuristic based on optimizing positions and radii.
    We parameterize the rectangle by width w and height h with 2(w+h)=4, so w+h=2.
    We place circles in a grid-like pattern and then optimize.
    
    Since SDP is complex to implement robustly in 120s, we use a strong local search
    with simulated annealing or gradient-based optimization on the positions and radii,
    ensuring non-overlap and boundary constraints.
    """
    # Perimeter 4 means 2*(w+h) = 4 => w+h = 2
    # We'll search over w in (0, 2), h = 2 - w
    # For a given w, h, we need to place 21 circles.
    
    # A good starting point: assume roughly square-ish rectangle
    # Let's try to optimize w and the circle positions/radii together.
    
    # For simplicity and robustness, let's use a grid-based initial guess and then
    # do local optimization with constraints handled via penalty or projection.
    
    # Actually, let's think about this differently. The objective is sum of radii.
    # Constraints: circles don't overlap, circles inside rectangle.
    
    # Let's use a different approach: fix the rectangle aspect ratio and optimize
    # the packing within it, then try several aspect ratios.
    
    best_value = 0
    best_result = None
    
    # Try several aspect ratios
    # w + h = 2, so let w = 2 * t, h = 2 * (1-t) for t in (0,1)
    # But we can also just try a few specific values.
    
    # For each aspect ratio, we'll use a grid initialization and then optimize.
    
    def pack_for_aspect(t, seed=42):
        """
        Try to pack 21 circles in rectangle of width 2t and height 2(1-t).
        Returns (value, circles_array) or None if failed.
        """
        w = 2 * t
        h = 2 * (1 - t)
        
        if w <= 0 or h <= 0:
            return None
        
        # Initialize with a grid pattern
        # 21 circles: try 5x5 grid minus 4 corners, or 7x3, etc.
        # Let's try a few grid configurations
        
        best_local_value = 0
        best_local_circles = None
        
        # Try grid configurations: (rows, cols) such that rows*cols >= 21
        for rows in range(1, 10):
            for cols in range(1, 10):
                if rows * cols < N:
                    continue
                if rows * cols > 40:  # Too many cells
                    continue
                
                # Create grid positions
                x_coords = np.linspace(0, w, cols + 2)[1:-1]  # Leave margin
                y_coords = np.linspace(0, h, rows + 2)[1:-1]
                
                # Generate all grid points
                grid_points = []
                for i in range(rows):
                    for j in range(cols):
                        grid_points.append([x_coords[j], y_coords[i]])
                
                grid_points = np.array(grid_points[:N])
                
                # Initial radii: half the minimum spacing
                if len(grid_points) < 2:
                    continue
                
                # Compute minimum distance between any two points
                dists = cdist(grid_points, grid_points)
                np.fill_diagonal(dists, np.inf)
                min_dist = np.min(dists)
                
                # Also consider distance to boundaries
                dist_to_boundaries = np.min([
                    grid_points[:, 0],
                    w - grid_points[:, 0],
                    grid_points[:, 1],
                    h - grid_points[:, 1]
                ], axis=0)
                
                # Initial radius: half of min(min_dist, 2*dist_to_boundaries)
                # Actually, radius must be <= dist_to_boundary and <= dist/2 to other circles
                r_init = np.minimum(0.5 * min_dist, dist_to_boundaries)
                r_init = np.maximum(r_init, 1e-4)
                
                # Now optimize positions and radii
                # Variable: x_0, y_0, r_0, x_1, y_1, r_1, ...
                def objective(vars):
                    pos = vars[:2*N].reshape(N, 2)
                    radii = vars[2*N:]
                    
                    # Penalty for out of bounds
                    penalty = 0
                    for i in range(N):
                        if pos[i, 0] < 0 or pos[i, 0] > w or pos[i, 1] < 0 or pos[i, 1] > h:
                            penalty += 1000 * (abs(pos[i, 0]) + abs(pos[i, 0] - w) + 
                                             abs(pos[i, 1]) + abs(pos[i, 1] - h))
                        # Radius too large for boundary
                        if radii[i] > pos[i, 0]:
                            penalty += 100 * (radii[i] - pos[i, 0])
                        if radii[i] > w - pos[i, 0]:
                            penalty += 100 * (radii[i] - (w - pos[i, 0]))
                        if radii[i] > pos[i, 1]:
                            penalty += 100 * (radii[i] - pos[i, 1])
                        if radii[i] > h - pos[i, 1]:
                            penalty += 100 * (radii[i] - (h - pos[i, 1]))
                    
                    # Penalty for overlap
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt(np.sum((pos[i] - pos[j])**2))
                            if dist < radii[i] + radii[j]:
                                penalty += 100 * (radii[i] + radii[j] - dist)
                    
                    # Objective: maximize sum of radii, so minimize negative
                    return -np.sum(radii) + penalty
                
                # Initial variables
                x0 = np.concatenate([grid_points.flatten(), r_init])
                
                # Bounds
                bounds = []
                for i in range(N):
                    bounds.append((0, w))  # x
                    bounds.append((0, h))  # y
                    bounds.append((1e-4, min(w, h)/2))  # r
                
                try:
                    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                                     options={'maxiter': 200, 'ftol': 1e-6})
                    
                    if result.success or result.fun < 1e-3:
                        pos = result.x[:2*N].reshape(N, 2)
                        radii = result.x[2*N:]
                        
                        # Verify constraints
                        valid = True
                        for i in range(N):
                            if pos[i, 0] < -1e-6 or pos[i, 0] > w + 1e-6 or \
                               pos[i, 1] < -1e-6 or pos[i, 1] > h + 1e-6 or \
                               radii[i] < 1e-6:
                                valid = False
                                break
                            if radii[i] > pos[i, 0] + 1e-6 or radii[i] > w - pos[i, 0] + 1e-6 or \
                               radii[i] > pos[i, 1] + 1e-6 or radii[i] > h - pos[i, 1] + 1e-6:
                                valid = False
                                break
                        
                        if valid:
                            for i in range(N):
                                for j in range(i+1, N):
                                    dist = np.sqrt(np.sum((pos[i] - pos[j])**2))
                                    if dist < radii[i] + radii[j] - 1e-6:
                                        valid = False
                                        break
                                if not valid:
                                    break
                        
                        if valid:
                            value = np.sum(radii)
                            if value > best_local_value:
                                best_local_value = value
                                best_local_circles = np.column_stack([pos, radii])
                
                except Exception:
                    continue
        
        return best_local_value, best_local_circles
    
    # Try several aspect ratios
    for t in [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]:
        result = pack_for_aspect(t)
        if result is not None:
            value, circles = result
            if circles is not None and value > best_value:
                best_value = value
                best_result = circles
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
