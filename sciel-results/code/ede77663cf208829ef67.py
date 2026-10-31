import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    timeout = 115  # Leave 5s margin for 120s limit
    
    # Objective function to minimize: -sum(r) + penalty
    def objective(params):
        # Unpack params: x_0...x_{N-1}, y_0...y_{N-1}, r_0...r_{N-1}
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Boundary penalties: circles must be inside [0,1]x[0,1]
        # x in [r, 1-r], y in [r, 1-r]
        bound_pen = 0.0
        # Left/Right
        bound_pen += np.sum(np.maximum(0, r - x)**2)
        bound_pen += np.sum(np.maximum(0, x - (1 - r))**2)
        # Bottom/Top
        bound_pen += np.sum(np.maximum(0, r - y)**2)
        bound_pen += np.sum(np.maximum(0, y - (1 - r))**2)
        
        # Overlap penalties: d_ij >= r_i + r_j
        # Vectorized calculation of pairwise distances
        # d_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
        # We only need i < j
        x_diff = np.subtract.outer(x, x)
        y_diff = np.subtract.outer(y, y)
        d_sq = x_diff**2 + y_diff**2
        # Set diagonal to inf to ignore self-overlap
        np.fill_diagonal(d_sq, np.inf)
        d = np.sqrt(d_sq)
        
        r_sum = np.add.outer(r, r)
        
        # Overlap amount: max(0, r_i + r_j - d_ij)
        overlap = np.maximum(0, r_sum - d)
        # Square the overlap to make penalty smooth-ish and emphasize large overlaps
        overlap_pen = np.sum(overlap**2)
        
        total_pen = bound_pen + overlap_pen
        
        # Maximize sum of radii -> Minimize -sum(r)
        # Weight the penalty heavily
        lambda_pen = 100.0
        return -np.sum(r) + lambda_pen * total_pen

    def random_init():
        # Random positions in [0.1, 0.9], radii in [0.05, 0.15]
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        return np.concatenate([x, y, r])

    def grid_init():
        # 5x5 grid is 25 circles. Add one in the center or shift.
        # Let's use a 5x5 grid of radius 0.1, and one small circle in the middle.
        # Actually, just initialize 26 circles in a jittered grid.
        side = 5
        coords = []
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx < N:
                    x = (i + 0.5) / side + np.random.uniform(-0.05, 0.05)
                    y = (j + 0.5) / side + np.random.uniform(-0.05, 0.05)
                    r = 0.1
                    coords.append([x, y, r])
                    idx += 1
        # If N > 25, add extra circles randomly or in gaps
        while idx < N:
            x = np.random.uniform(0.2, 0.8)
            y = np.random.uniform(0.2, 0.8)
            r = 0.05
            coords.append([x, y, r])
            idx += 1
            
        arr = np.array(coords)
        # Ensure within bounds roughly
        arr[:, 0] = np.clip(arr[:, 0], 0.1, 0.9)
        arr[:, 1] = np.clip(arr[:, 1], 0.1, 0.9)
        return np.concatenate([arr[:, 0], arr[:, 1], arr[:, 2]])

    def hex_init():
        # Hexagonal lattice initialization
        r = 0.1
        h = np.sqrt(3) * r
        w = 2 * r
        coords = []
        row = 0
        y = r + 0.1
        while y < 1 - r and len(coords) < N:
            x = r + 0.1
            offset = 0
            if row % 2 == 1:
                offset = w / 2
            while x < 1 - r and len(coords) < N:
                coords.append([x + offset, y, r])
                x += w
            row += 1
            y += h
        # Pad with random if needed
        while len(coords) < N:
            coords.append([np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8), 0.05])
        arr = np.array(coords)
        return np.concatenate([arr[:, 0], arr[:, 1], arr[:, 2]])

    initializations = [grid_init(), hex_init(), random_init(), random_init()]
    
    best_val = -np.inf
    best_params = None
    
    # Run optimization from each initialization
    for i, x0 in enumerate(initializations):
        if time.time() - start_time > timeout:
            break
            
        try:
            # Nelder-Mead is robust for non-smooth penalties
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception as e:
            pass

    if best_params is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    # Post-processing: Ensure validity
    # If any overlap or boundary violation remains, shrink radii slightly
    # We check constraints and scale radii down if necessary
    max_iter = 10
    for _ in range(max_iter):
        # Check boundary
        min_x = np.min(x - r)
        max_x = np.max(x + r)
        min_y = np.min(y - r)
        max_y = np.max(y + r)
        
        # Check overlap
        x_diff = np.subtract.outer(x, x)
        y_diff = np.subtract.outer(y, y)
        d_sq = x_diff**2 + y_diff**2
        np.fill_diagonal(d_sq, np.inf)
        d = np.sqrt(d_sq)
        r_sum = np.add.outer(r, r)
        
        # Find max violation
        # Boundary violation: we want r <= min(x, 1-x, y, 1-y)
        # Overlap violation: we want r_i + r_j <= d_ij
        # Simple fix: reduce all radii by a small factor if violation exists
        violation = 0.0
        if min_x < 0 or max_x > 1 or min_y < 0 or max_y > 1:
            # Calculate required shrinkage for boundaries
            # This is an approximation, just shrink slightly
            violation = 0.01
            
        # Check overlaps
        overlaps = np.maximum(0, r_sum - d)
        max_overlap = np.max(overlaps)
        if max_overlap > 1e-6:
            violation = max_overlap
            
        if violation > 1e-6:
            # Shrink radii by 1% or enough to fix
            shrink_factor = 0.99
            r = r * shrink_factor
        else:
            break
            
    # Final clip to ensure strictly inside
    r = np.maximum(r, 1e-6)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    # One last check to ensure no overlaps after clipping (clipping can cause overlaps)
    # If overlaps exist, shrink again
    for _ in range(5):
        x_diff = np.subtract.outer(x, x)
        y_diff = np.subtract.outer(y, y)
        d_sq = x_diff**2 + y_diff**2
        np.fill_diagonal(d_sq, np.inf)
        d = np.sqrt(d_sq)
        r_sum = np.add.outer(r, r)
        overlaps = np.maximum(0, r_sum - d)
        if np.max(overlaps) < 1e-6:
            break
        r = r * 0.99
        
    return np.column_stack((x, y, r))
