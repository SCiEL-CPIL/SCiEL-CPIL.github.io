"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _objective(params, N):
    """
    Minimize negative sum of radii.
    params: flattened array of [x1, y1, r1, x2, y2, r2, ...]
    Constraints are handled via penalty in the objective for simplicity in the inner loop,
    but we also use bounds.
    """
    pts = params.reshape(N, 3)
    x, y, r = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Objective: Maximize sum of radii -> Minimize -sum(r)
    obj = -np.sum(r)
    
    # Penalties
    # 1. Inside square: x >= r, x <= 1-r, y >= r, y <= 1-r
    # We use a soft penalty to allow the optimizer to navigate, but bounds in minimize help.
    # However, for robustness in a custom loop, we add penalties.
    pen = 0.0
    # Boundary violations
    pen += np.sum(np.maximum(0, r - x)**2)
    pen += np.sum(np.maximum(0, r - y)**2)
    pen += np.sum(np.maximum(0, x - (1 - r))**2)
    pen += np.sum(np.maximum(0, y - (1 - r))**2)
    
    # 2. Overlap violations: dist(i,j) >= ri + rj
    # Vectorized distance calculation
    # dx = x[:, None] - x[None, :]
    # This is O(N^2) memory, N=26 is small (676 floats), so it's fine.
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    min_dist = r[:, None] + r[None, :]
    # dist < min_dist is violation
    # penalty = sum(max(0, min_dist - dist)^2)
    viol = np.maximum(0, min_dist - dist)
    # Zero out diagonal
    np.fill_diagonal(viol, 0)
    pen += np.sum(viol**2)
    
    return obj + 100.0 * pen

def _is_valid(pts, N):
    """Check if a configuration is strictly valid (disjoint, inside square)."""
    x, y, r = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Boundary check
    if np.any(x < r - 1e-5) or np.any(y < r - 1e-5):
        return False
    if np.any(x > 1 - r + 1e-5) or np.any(y > 1 - r + 1e-5):
        return False
        
    # Radius check
    if np.any(r < 1e-6):
        return False

    # Overlap check
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    
    # Check all pairs i < j
    for i in range(N):
        for j in range(i + 1, N):
            if dist[i, j] < min_dist[i, j] - 1e-5:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin for 120s limit
    
    best_score = -np.inf
    best_pts = None
    
    # Initial configurations
    # 1. Grid
    side = int(np.ceil(np.sqrt(N)))
    grid_pts = np.zeros((N, 3))
    for i in range(N):
        row = i // side
        col = i % side
        grid_pts[i, 0] = (col + 0.5) / side
        grid_pts[i, 1] = (row + 0.5) / side
        grid_pts[i, 2] = 0.5 / side * 0.9 # Slightly smaller to start
    
    # 2. Random scatter
    np.random.seed(42)
    random_pts = np.random.rand(N, 3)
    random_pts[:, 0] = np.random.uniform(0.1, 0.9, N)
    random_pts[:, 1] = np.random.uniform(0.1, 0.9, N)
    random_pts[:, 2] = np.random.uniform(0.05, 0.1, N)
    
    # 3. "Wall" bias: Place some circles near corners/edges
    wall_pts = np.zeros((N, 3))
    # 4 corners
    wall_pts[0, :] = [0.1, 0.1, 0.1]
    wall_pts[1, :] = [0.9, 0.1, 0.1]
    wall_pts[2, :] = [0.1, 0.9, 0.1]
    wall_pts[3, :] = [0.9, 0.9, 0.1]
    # Rest random
    wall_pts[4:, 0] = np.random.uniform(0.1, 0.9, N-4)
    wall_pts[4:, 1] = np.random.uniform(0.1, 0.9, N-4)
    wall_pts[4:, 2] = np.random.uniform(0.05, 0.1, N-4)
    
    initial_configs = [grid_pts, random_pts, wall_pts]
    
    # Multi-start loop
    for start_idx in range(3):
        if time.time() > deadline:
            break
            
        current_pts = initial_configs[start_idx].copy()
        
        # Flatten for optimizer
        x0 = current_pts.flatten()
        
        # Bounds: x, y in [0.01, 0.99], r in [0.01, 0.5]
        # Note: r cannot be 0.5 because 2 circles of r=0.5 don't fit.
        # Max r is roughly 0.25 for 4 circles. For 26, avg r ~ 0.1.
        # Let's set bounds loosely, penalty handles the rest.
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99)) # x
            bounds.append((0.01, 0.99)) # y
            bounds.append((0.01, 0.4))  # r
        
        # Optimization settings
        # Use L-BFGS-B, it's robust and fast for this size.
        # We will run it for a fixed number of iterations or until convergence.
        try:
            res = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            opt_pts = res.x.reshape(N, 3)
            
            # Post-process: Ensure validity
            # If invalid, try to shrink radii slightly to fix overlaps?
            # Or just check validity.
            if _is_valid(opt_pts, N):
                score = np.sum(opt_pts[:, 2])
                if score > best_score:
                    best_score = score
                    best_pts = opt_pts.copy()
            else:
                # If invalid, try a simple relaxation to fix it
                # Reduce radii by 10% and re-check?
                # Or just skip.
                pass
                
        except Exception as e:
            pass

        # If we have time, do a second pass with a different starting point derived from the best found so far?
        # Or just stick to the 3 predefined starts for now to ensure we finish in time.
        # 200 iterations of L-BFGS-B on N=26 is very fast (<1s).
        # We can afford more restarts.
        
    # If we still have time, do random restarts
    restart_count = 0
    while time.time() < deadline - 5.0:
        # Generate a new random start
        new_pts = np.random.rand(N, 3)
        new_pts[:, 0] = np.random.uniform(0.1, 0.9, N)
        new_pts[:, 1] = np.random.uniform(0.1, 0.9, N)
        new_pts[:, 2] = np.random.uniform(0.05, 0.15, N)
        
        x0 = new_pts.flatten()
        try:
            res = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            opt_pts = res.x.reshape(N, 3)
            
            if _is_valid(opt_pts, N):
                score = np.sum(opt_pts[:, 2])
                if score > best_score:
                    best_score = score
                    best_pts = opt_pts.copy()
        except Exception:
            pass
        restart_count += 1
        # Limit restarts to avoid infinite loop if time check is broken
        if restart_count > 100:
            break

    # If no valid solution was found (shouldn't happen with grid), return grid
    if best_pts is None:
        return grid_pts
    
    return best_pts

# EVOLVE-BLOCK-END
