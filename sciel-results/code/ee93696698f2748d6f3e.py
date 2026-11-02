"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    best_score = -np.inf
    best_result = None
    
    # Define the objective function for scipy.optimize
    def objective(params):
        # params shape: (N, 3) -> x, y, r
        x = params[:, 0]
        y = params[:, 1]
        r = params[:, 2]
        
        # Sum of radii to maximize (so minimize negative)
        score = np.sum(r)
        
        # Boundary penalties
        # x - r >= 0, 1 - x - r >= 0, etc.
        # Violation is max(0, lower - val)^2
        pen_x_low = np.sum(np.maximum(0, 0 - (x - r))**2)
        pen_x_high = np.sum(np.maximum(0, 0 - (1 - x - r))**2)
        pen_y_low = np.sum(np.maximum(0, 0 - (y - r))**2)
        pen_y_high = np.sum(np.maximum(0, 0 - (1 - y - r))**2)
        
        # Collision penalties
        # d_ij >= r_i + r_j
        # Violation is max(0, r_i + r_j - d_ij)^2
        collision_pen = 0.0
        # Efficient distance calculation
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        d = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        violations = np.maximum(0, r_sum - d)
        collision_pen = np.sum(violations[mask]**2)
        
        total_penalty = pen_x_low + pen_x_high + pen_y_low + pen_y_high + collision_pen
        
        # Penalty weight
        # We want the penalty to dominate if constraints are violated.
        # Since r is around 0.1-0.2, r^2 is 0.01-0.04.
        # A weight of 100-1000 should be sufficient.
        penalty_weight = 100.0
        
        return -score + penalty_weight * total_penalty

    def run_optimization(initial_params):
        # Bounds: x,y in [0,1], r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
            
        # Use L-BFGS-B for speed
        res = minimize(
            objective, 
            initial_params, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        # Check if result is valid
        x = res.x.reshape(N, 3)[:, 0]
        y = res.x.reshape(N, 3)[:, 1]
        r = res.x.reshape(N, 3)[:, 2]
        
        # Verify constraints strictly
        valid = True
        # Boundary check
        if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
            valid = False
        
        # Collision check
        if valid:
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            d = np.sqrt(dx**2 + dy**2)
            r_sum = r[:, np.newaxis] + r[np.newaxis, :]
            mask = ~np.eye(N, dtype=bool)
            if np.any(r_sum[mask] > d[mask] + 1e-9):
                valid = False
        
        if valid:
            return np.sum(r), x, y, r
        else:
            # If invalid, try to project or just return low score
            # For now, return -inf
            return -np.inf, None, None, None

    # Strategy: Multi-start with diverse initializations
    
    # 1. Hexagonal Grid Initialization
    def init_hex():
        # Arrange in hex grid
        # 26 circles. Maybe 5 rows?
        # Row 0: 5, Row 1: 5, Row 2: 5, Row 3: 5, Row 4: 5, Row 5: 1?
        # Let's just place them on a grid and let the optimizer fix it.
        # A 5x5 grid is 25. Add 1 in the middle?
        # Or 4x7 = 28, remove 2?
        
        # Let's try a 5x5 grid + 1 center
        params = np.zeros((N, 3))
        
        # 5x5 grid
        xs = np.linspace(0.1, 0.9, 5)
        ys = np.linspace(0.1, 0.9, 5)
        
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < N:
                    params[idx, 0] = xs[i]
                    params[idx, 1] = ys[j]
                    params[idx, 2] = 0.1
                    idx += 1
        
        # If we have more, add them randomly or adjust
        if idx < N:
             # Add remaining
             for k in range(idx, N):
                 params[k, 0] = np.random.uniform(0.1, 0.9)
                 params[k, 1] = np.random.uniform(0.1, 0.9)
                 params[k, 2] = 0.1
                 
        return params

    # 2. Random Initialization
    def init_random(seed):
        np.random.seed(seed)
        params = np.zeros((N, 3))
        params[:, 0] = np.random.uniform(0.1, 0.9, N)
        params[:, 1] = np.random.uniform(0.1, 0.9, N)
        params[:, 2] = 0.1
        return params

    # 3. Grid Initialization (Square)
    def init_grid():
        params = np.zeros((N, 3))
        # 5x6 grid = 30 spots, take 26
        xs = np.linspace(0.1, 0.9, 5)
        ys = np.linspace(0.1, 0.9, 6)
        idx = 0
        for i in range(5):
            for j in range(6):
                if idx < N:
                    params[idx, 0] = xs[i]
                    params[idx, 1] = ys[j]
                    params[idx, 2] = 0.1
                    idx += 1
        return params

    inits = [
        init_hex(),
        init_grid(),
        init_random(1),
        init_random(2),
        init_random(3)
    ]
    
    for init in inits:
        score, x, y, r = run_optimization(init)
        if score > best_score:
            best_score = score
            best_result = np.column_stack((x, y, r))
            
    # Fallback if all failed (should not happen)
    if best_result is None:
        # Return a valid trivial solution
        best_result = np.zeros((N, 3))
        # Place them in a line? No, just small circles in corners?
        # Just return the hex init with small radii
        best_result = init_hex()
        best_result[:, 2] = 0.01
        
    return best_result
# EVOLVE-BLOCK-END
