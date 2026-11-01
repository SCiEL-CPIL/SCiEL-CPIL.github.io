"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = time.time() + 110.0  # Safety margin before 120s limit

def _feasibility_and_value(circles):
    """
    circles: (N, 3) array of (x, y, r)
    Returns (value, feasible)
    value = sum of radii
    feasible = True if all circles are inside a rectangle of perimeter <= 4
              and no two circles overlap.
    """
    if circles.shape[0] != N:
        return 0.0, False
    
    # Compute bounding box
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    if np.any(r <= 0):
        return 0.0, False
    
    min_x = np.min(x - r)
    max_x = np.max(x + r)
    min_y = np.min(y - r)
    max_y = np.max(y + r)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Perimeter constraint
    if 2 * (width + height) > 4.0 + 1e-9:
        return 0.0, False
    
    # Check for overlaps
    # For each pair, distance between centers must be >= sum of radii
    # Use vectorized check for efficiency
    n = N
    # Compute all pairwise distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    sum_r = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Check all pairs i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.any(dist[mask] < sum_r[mask] - 1e-9):
        return 0.0, False
    
    value = np.sum(r)
    return value, True

def _grid_initialization():
    """Create a grid-based initial configuration."""
    # Use a grid that fits 21 circles. 5x5 grid = 25, so use 5x5 and take first 21.
    # Or 4x6 = 24. Let's try 5x5.
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append([x, y, r])
    return np.array(cells[:N])

def _random_initialization():
    """Create a random initial configuration."""
    # Random positions in a square of side 1 (perimeter 4)
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    r = np.random.uniform(0.05, 0.15, N)
    return np.column_stack((x, y, r))

def _optimize_from_initial(init):
    """
    Use scipy minimize to optimize the configuration.
    We maximize sum of radii subject to constraints.
    Since scipy minimizes, we minimize negative sum of radii.
    """
    # Flatten the array
    x0 = init.flatten()
    
    # Objective: minimize -sum(radii)
    def objective(v):
        circles = v.reshape(-1, 3)
        val, feas = _feasibility_and_value(circles)
        if not feas:
            return 1e10
        return -val
    
    # Constraints: 
    # 1. Perimeter <= 4
    # 2. No overlaps
    # 3. Radii > 0
    
    # For simplicity, use a penalty method or just rely on the objective function
    # returning a large value for infeasible points.
    
    # Bounds: x, y in [0, 1], r in [0.01, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.01, 0.5)) # r
    
    # Use L-BFGS-B or SLSQP
    try:
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        return result.x.reshape(-1, 3)
    except Exception:
        return init

def construct():
    start_time = time.time()
    
    # Try multiple initializations and optimizations
    best_circles = None
    best_value = 0.0
    
    # 1. Grid initialization
    init_grid = _grid_initialization()
    val, feas = _feasibility_and_value(init_grid)
    if feas and val > best_value:
        best_value = val
        best_circles = init_grid.copy()
    
    # Optimize from grid
    if time.time() < DEADLINE - 30:
        opt_grid = _optimize_from_initial(init_grid)
        val, feas = _feasibility_and_value(opt_grid)
        if feas and val > best_value:
            best_value = val
            best_circles = opt_grid.copy()
    
    # 2. Random initializations
    for attempt in range(10):
        if time.time() > DEADLINE - 30:
            break
        
        init_rand = _random_initialization()
        val, feas = _feasibility_and_value(init_rand)
        if feas and val > best_value:
            best_value = val
            best_circles = init_rand.copy()
        
        # Optimize from random
        if time.time() < DEADLINE - 30:
            opt_rand = _optimize_from_initial(init_rand)
            val, feas = _feasibility_and_value(opt_rand)
            if feas and val > best_value:
                best_value = val
                best_circles = opt_rand.copy()
    
    if best_circles is None:
        # Fallback to grid
        best_circles = _grid_initialization()
    
    return best_circles
