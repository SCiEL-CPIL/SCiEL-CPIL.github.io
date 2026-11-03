import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
TIME_LIMIT = 115.0  # Leave 5s margin for the 120s limit
START_TIME = time.time()

def compute_radius(centers):
    """Compute the radius of the best packing given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers[:, 0] < -0.5) or np.any(centers[:, 0] > 0.5) or np.any(centers[:, 1] < -0.5) or np.any(centers[:, 1] > 0.5):
        return 0.0
    # distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # wall distances
    wall_dists = np.min(np.stack([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ]), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_dist / 2.0, min_wall_dist)

def objective(flat):
    """Objective function: minimize sum of inverse squared distances (pair + wall)."""
    x = flat.reshape(N, 2)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-10)
    E_pair = np.sum(1.0 / d2)
    
    dx = 0.5 - np.abs(x[:, 0])
    dy = 0.5 - np.abs(x[:, 1])
    d_left = 0.5 + x[:, 0]
    d_right = 0.5 - x[:, 0]
    d_down = 0.5 + x[:, 1]
    d_up = 0.5 - x[:, 1]
    
    d_left = np.maximum(d_left, 1e-6)
    d_right = np.maximum(d_right, 1e-6)
    d_down = np.maximum(d_down, 1e-6)
    d_up = np.maximum(d_up, 1e-6)
    
    E_wall = np.sum(1.0/d_left**2 + 1.0/d_right**2 + 1.0/d_down**2 + 1.0/d_up**2)
    
    return E_pair + E_wall

def initial_hex_grid():
    """Generate a hexagonal grid initialization."""
    # Target radius ~ 0.0366
    # Hexagonal spacing: dx = 2r, dy = 2r * sqrt(3)/2 = r*sqrt(3)
    # For 200 circles, roughly 14x14 grid with hex offset
    r_est = 0.0366
    dx = 2 * r_est
    dy = r_est * np.sqrt(3)
    
    points = []
    # Generate hex grid
    for j in range(20):
        for i in range(20):
            x = (i - 10) * dx + (0.5 * dx if j % 2 else 0)
            y = (j - 10) * dy
            if -0.49 <= x <= 0.49 and -0.49 <= y <= 0.49:
                points.append((x, y))
    
    points = np.array(points)
    if len(points) < N:
        # Fill remaining with square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = np.vstack([points, grid[len(points):N]])
    else:
        points = points[:N]
    
    return points

def initial_square_grid():
    """Generate a square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def initial_random():
    """Generate a random initialization."""
    return np.random.uniform(-0.45, 0.45, size=(N, 2))

def optimize_from(x0, method='L-BFGS-B', maxiter=500):
    """Optimize from initial point x0."""
    bounds = [(-0.5, 0.5)] * (2 * N)
    res = minimize(objective, x0.flatten(), method=method, bounds=bounds,
                   options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
    return res.x.reshape(N, 2)

def construct():
    start_time = time.time()
    best_r = 0
    best_x = None
    
    # Strategy 1: Hex grid + optimization
    x0_hex = initial_hex_grid()
    x_opt = optimize_from(x0_hex, method='L-BFGS-B', maxiter=300)
    r = compute_radius(x_opt)
    if r > best_r:
        best_r = r
        best_x = x_opt.copy()
    
    if time.time() - start_time > TIME_LIMIT:
        return best_x
    
    # Strategy 2: Square grid + optimization
    x0_sq = initial_square_grid()
    x_opt = optimize_from(x0_sq, method='L-BFGS-B', maxiter=300)
    r = compute_radius(x_opt)
    if r > best_r:
        best_r = r
        best_x = x_opt.copy()
    
    if time.time() - start_time > TIME_LIMIT:
        return best_x
    
    # Strategy 3: Multiple random restarts
    for i in range(10):
        if time.time() - start_time > TIME_LIMIT:
            break
        
        np.random.seed(i + 100)
        x0_rand = initial_random()
        x_opt = optimize_from(x0_rand, method='L-BFGS-B', maxiter=200)
        r = compute_radius(x_opt)
        if r > best_r:
            best_r = r
            best_x = x_opt.copy()
    
    # Strategy 4: Refine best solution with more iterations
    if time.time() - start_time < TIME_LIMIT - 10:
        x_opt = optimize_from(best_x, method='L-BFGS-B', maxiter=500)
        r = compute_radius(x_opt)
        if r > best_r:
            best_r = r
            best_x = x_opt.copy()
    
    return best_x
