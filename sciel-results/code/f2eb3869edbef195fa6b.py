"""
Rewritten program using a smooth max-min optimization approach for circle packing.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _get_initial_guess_hex():
    """
    Generate an initial guess based on a hexagonal lattice.
    """
    # We want to pack N circles.
    # Approximate number of rows
    rows = int(np.ceil(np.sqrt(N)))
    cols = int(np.ceil(N / rows))
    
    # Create a hexagonal grid
    # Spacing is initially 1.0 / cols
    dx = 1.0 / cols
    dy = dx * np.sqrt(3) / 2
    
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= N:
                break
            x = (j + 0.5) * dx - 0.5
            y = (i + 0.5) * dy - 0.5
            # Shift even rows by dx/2 to make it hexagonal
            if i % 2 == 1:
                x += dx / 2
            points.append([x, y])
    
    # If we have too few, add some random points or adjust grid
    # If we have too many, truncate
    if len(points) < N:
        # Fill remaining with random points in the square
        for _ in range(N - len(points)):
            points.append([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)])
    else:
        points = points[:N]
        
    return np.array(points)

def _get_initial_guess_grid():
    """
    Generate an initial guess based on a square grid.
    """
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _soft_min_dist(pts, beta=20.0):
    """
    Compute the soft minimum of all pairwise distances and wall distances.
    Returns a scalar value. Lower is better (we will maximize the negative).
    """
    pts = np.array(pts).reshape(-1, 2)
    n = len(pts)
    
    # Pairwise distances
    # d[i,j] = ||p_i - p_j||
    # We want min_{i<j} d[i,j]
    
    # Efficient computation of all pairwise distances
    # Using broadcasting
    # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=2)
    
    # To save memory and time, we can do it in chunks or just use scipy
    from scipy.spatial.distance import pdist
    dists = pdist(pts, 'euclidean')
    
    # Wall distances
    # dist to wall = min(x - (-0.5), 0.5 - x, y - (-0.5), 0.5 - y)
    # = min(0.5 + x, 0.5 - x, 0.5 + y, 0.5 - y)
    wall_dists = np.minimum.reduce([
        0.5 + pts[:, 0],
        0.5 - pts[:, 0],
        0.5 + pts[:, 1],
        0.5 - pts[:, 1]
    ])
    
    # Combine all distances
    all_dists = np.concatenate([dists, wall_dists])
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    # To avoid overflow, we can shift by min
    min_d = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - min_d))
    soft_min_val = min_d - (1.0 / beta) * np.log(np.sum(exp_terms))
    
    return -soft_min_val # We want to minimize this, so we return negative

def _soft_min_dist_grad(pts, beta=20.0):
    """
    Compute the gradient of the soft min distance.
    """
    pts = np.array(pts).reshape(-1, 2)
    n = len(pts)
    
    # This is complex to derive analytically for all pairs.
    # For simplicity and robustness within time limits, 
    # we might use a numerical gradient or a simpler optimizer like Powell.
    # However, L-BFGS-B with numerical gradient is often sufficient for 100 variables.
    # Let's rely on L-BFGS-B's internal numerical gradient or use Powell.
    # Powell is derivative-free and handles non-smoothness better if beta is high.
    # Let's switch to Powell to avoid gradient implementation errors and overhead.
    pass

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    best_pts = None
    best_score = -np.inf
    
    # Try multiple initializations
    inits = [
        _get_initial_guess_hex(),
        _get_initial_guess_grid()
    ]
    
    # Add some random initializations
    for _ in range(3):
        inits.append(np.random.uniform(-0.4, 0.4, (N, 2)))
        
    beta = 30.0
    
    for i, init in enumerate(inits):
        if time.time() - start_time > deadline:
            break
            
        # Use Powell method which is derivative-free and robust
        # Objective: minimize -soft_min_dist
        # We can just use the raw min distance with a penalty for smoothness if needed,
        # but Powell handles non-smooth min() well if we just define it.
        
        def objective(x):
            pts = x.reshape(-1, 2)
            # Calculate min distance
            # Pairwise
            from scipy.spatial.distance import pdist
            dists = pdist(pts, 'euclidean')
            # Wall
            wall_dists = np.minimum.reduce([
                0.5 + pts[:, 0],
                0.5 - pts[:, 0],
                0.5 + pts[:, 1],
                0.5 - pts[:, 1]
            ])
            min_d = np.min(np.concatenate([dists, wall_dists]))
            return -min_d
        
        try:
            res = minimize(objective, init, method='Powell', 
                           options={'maxiter': 1000, 'maxfev': 5000})
            
            if res.success or res.fun < best_score:
                best_score = res.fun
                best_pts = res.x.reshape(-1, 2)
                
        except Exception as e:
            # If optimization fails, keep the initial guess if it's better
            pass
            
    # If we didn't find any, return the best initial guess
    if best_pts is None:
        best_pts = inits[0]
        
    return best_pts
