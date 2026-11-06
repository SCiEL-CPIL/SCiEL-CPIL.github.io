"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE_SECONDS = 110.0  # Safety margin before 120s limit

def compute_radius(pts):
    """
    Computes the maximum radius for the given centers.
    pts: (N, 2) array of centers.
    Returns r such that circles of radius r are disjoint and inside [-0.5, 0.5]^2.
    """
    # 1. Distance to boundary
    # d_x = 0.5 - |x|, d_y = 0.5 - |y|
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    min_bound = np.min(np.minimum(d_x, d_y))
    
    # If any point is outside, min_bound is negative.
    # The radius must be non-negative.
    # The formula is min(min_dist_inter/2, min_bound).
    
    # 2. Distance between centers
    # Use broadcasting for N=50 (50x50 matrix)
    # pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_inter = np.sqrt(np.min(dist_sq))
    
    r_inter = min_dist_inter / 2.0
    r = np.minimum(r_inter, min_bound)
    
    return r

def objective(x):
    """
    Objective function to minimize: -radius.
    """
    pts = x.reshape(-1, 2)
    r = compute_radius(pts)
    return -r

def generate_hex_initialization(spacing=1.0/7.0):
    """
    Generates a hexagonal lattice initialization.
    """
    points = []
    # Generate a large enough grid to cover the square
    # Square is 1x1. Spacing is s.
    # Need about 1/s points in each direction.
    # Add margin.
    n_range = int(np.ceil(1.5 / spacing))
    
    for j in range(-n_range, n_range + 1):
        for i in range(-n_range, n_range + 1):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * (np.sqrt(3) / 2.0)
            points.append([x, y])
            
    points = np.array(points)
    
    # Center the points
    centroid = np.mean(points, axis=0)
    points -= centroid
    
    # Filter points inside the square [-0.5, 0.5]
    # We want them to be strictly inside to avoid boundary issues initially
    # But for initialization, being on the edge is okay if we optimize.
    # Let's filter by |x| < 0.5 and |y| < 0.5
    mask = (np.abs(points[:, 0]) < 0.5) & (np.abs(points[:, 1]) < 0.5)
    points = points[mask]
    
    if len(points) > N:
        # Randomly select N points
        # To be deterministic, we could sort by distance to center and pick closest?
        # Or just random. Random is fine for restarts.
        indices = np.random.choice(len(points), N, replace=False)
        points = points[indices]
    elif len(points) < N:
        # Pad with random points
        num_pad = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, size=(num_pad, 2))
        points = np.vstack([points, random_pts])
        
    return points

def generate_grid_initialization():
    """
    Generates a 7x7 grid + 1 point initialization.
    """
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # Add 1 point. Where?
    # The center of the square is (0,0).
    # The grid point closest to (0,0) is (0,0) if m is odd.
    # For m=7, ticks = [-3/7, -2/7, -1/7, 0, 1/7, 2/7, 3/7].
    # So (0,0) is in the grid.
    # We need to add a point that is far from others.
    # Maybe (0, 1/14)? Or just random.
    # Let's add a point at a "hole" location, e.g., between 4 grid points.
    # Hole at (1/14, 1/14)?
    # Distance to (0,0) is sqrt(2)/14 ~ 0.1.
    # Required dist 2r ~ 0.14.
    # So it will overlap.
    # The optimizer will fix it.
    new_pt = np.array([0.0, 1.0/14.0])
    grid = np.vstack([grid, new_pt])
    
    # Shuffle to break symmetry?
    # np.random.shuffle(grid)
    
    return grid

def optimize_from_start(start_pts, max_time_per_run=20.0):
    """
    Runs local optimization starting from start_pts.
    Returns the best points found.
    """
    x0 = start_pts.flatten()
    
    # Use Powell method which is robust for non-smooth functions
    # maxiter is a heuristic. We also check time.
    result = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'ftol': 1e-8, 'xatol': 1e-8})
    
    best_pts = result.x.reshape(-1, 2)
    # Clip to ensure validity
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

def construct():
    start_time = time.time()
    
    # List of initializations
    inits = []
    
    # 1. Hexagonal
    inits.append(generate_hex_initialization())
    
    # 2. Grid
    inits.append(generate_grid_initialization())
    
    # 3. Random
    inits.append(np.random.uniform(-0.4, 0.4, size=(N, 2)))
    
    # 4. Another Hex with slightly different spacing
    inits.append(generate_hex_initialization(spacing=0.14))
    
    best_pts = inits[0]
    best_r = compute_radius(best_pts)
    
    for i, init in enumerate(inits):
        # Check time
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 10: # Leave 10s for final check/return
            break
            
        # Optimize
        try:
            optimized_pts = optimize_from_start(init)
            r = compute_radius(optimized_pts)
            
            if r > best_r:
                best_r = r
                best_pts = optimized_pts
                
        except Exception:
            pass
            
    # Final safety clip
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

# EVOLVE-BLOCK-END
