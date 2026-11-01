import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(positions):
    """
    Compute the radius of the largest equal spheres that can be placed
    at the given positions inside the unit cube [-0.5, 0.5]^3.
    """
    # Pairwise distances
    # positions shape: (N, 3)
    # We need min_{i<j} ||p_i - p_j||
    
    # Efficient way to compute pairwise distances
    # diff[i, j, k] = positions[i, k] - positions[j, k]
    # dist[i, j] = sqrt(sum(diff^2))
    
    # To save memory and time, we can compute in chunks or just use broadcasting
    # N=50 is small, so full matrix is fine.
    
    # Distance matrix
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # Cube is [-0.5, 0.5]^3
    # Distance to wall for point p is min(0.5 - p_i, p_i + 0.5) for each dim
    # Actually, distance to the boundary of the cube.
    # For a point x in [-0.5, 0.5], dist to wall is min(x - (-0.5), 0.5 - x) = min(x+0.5, 0.5-x)
    # Since x is in [-0.5, 0.5], x+0.5 >= 0 and 0.5-x >= 0.
    # The minimum distance to any wall is min over all coordinates and both sides.
    
    # dists_to_walls[i] = min(0.5 - positions[i,0], positions[i,0] + 0.5, ...)
    # Actually, just min(0.5 - |positions[i,0]|, 0.5 - |positions[i,1]|, 0.5 - |positions[i,2]|)
    # Wait, the cube is centered at origin.
    # Walls are at x=0.5, x=-0.5, y=0.5, y=-0.5, z=0.5, z=-0.5.
    # Distance from (x,y,z) to nearest wall is min(0.5-|x|, 0.5-|y|, 0.5-|z|).
    
    abs_pos = np.abs(positions)
    dists_to_walls = np.min(0.5 - abs_pos, axis=1)
    
    min_wall_dist = np.min(dists_to_walls)
    
    # The radius is half the minimum of the closest pair distance and the closest wall distance
    r = min(min_pair_dist, min_wall_dist) / 2.0
    
    # Ensure validity: if any point is outside, radius is negative or invalid?
    # The problem states: "A centre on or outside the cube... make the candidate invalid."
    # So we should ensure points stay inside. 
    # If a point goes outside, 0.5 - |x| becomes negative.
    # The formula r = min(...)/2 will be negative.
    # We want to maximize r.
    # So minimizing -r is correct.
    
    return r

def objective(x):
    """
    Objective function for minimization.
    We want to maximize the radius, so we minimize -radius.
    """
    positions = x.reshape(-1, 3)
    r = compute_radius(positions)
    return -r

def bcc_init():
    """
    Generate initial positions based on a Body-Centered Cubic lattice.
    """
    # A 3x3x3 grid of BCC cells gives 27 * 2 = 54 points.
    # We will generate these and select 50.
    
    # BCC unit cell points (relative to cell origin):
    # (0,0,0) and (0.5, 0.5, 0.5)
    
    points = []
    # Grid size 3x3x3
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Cell origin
                ox = -0.5 + i * (1.0/3.0)
                oy = -0.5 + j * (1.0/3.0)
                oz = -0.5 + k * (1.0/3.0)
                
                # Points in this cell
                # Corner
                points.append([ox, oy, oz])
                # Body center
                points.append([ox + 1.0/6.0, oy + 1.0/6.0, oz + 1.0/6.0])
                
    points = np.array(points)
    
    # We have 54 points. We need 50.
    # Let's just take the first 50. Or random 50?
    # Taking first 50 might bias towards corners.
    # Let's shuffle and take 50.
    np.random.shuffle(points)
    return points[:50]

def random_init():
    """
    Generate random initial positions inside the cube.
    """
    return np.random.uniform(-0.5, 0.5, (N, 3))

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin from 150s
    
    best_radius = -1.0
    best_positions = None
    
    # Try a few initializations
    # 1. BCC
    # 2. Random
    # 3. Grid (fallback)
    
    initializations = []
    
    # BCC init
    initializations.append(bcc_init())
    
    # Random inits
    for _ in range(5):
        initializations.append(random_init())
        
    # Simple grid init (just in case)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initializations.append(grid[:N])

    for init_pos in initializations:
        if time.time() - start_time > timeout:
            break
            
        # Flatten for optimizer
        x0 = init_pos.flatten()
        
        try:
            # Use Nelder-Mead
            # maxiter and maxfev need to be tuned based on time budget
            # N=50, so 150 variables.
            # Nelder-Mead can be slow.
            # Let's set a reasonable maxfev.
            result = minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 1000, 'maxfev': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if result.success or result.fun < -best_radius:
                r = -result.fun
                if r > best_radius:
                    best_radius = r
                    best_positions = result.x.reshape(-1, 3)
                    
        except Exception as e:
            # If optimization fails, ignore this restart
            pass

    # If no valid position found (should not happen), return random
    if best_positions is None:
        best_positions = random_init()
        
    return best_positions
