import numpy as np
from scipy.optimize import minimize
import time

N = 11
MIN_DIST = np.sqrt(3)

def hexagon_side_length(points):
    """
    Calculate the minimal side length of a regular hexagon (centered at 0,0, 
    flat-top orientation relative to x-axis? No, vertices at 0, 60, etc)
    that contains all points.
    The hexagon is defined by |x cos(60k) + y sin(60k)| <= L * sqrt(3)/2 for k=0,1,2.
    So L = max |x cos(60k) + y sin(60k)| * 2/sqrt(3).
    """
    if len(points) == 0:
        return 0.0
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Angles for the 3 unique normal vectors
    # k=0: 0 deg -> (1, 0)
    # k=1: 60 deg -> (0.5, sqrt(3)/2)
    # k=2: 120 deg -> (-0.5, sqrt(3)/2)
    
    vals = np.zeros((len(points), 3))
    vals[:, 0] = np.abs(x)
    vals[:, 1] = np.abs(0.5 * x + (np.sqrt(3)/2) * y)
    vals[:, 2] = np.abs(-0.5 * x + (np.sqrt(3)/2) * y)
    
    max_val = np.max(vals)
    L = max_val * 2 / np.sqrt(3)
    return L

def cost_func(params):
    """
    Minimize the bounding hexagon side length + penalty for overlaps.
    """
    points = params.reshape(N, 2)
    L = hexagon_side_length(points)
    
    # Penalty for overlaps
    # Calculate all pairwise distances
    # dist[i,j] = ||p_i - p_j||
    dists = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
    # Exclude diagonal
    mask = ~np.eye(N, dtype=bool)
    min_dist = np.min(dists[mask])
    
    if min_dist < MIN_DIST:
        # Strong penalty
        penalty = (MIN_DIST - min_dist) ** 2
    else:
        penalty = 0.0
        
    return L + 100.0 * penalty

def initial_guess():
    """
    Generate an initial guess for the 11 points.
    Start with a 7-point cluster and add 4 more.
    """
    # 7 point cluster: center + 6 neighbors at dist sqrt(3)
    pts = []
    pts.append([0.0, 0.0])
    for k in range(6):
        angle = k * np.pi / 3
        pts.append([MIN_DIST * np.cos(angle), MIN_DIST * np.sin(angle)])
    
    # Add 4 more points, slightly outside
    for k in range(4):
        angle = (k + 0.5) * np.pi / 3
        # Place them a bit further out
        r = MIN_DIST * 1.5
        pts.append([r * np.cos(angle), r * np.sin(angle)])
        
    return np.array(pts).flatten()

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    best_L = np.inf
    best_params = None
    
    # Run multiple restarts
    for i in range(50):
        if time.time() > deadline:
            break
            
        x0 = initial_guess()
        # Add some noise
        x0 += np.random.normal(0, 0.5, size=x0.shape)
        
        try:
            res = minimize(cost_func, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000})
            
            L = hexagon_side_length(res.x.reshape(N, 2))
            # Verify no overlaps
            points = res.x.reshape(N, 2)
            dists = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
            min_dist = np.min(dists[~np.eye(N, dtype=bool)])
            
            if min_dist >= MIN_DIST - 1e-5:
                if L < best_L:
                    best_L = L
                    best_params = res.x
        except Exception:
            pass
            
    if best_params is None:
        # Fallback to simple line or grid if optimization fails
        # Just return a valid, albeit suboptimal, configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    points = best_params.reshape(N, 2)
    S = best_L + 1.0
    
    # Construct inner list
    # Format: (x, y, angle_degrees)
    # All hexagons are aligned with the outer, so angle is 0.
    inner = [[float(p[0]), float(p[1]), 0.0] for p in points]
    
    return inner, (0.0, 0.0), float(S), 0.0

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to ensure it is inside the evolve block or accessible.
# The prompt says "Keep the function names and signatures of the seed program".
# The seed program has construct() inside the EVOLVE-BLOCK.
# I will place the helper functions and construct inside the block.
# EVOLVE-BLOCK-END
