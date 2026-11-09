# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def _score(points):
    """Compute R = (d_min / d_max)^2. We maximize this, so we minimize -R."""
    n = points.shape[0]
    if n < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-15:
        return 0.0
    return (d_min / d_max) ** 2

def _neg_score(points):
    """Objective function for minimizer: negative of R."""
    return -_score(points)

def _generate_hexagonal_subset():
    """
    Generate 16 points from a hexagonal lattice.
    This is a known good starting point for packing problems.
    """
    points = []
    # Generate a grid of hex lattice points
    # Hex lattice basis: (1, 0), (0.5, sqrt(3)/2)
    s = 1.0
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * s + 0.5 * j * s
            y = j * s * np.sqrt(3) / 2
            points.append([x, y])
    
    points = np.array(points)
    # Sort by distance to origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    # Take the 16 closest to the center
    pts = points[idx[:16]]
    return pts

def _generate_circular_rings():
    """
    Generate points on two concentric circles.
    6 on inner, 10 on outer? Or 8 and 8?
    Let's try 6 inner, 10 outer.
    """
    pts = []
    # Inner ring: 6 points
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    # Outer ring: 10 points
    for i in range(10):
        angle = 2 * np.pi * i / 10 + np.pi / 10 # Offset to interleave
        pts.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    return np.array(pts)

def _generate_grid_jittered():
    """
    Generate a 4x4 grid with some jitter.
    """
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    # Add small jitter to break symmetry
    pts += np.random.randn(16, 2) * 0.1
    return pts

def _generate_random():
    """
    Generate random points in a unit square.
    """
    return np.random.randn(16, 2)

def _optimize_nelder_mead(points, time_budget, rng):
    """
    Use Nelder-Mead to optimize the point configuration.
    """
    start_time = time.time()
    
    # Flatten points to a 1D array for scipy.optimize
    x0 = points.flatten()
    
    def objective(x):
        # Reshape to (16, 2)
        pts = x.reshape(16, 2)
        # Check time
        if time.time() - start_time > time_budget:
            raise TimeoutError
        return _neg_score(pts)
    
    # Nelder-Mead options
    # maxiter is a rough guide, but we also use time
    # n=32 variables, so 33 simplex points.
    # Initial step size matters. 
    # We normalize the input points to have max norm ~1 before flattening.
    
    try:
        # Use 'nelder-mead'
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        
        # Check if it timed out or failed
        if res.x is None:
            return points, _score(points)
            
        best_pts = res.x.reshape(16, 2)
        best_score = _score(best_pts)
        return best_pts, best_score
        
    except TimeoutError:
        return points, _score(points)
    except Exception:
        return points, _score(points)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2
    deadline = time.time() + 590 # 10s safety margin

    best_points = None
    best_score = -1.0
    
    rng = np.random.default_rng(42)

    # List of initializers
    initializers = [
        _generate_hexagonal_subset,
        _generate_circular_rings,
        _generate_grid_jittered,
        _generate_random
    ]
    
    # We will run multiple iterations.
    # First pass: Run each initializer once.
    # Second pass: If time permits, run random restarts.
    
    # To maximize diversity, we shuffle the initializers or just run them in sequence.
    # Let's run them in sequence.
    
    num_rounds = 10
    for r in range(num_rounds):
        if time.time() > deadline - 30:
            break
            
        # Select an initializer
        # In the first few rounds, use the structured ones.
        if r < len(initializers):
            init_func = initializers[r]
        else:
            # Use random or a slightly perturbed version of the best found so far?
            # Let's just use random for now to explore new basins.
            init_func = _generate_random
            
        init_pts = init_func()
        
        # Normalize initial points
        init_pts = init_pts - init_pts.mean(axis=0)
        norm = np.max(np.linalg.norm(init_pts, axis=1))
        if norm > 1e-10:
            init_pts = init_pts / norm
            
        current_score = _score(init_pts)
        
        # If this initialization is already very good, we might skip optimization or use a tighter budget?
        # For now, always optimize.
        
        # Allocate time budget for this optimization
        # We have ~590s total.
        # Let's give each optimization ~30-50s.
        # If we do 10 rounds, that's 300-500s.
        time_budget = min(40, deadline - time.time() - 10)
        
        if time_budget < 5:
            break
            
        optimized_pts, opt_score = _optimize_nelder_mead(init_pts, time_budget, rng)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = optimized_pts.copy()
            
        # If we found a very good solution (R > 0.25 is quite good for 16 points in 2D? 
        # Actually, max R is 1.0 for equilateral. 
        # For 16 points, the optimal packing ratio is likely around 0.2-0.4?
        # Let's check known values. 
        # For n=16, the "best" known packing has min/max ratio around 0.4-0.5? 
        # R = ratio^2. So R ~ 0.16 - 0.25.
        # If we hit R > 0.2, it's a strong candidate.
        
    # If we have time left, do a final polish on the best solution found
    if best_points is not None and time.time() < deadline - 10:
        remaining = deadline - time.time() - 10
        if remaining > 10:
            # Polish the best point
            # Use a smaller step size or more iterations?
            # Nelder-Mead with 'adaptive' should have converged.
            # Let's try running it again with a tighter tolerance if time permits.
            # Or just return.
            pass

    if best_points is None:
        # Fallback
        return np.random.randn(n, d)
        
    return best_points

# EVOLVE-BLOCK-END
