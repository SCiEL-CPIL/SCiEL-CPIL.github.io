import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time

N = 50

def _smooth_min(z, sigma):
    """
    Smooth approximation of the minimum of vector z.
    z: 1D array of distances
    sigma: smoothing parameter
    """
    # Shift to avoid overflow in exp
    z_shifted = z - np.min(z)
    # If sigma is very small, this approximates min(z)
    # log(sum(exp(-z/sigma))) * sigma
    # We want to maximize min(z), so we minimize -smooth_min(z)
    # smooth_min = -sigma * log(sum(exp(-z/sigma)))
    
    # Numerical stability:
    # exp(-z/sigma) can overflow if z is very negative, but distances are positive.
    # However, if z is large, exp(-z/sigma) underflows to 0, which is fine.
    # The term -z/sigma is negative.
    
    # To be safe, we can use the log-sum-exp trick:
    # log(sum(exp(a_i))) = max(a) + log(sum(exp(a_i - max(a))))
    # Here a_i = -z_i / sigma
    a = -z / sigma
    max_a = np.max(a)
    if not np.isfinite(max_a):
        return -np.inf
    log_sum = max_a + np.log(np.sum(np.exp(a - max_a)))
    return -sigma * log_sum

def _objective(centres_flat, sigma):
    """
    Objective function to minimize.
    We want to maximize the minimum distance, so we minimize the negative smooth min.
    """
    centres = centres_flat.reshape(N, 2)
    
    # Pairwise distances
    d = pdist(centres)
    
    # Boundary distances
    # Distance to each of the 4 walls
    # min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y) for each point
    # This is 0.5 - max(|x|, |y|) ... wait.
    # The distance to the boundary of the square [-0.5, 0.5]x[-0.5, 0.5] for a point (x,y)
    # is min(0.5 - x, x + 0.5, 0.5 - y, y + 0.5)
    # which simplifies to 0.5 - max(|x|, |y|)
    # Actually, min(0.5-x, 0.5+x) = 0.5 - |x|.
    # So dist_to_boundary = min(0.5 - |x|, 0.5 - |y|) = 0.5 - max(|x|, |y|).
    
    bdist = 0.5 - np.maximum(np.abs(centres[:, 0]), np.abs(centres[:, 1]))
    
    # Combine all distances
    # The radius is limited by d_ij / 2 and b_i
    # So we form a vector of "effective radii"
    # r_ij = d_ij / 2
    # r_i = b_i
    r_vec = np.concatenate([d / 2.0, bdist])
    
    # We want to maximize min(r_vec)
    # smooth_min approximates min(r_vec)
    sm = _smooth_min(r_vec, sigma)
    
    return -sm

def _hex_init():
    """
    Initialize with a hexagonal pattern.
    """
    # Approximate hexagonal packing
    # For 50 points, we can try to fit a hexagonal grid
    # Side length of hexagon packing is related to radius.
    # Let's just create a grid and perturb it or use a known good seed.
    # A simple hexagonal arrangement:
    # Rows with alternating offsets.
    
    points = []
    # Try to fit 50 points
    # 7 rows of 7 or 8?
    # 7*7=49. Add 1 more?
    # Let's use a grid that is close to hexagonal.
    m = 7
    # Spacing
    # If we fit 7x7 in [-0.5, 0.5], spacing is 1/6.
    # Hex spacing: dx = 1/6, dy = sqrt(3)/2 * dx.
    
    dx = 1.0 / 6.0
    dy = np.sqrt(3) / 2.0 * dx
    
    # Center the grid
    # Total width = 6 * dx = 1.0
    # Total height = 6 * dy = 6 * sqrt(3)/12 = sqrt(3)/2 ~ 0.866
    # This fits in the square.
    
    for i in range(m):
        for j in range(m):
            x = (j - (m-1)/2.0) * dx
            y = (i - (m-1)/2.0) * dy
            if i % 2 == 1:
                x += dx / 2.0
            points.append([x, y])
            
    # We have 49 points. Need 1 more.
    # Add a point in the center or a gap?
    # The center of the 7x7 grid (i=3, j=3) is occupied.
    # Let's just use a random 50th point near the center or extend the grid.
    # Better: Use 8x7 = 56 points and prune? Or just add one random.
    # Actually, 50 is not a perfect hex number.
    # Let's just use the 49 and add one random near the middle.
    points.append([0.0, 0.0]) # Center is occupied by (3,3) in 0-indexed 7x7?
    # i=3, j=3 -> x=0, y=0. Yes, occupied.
    # Let's shift the last one or just use a random point.
    # To avoid overlap, let's just use a random point far from others?
    # Or better, use a slightly different grid.
    
    # Alternative: Use a 5x10 grid?
    # Let's stick to the 49+1 but move the 50th to a hole.
    # In a hex grid, holes are small.
    # Let's just use a random initialization for the 50th and let the optimizer fix it.
    
    return np.array(points)

def _random_init(rng):
    return (rng.random((N, 2)) - 0.5) * 0.95

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 15s safety margin
    
    rng = np.random.default_rng(42)
    
    best_centres = None
    best_radius = -1.0
    
    # List of initializations
    inits = []
    inits.append(_hex_init())
    for _ in range(10):
        inits.append(_random_init(rng))
    
    # Sigma annealing schedule
    sigmas = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        current_centres = init.copy()
        
        for sigma in sigmas:
            if time.time() > deadline - 5:
                break
                
            # Flatten for optimizer
            x0 = current_centres.flatten()
            
            # Bounds
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            try:
                # Use L-BFGS-B
                # maxiter needs to be sufficient for convergence
                result = minimize(
                    _objective,
                    x0,
                    args=(sigma,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7}
                )
                
                if result.success or result.fun < 1e10:
                    new_centres = result.x.reshape(N, 2)
                    # Clamp to ensure validity
                    new_centres = np.clip(new_centres, -0.5, 0.5)
                    
                    # Calculate actual radius
                    d = pdist(new_centres)
                    min_pair = np.min(d) / 2.0
                    bdist = 0.5 - np.maximum(np.abs(new_centres[:, 0]), np.abs(new_centres[:, 1]))
                    min_bound = np.min(bdist)
                    r = min(min_pair, min_bound)
                    
                    if r > best_radius:
                        best_radius = r
                        best_centres = new_centres.copy()
                    
                    current_centres = new_centres.copy()
            except Exception:
                pass
    
    # Final polish with a very small sigma to sharpen
    if best_centres is not None and time.time() < deadline - 2:
        x0 = best_centres.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        try:
            result = minimize(
                _objective,
                x0,
                args=(0.0005,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success:
                new_centres = result.x.reshape(N, 2)
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                d = pdist(new_centres)
                min_pair = np.min(d) / 2.0
                bdist = 0.5 - np.maximum(np.abs(new_centres[:, 0]), np.abs(new_centres[:, 1]))
                min_bound = np.min(bdist)
                r = min(min_pair, min_bound)
                
                if r > best_radius:
                    best_radius = r
                    best_centres = new_centres.copy()
        except Exception:
            pass
            
    if best_centres is None:
        # Fallback
        best_centres = _hex_init()
        
    return best_centres
