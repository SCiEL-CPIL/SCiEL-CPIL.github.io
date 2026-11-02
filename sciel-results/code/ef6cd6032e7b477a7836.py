"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    safety_margin = 5.0  # return 5s before deadline
    
    # Initial guess: hexagonal lattice
    # For 200 circles, a 14x15 grid with hexagonal offset
    # Let's use a simple grid first, then optimize
    m = int(np.ceil(np.sqrt(N)))
    # Use a hexagonal packing initialization
    points = _hex_init(N)
    
    # Optimize using gradient ascent on the log-determinant of Gram matrix
    # or use a simpler approach: maximize minimum distance via iterative repulsion
    
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Iterative optimization with multiple restarts if time permits
    num_iters = 5000
    lr = 0.001
    
    for iteration in range(num_iters):
        if time.time() > deadline - safety_margin:
            break
        
        # Compute pairwise distances
        d = pdist(points)
        min_d = np.min(d)
        
        # Gradient: for each pair, push apart if close
        # Use a soft repulsion force
        # Force on point i from point j: (p_i - p_j) / |p_i - p_j|^2 * strength
        # But we want to maximize min distance, so use a potential
        
        # Compute all pairwise distances (N x N)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        
        # Mask diagonal
        np.fill_diagonal(dist, np.inf)
        
        # Gradient of min distance is tricky; use sum of inverse distances
        # or use a soft min via log-sum-exp
        
        # Simpler: use repulsive force proportional to 1/d^2 for close pairs
        # Only consider pairs closer than some threshold
        threshold = 2 * best_score + 0.01
        mask = dist < threshold
        np.fill_diagonal(mask, False)
        
        # Force: sum over j of (p_i - p_j) / dist_ij^2 for close pairs
        inv_dist_sq = np.zeros_like(dist)
        inv_dist_sq[mask] = 1.0 / dist[mask]**2
        np.fill_diagonal(inv_dist_sq, 0)
        
        # Force on each point
        force = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Update
        points = points + lr * force
        
        # Project back to square [-0.5, 0.5]^2
        points = np.clip(points, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Check score
        current_score = _score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        # Adaptive learning rate
        if iteration % 100 == 0:
            lr *= 0.95
    
    # Final polish with scipy minimize
    if time.time() < deadline - safety_margin - 10:
        try:
            def neg_score(x):
                pts = x.reshape(-1, 2)
                return -_score(pts)
            
            res = minimize(neg_score, best_points.flatten(), method='L-BFGS-B',
                          bounds=[(-0.5, 0.5)] * 2 * N,
                          options={'maxiter': 100, 'ftol': 1e-10})
            if res.success or res.fun < -best_score:
                candidate = res.x.reshape(-1, 2)
                candidate_score = _score(candidate)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate
        except Exception:
            pass
    
    return best_points


def _hex_init(n):
    """Generate hexagonal lattice initialization for n points."""
    # Hexagonal packing: rows offset by half spacing
    # For a square, use a grid with hexagonal offset
    rows = int(np.ceil(np.sqrt(n * np.sqrt(3) / 2)))
    cols = int(np.ceil(n / rows))
    
    # Spacing for hexagonal packing
    # In hexagonal packing, horizontal spacing = s, vertical = s * sqrt(3)/2
    # For n points in cols x rows grid
    # We want to fit in [-0.5, 0.5]^2
    # cols * s <= 1, rows * s * sqrt(3)/2 <= 1
    s = min(1.0 / cols, 2.0 / (rows * np.sqrt(3)))
    
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= n:
                break
            x = (j - cols / 2 + 0.5) * s + (i % 2) * s / 2
            y = (i - rows / 2 + 0.5) * s * np.sqrt(3) / 2
            points.append([x, y])
        if len(points) >= n:
            break
    
    points = np.array(points[:n])
    # Center the points
    points -= np.mean(points, axis=0)
    # Scale to fit in square
    max_extent = np.max(np.abs(points))
    if max_extent > 0.5:
        points *= (0.5 - 1e-6) / max_extent
    return points


def _score(points):
    """Compute the maximum common radius for the given points."""
    if len(points) < 2:
        return 0.0
    
    # Minimum pairwise distance / 2
    d = pdist(points)
    min_pair_dist = np.min(d)
    
    # Minimum distance to boundary
    dist_to_boundary = np.min(np.abs(0.5 - np.abs(points)))
    
    return min(min_pair_dist / 2, dist_to_boundary)
# EVOLVE-BLOCK-END
