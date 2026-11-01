"""Optimized packing of 50 equal spheres in a unit cube using multi-start local search."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 50
CUBE_BOUND = 0.5
DEADLINE_MARGIN = 5.0  # seconds before the 150s limit


def _score(centers):
    """Compute the maximum common radius for given centers."""
    # Check bounds
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    # Check if any center is outside the cube
    if np.any(np.abs(centers) > CUBE_BOUND + 1e-12):
        return 0.0
    # Wall clearance
    wall_clearances = CUBE_BOUND - np.abs(centers)
    min_wall = np.min(wall_clearances)
    if min_wall < 0:
        return 0.0
    # Pairwise distances
    tree = cKDTree(centers)
    # Get all pairs with distance less than some threshold to avoid O(N^2) memory
    # For N=50, we can just compute all pairwise distances
    from scipy.spatial.distance import pdist
    dists = pdist(centers, metric='euclidean')
    min_pair_dist = np.min(dists)
    r = min(min_pair_dist / 2.0, min_wall)
    return max(0.0, r)


def _neg_score(centers):
    """Negative of the score for minimization."""
    return -_score(centers)


def _local_search(centers, max_iters=200, tol=1e-12):
    """Local search using gradient-free optimization (Nelder-Mead) on the score."""
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    # Use Nelder-Mead on the negative score
    # We need to handle the non-smoothness carefully
    # Instead, let's do a coordinate-wise local search
    
    for it in range(max_iters):
        improved = False
        # Try small perturbations in random directions
        for _ in range(3):
            direction = np.random.randn(N, 3)
            direction /= np.linalg.norm(direction)
            for step in [0.001, 0.0005, 0.0001, 0.00005]:
                new_centers = best_centers + step * direction
                # Clamp to cube
                new_centers = np.clip(new_centers, -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)
                new_score = _score(new_centers)
                if new_score > best_score + 1e-15:
                    best_centers = new_centers
                    best_score = new_score
                    improved = True
                    break
        if not improved:
            break
    
    return best_centers, best_score


def _lloyd_iteration(centers, iters=50):
    """Voronoi Lloyd's algorithm to equalize clearance."""
    for _ in range(iters):
        # Compute Voronoi cells using scipy's Voronoi
        from scipy.spatial import Voronoi
        try:
            vor = Voronoi(centers)
        except Exception:
            break
        
        new_centers = centers.copy()
        for i in range(N):
            # Get the region for point i
            region = vor.regions[vor.point_region[i]]
            if -1 in region or len(region) == 0:
                continue
            
            # Get the vertices of the Voronoi cell
            verts = vor.vertices[region]
            if len(verts) == 0:
                continue
            
            # Clip to cube
            # For simplicity, just use the centroid of the Voronoi cell vertices
            centroid = np.mean(verts, axis=0)
            
            # Clip to cube
            centroid = np.clip(centroid, -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)
            new_centers[i] = centroid
        
        # Check convergence
        if np.max(np.abs(new_centers - centers)) < 1e-10:
            centers = new_centers
            break
        centers = new_centers
    
    return centers


def _soft_min_score(centers, temperature=0.001):
    """Soft-min approximation for smoother optimization."""
    wall_clearances = CUBE_BOUND - np.abs(centers)
    min_wall = np.min(wall_clearances)
    
    from scipy.spatial.distance import pdist
    dists = pdist(centers, metric='euclidean')
    # Soft min of distances
    soft_min_dist = -temperature * np.log(np.mean(np.exp(-dists / temperature)))
    
    r = min(soft_min_dist / 2.0, min_wall)
    return r


def _gradient_ascent(centers, max_iters=100, lr=0.001):
    """Simple gradient ascent using finite differences on the soft-min score."""
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    for it in range(max_iters):
        # Compute gradient via finite differences
        eps = 1e-5
        grad = np.zeros_like(centers)
        for i in range(N):
            for j in range(3):
                center_plus = best_centers.copy()
                center_plus[i, j] += eps
                center_plus = np.clip(center_plus, -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)
                score_plus = _soft_min_score(center_plus, temperature=0.001)
                
                center_minus = best_centers.copy()
                center_minus[i, j] -= eps
                center_minus = np.clip(center_minus, -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)
                score_minus = _soft_min_score(center_minus, temperature=0.001)
                
                grad[i, j] = (score_plus - score_minus) / (2 * eps)
        
        # Update
        new_centers = best_centers + lr * grad
        new_centers = np.clip(new_centers, -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)
        
        new_score = _score(new_centers)
        if new_score > best_score + 1e-15:
            best_centers = new_centers
            best_score = new_score
        else:
            # Reduce learning rate
            lr *= 0.5
            if lr < 1e-8:
                break
    
    return best_centers, best_score


def _cubic_grid_init():
    """Create a cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]


def _random_init():
    """Random initialization within the cube."""
    return np.random.uniform(-CUBE_BOUND + 0.01, CUBE_BOUND - 0.01, (N, 3))


def _jittered_grid_init():
    """Grid initialization with small random jitter."""
    base = _cubic_grid_init()
    jitter = np.random.randn(N, 3) * 0.005
    centers = base + jitter
    return np.clip(centers, -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)


def construct():
    """Construct the optimal packing of 50 spheres in a unit cube."""
    start_time = time.time()
    
    # Start with the grid initialization
    best_centers = _cubic_grid_init()
    best_score = _score(best_centers)
    
    # Apply Lloyd's algorithm to equalize clearance
    lloyd_centers = _lloyd_iteration(best_centers.copy(), iters=100)
    lloyd_score = _score(lloyd_centers)
    if lloyd_score > best_score:
        best_centers = lloyd_centers
        best_score = lloyd_score
    
    # Local search refinement
    refined_centers, refined_score = _local_search(best_centers.copy(), max_iters=500)
    if refined_score > best_score:
        best_centers = refined_centers
        best_score = refined_score
    
    # Gradient ascent with soft-min
    grad_centers, grad_score = _gradient_ascent(best_centers.copy(), max_iters=200)
    if grad_score > best_score:
        best_centers = grad_centers
        best_score = grad_score
    
    # Multi-start with random initializations
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    n_restarts = 0
    while time.time() < deadline and n_restarts < 20:
        n_restarts += 1
        if n_restarts == 1:
            # Already did grid, try jittered
            init_centers = _jittered_grid_init()
        else:
            init_centers = _random_init()
        
        # Quick Lloyd's
        lloyd_c = _lloyd_iteration(init_centers.copy(), iters=30)
        lloyd_s = _score(lloyd_c)
        
        # Local search
        local_c, local_s = _local_search(lloyd_c.copy(), max_iters=200)
        
        if local_s > best_score:
            best_centers = local_c
            best_score = local_s
        
        # Check time
        if time.time() > deadline - 30:
            break
    
    # Final polish with tighter local search
    if time.time() < deadline - 10:
        final_c, final_s = _local_search(best_centers.copy(), max_iters=1000)
        if final_s > best_score:
            best_centers = final_c
            best_score = final_s
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)
    return best_centers
