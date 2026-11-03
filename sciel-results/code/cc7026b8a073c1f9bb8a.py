import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 115.0  # Safety margin before 120s limit

def _score(centers):
    """Compute the radius for a given configuration (vectorized)."""
    # Boundary constraint
    boundary = (0.5 - np.abs(centers)).min()
    if boundary < 0:
        return 0.0
    
    # Pairwise distances - vectorized
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    
    r = min(boundary, min_dist / 2.0)
    return r

def _score_fast(centers, prev_min_idx=None):
    """Faster scoring that reuses previous min distance index if provided."""
    # Boundary constraint
    boundary = (0.5 - np.abs(centers)).min()
    if boundary < 0:
        return 0.0
    
    # Pairwise distances - vectorized
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    
    r = min(boundary, min_dist / 2.0)
    return r

def _grid_init(m=7):
    """Create a grid initialization for N points."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Take first N points
    return grid[:N]

def _hex_init():
    """Create a hexagonal packing initialization."""
    centers = []
    # Hexagonal packing in a square
    # Try to fit 50 circles in hexagonal arrangement
    row_spacing = 1.0 / 8
    col_spacing = row_spacing * 2 / np.sqrt(3)
    
    for row in range(8):
        y = -0.5 + row_spacing * (row + 0.5)
        offset = 0 if row % 2 == 0 else col_spacing / 2
        for col in range(8):
            x = -0.5 + col_spacing * (col + 0.5) + offset
            if abs(x) <= 0.5 and abs(y) <= 0.5:
                centers.append([x, y])
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
    
    # If we don't have enough, add random points
    while len(centers) < N:
        centers.append(np.random.uniform(-0.45, 0.45, 2))
    
    return np.array(centers[:N])

def _random_init():
    """Create a random initialization."""
    return np.random.uniform(-0.45, 0.45, (N, 2))

def _force_relax(centers, iterations=500, dt=0.01):
    """Simple force-based relaxation to resolve overlaps."""
    centers = centers.copy()
    r = _score(centers)
    
    for it in range(iterations):
        # Compute forces
        forces = np.zeros_like(centers)
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Repulsive force for overlapping circles
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 2 * r and d > 1e-10:
                    # Repulsive force
                    force = (2 * r - d) / d
                    forces[i] += force * diffs[i, j]
        
        # Boundary forces
        for i in range(N):
            for k in range(2):
                if centers[i, k] > 0.5 - r:
                    forces[i, k] -= 10 * (centers[i, k] - (0.5 - r))
                elif centers[i, k] < -0.5 + r:
                    forces[i, k] -= 10 * (centers[i, k] - (-0.5 + r))
        
        # Update positions
        centers += dt * forces
        # Clamp to square
        centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
    
    return centers

def _objective(x, r_target=None):
    """Objective function for optimization: maximize minimum radius."""
    centers = x.reshape(-1, 2)
    r = _score(centers)
    return -r

def _gradient(x):
    """Compute gradient of negative radius (numerical)."""
    centers = x.reshape(-1, 2)
    r = _score(centers)
    
    # Numerical gradient
    eps = 1e-7
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        r_plus = _score(x_plus.reshape(-1, 2))
        grad[i] = (-r_plus + r) / eps
    
    return grad

def _optimize_lbfgsb(x0, maxiter=1000):
    """Optimize using L-BFGS-B."""
    try:
        res = minimize(_objective, x0, method='L-BFGS-B', 
                      jac=_gradient,
                      options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
        return res.x
    except Exception:
        return x0

def _optimize_sa(x0, iterations=2000, t0=0.1, t1=0.001):
    """Simulated annealing optimization."""
    centers = x0.reshape(-1, 2).copy()
    current_r = _score(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    for it in range(iterations):
        # Temperature schedule
        t = t0 * (t1 / t0) ** (it / iterations)
        
        # Random perturbation
        new_centers = centers + np.random.normal(0, 0.01, centers.shape)
        new_centers = np.clip(new_centers, -0.5 + 1e-6, 0.5 - 1e-6)
        
        new_r = _score(new_centers)
        
        # Accept if better or with probability
        if new_r > current_r or np.random.random() < np.exp((new_r - current_r) / t):
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
    
    return best_centers.flatten()

def _local_refine(x0, maxiter=500):
    """Local refinement using Nelder-Mead."""
    try:
        res = minimize(_objective, x0, method='Nelder-Mead',
                      options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        return res.x
    except Exception:
        return x0

def _voronoi_refine(centers):
    """Voronoi-based active set refinement: move points that are close to boundaries or each other."""
    centers = centers.copy()
    for _ in range(20):
        r = _score(centers)
        if r <= 0:
            break
            
        # Find the most constrained pairs
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find minimum distance pair
        min_idx = np.unravel_index(dists.argmin(), dists.shape)
        i, j = min_idx
        
        # Move both points apart slightly
        direction = (centers[i] - centers[j])
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction /= norm
            centers[i] += 0.001 * direction
            centers[j] -= 0.001 * direction
        
        # Check boundary constraints
        for k in range(N):
            for dim in range(2):
                if centers[k, dim] > 0.5 - r:
                    centers[k, dim] = 0.5 - r - 0.001
                elif centers[k, dim] < -0.5 + r:
                    centers[k, dim] = -0.5 + r + 0.001
        
        centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
    
    return centers

def construct():
    start_time = time.time()
    
    # Try multiple initializations and optimization strategies
    best_x = None
    best_score = -1.0
    
    # Different initializations
    inits = [
        _grid_init(7),
        _hex_init(),
        _random_init(),
        _random_init(),
        _random_init()
    ]
    
    for init in inits:
        if time.time() > DEADLINE - 30:
            break
            
        centers = init.copy()
        
        # Force relaxation first
        centers = _force_relax(centers, iterations=200, dt=0.005)
        x0 = centers.flatten()
        
        # L-BFGS-B optimization
        x1 = _optimize_lbfgsb(x0, maxiter=500)
        score1 = _score(x1.reshape(-1, 2))
        
        if score1 > best_score:
            best_score = score1
            best_x = x1.copy()
        
        # Simulated annealing
        if time.time() > DEADLINE - 20:
            break
        x2 = _optimize_sa(x1, iterations=1000)
        score2 = _score(x2.reshape(-1, 2))
        
        if score2 > best_score:
            best_score = score2
            best_x = x2.copy()
        
        # Local refinement
        x3 = _local_refine(best_x, maxiter=300)
        score3 = _score(x3.reshape(-1, 2))
        
        if score3 > best_score:
            best_score = score3
            best_x = x3.copy()
        
        # Voronoi refinement
        if time.time() > DEADLINE - 10:
            break
        centers_v = _voronoi_refine(best_x.reshape(-1, 2))
        score_v = _score(centers_v)
        
        if score_v > best_score:
            best_score = score_v
            best_x = centers_v.flatten()
    
    # Final local refinement
    if time.time() < DEADLINE - 5:
        x_final = _local_refine(best_x, maxiter=500)
        score_final = _score(x_final.reshape(-1, 2))
        if score_final > best_score:
            best_x = x_final
            best_score = score_final
    
    # Ensure valid output
    centers = best_x.reshape(-1, 2)
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
