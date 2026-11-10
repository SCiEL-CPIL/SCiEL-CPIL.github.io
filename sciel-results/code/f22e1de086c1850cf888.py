"""
Hybrid packing of 50 equal spheres in a unit cube.
Strategy:
1. Initialize using a 4x4x4 grid (structured, well-spaced).
2. Global optimization using a smooth repulsive potential (L-BFGS-B) to escape poor local minima.
3. Local refinement using Simulated Annealing on the exact minimum-distance metric to maximize the radius.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    if min_wall < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r = min(min_wall, min_pair / 2.0)
    return r if r > 0 else 0.0

def _repulsion_potential(x):
    """
    Smooth potential energy: sum of inverse squared distances.
    Minimizing this pushes points apart.
    """
    centers = x.reshape(N, 3)
    
    # Pairwise repulsion
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1e10) # Avoid div by zero
    
    # Use 1/d^2 for standard repulsion
    energy = np.sum(1.0 / dists_sq)
    
    return energy

def _initial_grid(n):
    """Generate a 4x4x4 grid subset."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    if len(grid) >= n:
        return grid[:n].copy()
    else:
        # Fallback
        return np.random.uniform(-0.45, 0.45, size=(n, 3))

def _global_optimize(centers, max_time):
    """Run L-BFGS-B on the repulsion potential."""
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # L-BFGS-B is efficient for this
    try:
        res = minimize(
            _repulsion_potential, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
        )
        # Check if we ran out of time or if the result is valid
        if time.time() - start_time < max_time and res.x is not None:
             return res.x.reshape(N, 3)
    except Exception:
        pass
        
    return centers

def _local_refine(centers, max_time):
    """
    Simulated Annealing / Local Search on the exact min-radius metric.
    This corrects the 'average' bias of the repulsion potential to focus on the 'minimum'.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    current_centers = centers.copy()
    current_radius = best_radius
    
    rng = np.random.RandomState(42)
    T_start = 0.01
    T_end = 0.0001
    n_steps = 20000 # Will be cut by time
    
    for step in range(n_steps):
        if time.time() - start_time > max_time:
            break
        
        frac = step / n_steps
        T = T_start * (T_end / T_start) ** frac
        
        i = rng.randint(N)
        
        # Adaptive step size
        step_size = 0.01 * (1.0 - 0.5 * frac)
        delta = rng.uniform(-step_size, step_size, 3)
        
        new_centers = current_centers.copy()
        new_centers[i] += delta
        new_centers[i] = np.clip(new_centers[i], -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius >= current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
        else:
            delta_e = new_radius - current_radius
            prob = np.exp(delta_e / (T * 0.01))
            if rng.random() < prob:
                current_centers = new_centers
                current_radius = new_radius

    return best_centers, best_radius

def construct():
    start_time = time.time()
    remaining = DEADLINE - (time.time() - start_time)
    
    # 1. Initial Grid
    centers = _initial_grid(N)
    
    # 2. Global Optimization (Smooth Potential)
    # Allocate ~60% of time to global, 40% to local
    time_global = remaining * 0.6
    time_local = remaining * 0.4 - 5.0 # 5s safety margin
    
    if time_global > 5.0:
        centers = _global_optimize(centers, time_global)
        
    # 3. Local Refinement (Exact Metric)
    remaining_local = DEADLINE - (time.time() - start_time) - 2.0 # 2s safety
    if remaining_local > 1.0:
        centers, best_radius = _local_refine(centers, remaining_local)
    else:
        best_radius = _compute_radius(centers)
        
    # Final check
    final_radius = _compute_radius(centers)
    if final_radius > best_radius:
        best_radius = final_radius
        
    # Ensure validity
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        # Fallback
        return _initial_grid(N)
        
    return centers
