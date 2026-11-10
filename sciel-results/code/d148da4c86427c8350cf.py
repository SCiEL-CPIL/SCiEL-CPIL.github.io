"""Optimized packing of 100 equal spheres in a unit cube."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 100
# Deadline guard: evaluator kills after 150s; keep a safety margin.
_DEADLINE = 140.0  # seconds
_START_TIME = None


def _compute_radius(centres):
    """Compute the largest common radius for given centres."""
    # Distance to walls
    wall_dists = 0.5 - np.abs(centres)
    min_wall = wall_dists.min()
    
    # Pairwise distances
    # For N=100, we can compute all pairwise distances efficiently
    # Use broadcasting: diff[i,j,k] = centres[i,k] - centres[j,k]
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min()
    
    return min(min_wall, min_pair / 2.0)


def _objective(params, N=N):
    """Objective function: minimize negative radius (equivalent to maximizing radius).
    params: flat array of length 3*N representing centres.
    """
    centres = params.reshape(N, 3)
    r = _compute_radius(centres)
    return -r


def _objective_grad(params, N=N):
    """Gradient of the objective function."""
    centres = params.reshape(N, 3)
    
    # Compute wall distances
    wall_dists = 0.5 - np.abs(centres)
    min_wall = wall_dists.min()
    
    # Compute pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min()
    
    r = min(min_wall, min_pair / 2.0)
    
    # Compute gradient
    grad = np.zeros_like(params)
    
    if min_wall <= min_pair / 2.0:
        # Wall constraint is active
        # Gradient of min_wall w.r.t. centres
        # min_wall = min_i min_k (0.5 - |c_ik|)
        # For each centre i and coordinate k where it's the minimum:
        # d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
        for i in range(N):
            for k in range(3):
                if wall_dists[i, k] <= min_wall + 1e-12:
                    if centres[i, k] > 0:
                        grad[i * 3 + k] = 1.0  # d(-r)/dc = -d(r)/dc, and d(min_wall)/dc_ik = -sign(c_ik), so d(-min_wall)/dc_ik = sign(c_ik)
                    elif centres[i, k] < 0:
                        grad[i * 3 + k] = -1.0
                    else:
                        grad[i * 3 + k] = 0.0
    else:
        # Pair constraint is active
        # Find the pair with minimum distance
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        
        # r = dist(i,j) / 2
        # d(r)/dc_i = (c_i - c_j) / dist(i,j)
        # d(r)/dc_j = (c_j - c_i) / dist(i,j)
        d = diff[i, j]  # c_i - c_j
        dist_ij = dists[i, j]
        
        if dist_ij > 1e-12:
            # d(-r)/dc_i = -d(r)/dc_i = -(c_i - c_j) / dist(i,j)
            grad[i * 3:i * 3 + 3] = -d / dist_ij
            # d(-r)/dc_j = -d(r)/dc_j = -(c_j - c_i) / dist(i,j) = (c_i - c_j) / dist(i,j)
            grad[j * 3:j * 3 + 3] = d / dist_ij
    
    return grad


def _random_initialization(N=N, seed=None):
    """Generate random initial centres within the cube."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Random positions in [-0.45, 0.45] to stay well inside
    centres = rng.uniform(-0.45, 0.45, size=(N, 3))
    return centres.flatten()


def _grid_initialization(N=N):
    """Generate grid-based initial centres."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].flatten()


def _local_optimize(params, maxiter=200, timeout=10.0):
    """Run local optimization from given parameters."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    remaining = _DEADLINE - time.time()
    if remaining < 2.0:
        return params
    
    try:
        result = minimize(
            _objective,
            params,
            jac=_objective_grad,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': maxiter, 'maxfun': 10000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        return result.x
    except Exception:
        return params


def _anneal_and_optimize(params, temp_start=0.01, temp_end=0.001, n_steps=50, timeout=15.0):
    """Simple simulated annealing followed by local optimization."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    current = params.copy()
    current_obj = _objective(current)
    best = current.copy()
    best_obj = current_obj
    
    t_start = time.time()
    
    for step in range(n_steps):
        if time.time() - t_start > timeout:
            break
        if time.time() > _DEADLINE - 5.0:
            break
        
        temp = temp_start * (temp_end / temp_start) ** (step / max(n_steps - 1, 1))
        
        # Perturbation
        perturbation = np.random.normal(0, temp, size=current.shape)
        proposed = current + perturbation
        
        # Clip to bounds
        proposed = np.clip(proposed, -0.5, 0.5)
        
        proposed_obj = _objective(proposed)
        
        # Accept if better or with probability exp((new-old)/T) for minimization
        if proposed_obj < current_obj:
            current = proposed
            current_obj = proposed_obj
            if current_obj < best_obj:
                best = current.copy()
                best_obj = current_obj
        else:
            delta = proposed_obj - current_obj
            if delta < 0:
                current = proposed
                current_obj = proposed_obj
                if current_obj < best_obj:
                    best = current.copy()
                    best_obj = current_obj
            else:
                prob = np.exp(-delta / max(temp, 1e-12))
                if np.random.random() < prob:
                    current = proposed
                    current_obj = proposed_obj
                    if current_obj < best_obj:
                        best = current.copy()
                        best_obj = current_obj
    
    # Final local optimization
    result = _local_optimize(best, maxiter=300, timeout=10.0)
    result_obj = _objective(result)
    if result_obj < best_obj:
        best = result
        best_obj = result_obj
    
    return best


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    np.random.seed(42)
    
    best_params = None
    best_obj = np.inf
    
    # Strategy 1: Grid initialization + local optimization
    grid_params = _grid_initialization()
    grid_result = _local_optimize(grid_params, maxiter=300, timeout=20.0)
    grid_obj = _objective(grid_result)
    if grid_obj < best_obj:
        best_obj = grid_obj
        best_params = grid_result
    
    # Strategy 2: Multiple random initializations with annealing
    n_random_starts = 5
    for i in range(n_random_starts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        random_params = _random_initialization(seed=100 + i)
        annealed = _anneal_and_optimize(random_params, temp_start=0.02, temp_end=0.001, n_steps=30, timeout=12.0)
        annealed_obj = _objective(annealed)
        
        if annealed_obj < best_obj:
            best_obj = annealed_obj
            best_params = annealed
    
    # Strategy 3: Perturb the best solution and optimize
    if time.time() < _DEADLINE - 15.0:
        for i in range(3):
            if time.time() > _DEADLINE - 10.0:
                break
            
            perturbed = best_params + np.random.normal(0, 0.005, size=best_params.shape)
            perturbed = np.clip(perturbed, -0.5, 0.5)
            result = _local_optimize(perturbed, maxiter=200, timeout=10.0)
            result_obj = _objective(result)
            
            if result_obj < best_obj:
                best_obj = result_obj
                best_params = result
    
    # Final high-precision polish
    if time.time() < _DEADLINE - 5.0:
        final_result = _local_optimize(best_params, maxiter=500, timeout=8.0)
        final_obj = _objective(final_result)
        if final_obj < best_obj:
            best_params = final_result
            best_obj = final_obj
    
    centres = best_params.reshape(N, 3)
    
    # Ensure all values are finite and within bounds
    if not np.all(np.isfinite(centres)):
        # Fallback to grid
        return _grid_initialization().reshape(N, 3)
    
    centres = np.clip(centres, -0.5, 0.5)
    
    return centres
