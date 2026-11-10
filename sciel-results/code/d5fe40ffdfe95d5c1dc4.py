"""Improved packing of 50 equal spheres in a unit cube using gradient-based optimization."""
import numpy as np
import time
import math
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _soft_min(values, beta=50.0):
    """Soft minimum of an array of values."""
    return -np.logaddexp.reduce(np.log(values)) / beta if len(values) > 0 else 0.0

def _score_kdtree(centers, beta=50.0):
    """Compute soft-minimum score using KDTree for speed."""
    # Wall distances
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    # Pairwise distances via KDTree
    tree = cKDTree(centers)
    # Query k=2 nearest neighbors (self + nearest)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    # Soft min combines both constraints
    # Score is min(min_pair/2, min_wall)
    # Use soft min for differentiability
    a = min_pair / 2.0
    b = min_wall
    # Soft min of a and b
    return -math.log(math.exp(-beta * a) + math.exp(-beta * b)) / beta

def _exact_score(centers):
    """Exact minimum radius score."""
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    return min(min_pair / 2.0, min_wall)

def _is_valid(centers):
    """Check if centers are valid: finite, inside cube."""
    if centers.shape != (N, 3):
        return False
    if not np.all(np.isfinite(centers)):
        return False
    if np.any(np.abs(centers) > 0.5 + 1e-10):
        return False
    return True

def _objective_flat(x, beta=50.0):
    """Objective function for optimizer: minimize negative soft-min score."""
    centers = x.reshape(N, 3)
    # Clamp to stay inside cube
    centers = np.clip(centers, -0.5, 0.5)
    score = _score_kdtree(centers, beta=beta)
    return -score

def _gradient_flat(x, beta=50.0):
    """Numerical gradient for the objective."""
    centers = x.reshape(N, 3)
    centers = np.clip(centers, -0.5, 0.5)
    
    # Compute gradient numerically
    eps = 1e-7
    grad = np.zeros_like(x)
    for i in range(N):
        for j in range(3):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i*3+j] += eps
            x_minus[i*3+j] -= eps
            f_plus = _objective_flat(x_plus, beta=beta)
            f_minus = _objective_flat(x_minus, beta=beta)
            grad[i*3+j] = (f_plus - f_minus) / (2 * eps)
    return grad

def _initial_guess_pfoertner_style():
    """
    Generate an initial guess inspired by Pfoertner's packing.
    The best known packing has a structured arrangement.
    We use a 5x5x2 grid with some perturbations.
    """
    centers = np.zeros((N, 3))
    
    # Use a 5x5x2 arrangement: 50 = 5*5*2
    # Place in two layers
    m = 5
    spacing = 1.0 / (m + 1)  # ~0.1667
    
    idx = 0
    for layer in range(2):
        z = -0.5 + (layer + 0.5) * (1.0 / 2)
        for i in range(m):
            for j in range(m):
                x = -0.5 + (i + 0.5) * spacing
                y = -0.5 + (j + 0.5) * spacing
                centers[idx] = [x, y, z]
                idx += 1
    
    return centers

def _initial_guess_random(seed=0):
    """Random initial guess."""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-0.4, 0.4, size=(N, 3))
    return centers

def _initial_guess_grid_perturbed(seed=0):
    """Grid with random perturbation."""
    rng = np.random.RandomState(seed)
    centers = _initial_guess_pfoertner_style()
    centers += rng.uniform(-0.02, 0.02, size=(N, 3))
    centers = np.clip(centers, -0.5, 0.5)
    return centers

def _local_optimize(centers, time_limit=20.0, seed=0):
    """
    Optimize a set of centers using L-BFGS-B with soft-minimum objective.
    """
    start_time = time.time()
    
    x0 = centers.flatten()
    
    # First pass with coarser beta
    beta_coarse = 20.0
    result1 = minimize(
        _objective_flat,
        x0,
        args=(beta_coarse,),
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (N * 3),
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if time.time() - start_time > time_limit * 0.7:
        centers_opt = result1.x.reshape(N, 3)
        return np.clip(centers_opt, -0.5, 0.5)
    
    # Second pass with finer beta
    beta_fine = 80.0
    x1 = result1.x
    result2 = minimize(
        _objective_flat,
        x1,
        args=(beta_fine,),
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (N * 3),
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    centers_opt = result2.x.reshape(N, 3)
    return np.clip(centers_opt, -0.5, 0.5)

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    best_centers = None
    best_score = 0.0
    
    # Multiple starting points
    initializers = [
        ("pfoertner", _initial_guess_pfoertner_style()),
        ("grid_perturbed_0", _initial_guess_grid_perturbed(0)),
        ("grid_perturbed_1", _initial_guess_grid_perturbed(1)),
        ("grid_perturbed_2", _initial_guess_grid_perturbed(2)),
        ("random_0", _initial_guess_random(0)),
        ("random_1", _initial_guess_random(1)),
    ]
    
    for name, init_centers in initializers:
        if time.time() > deadline - 5.0:
            break
        
        # Evaluate initial score
        init_score = _exact_score(init_centers)
        if init_score > best_score:
            best_score = init_score
            best_centers = init_centers.copy()
        
        # Optimize
        opt_centers = _local_optimize(init_centers, time_limit=15.0)
        opt_score = _exact_score(opt_centers)
        
        if opt_score > best_score:
            best_score = opt_score
            best_centers = opt_centers.copy()
        
        # If we still have time, do another round of optimization from the improved solution
        if time.time() < deadline - 10.0 and opt_score > 0.13:
            opt_centers2 = _local_optimize(opt_centers, time_limit=10.0)
            opt_score2 = _exact_score(opt_centers2)
            if opt_score2 > best_score:
                best_score = opt_score2
                best_centers = opt_centers2.copy()
    
    # Final refinement on the best solution if time permits
    if time.time() < deadline - 5.0 and best_centers is not None:
        final_centers = _local_optimize(best_centers, time_limit=10.0)
        final_score = _exact_score(final_centers)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_centers.copy()
    
    if best_centers is None:
        best_centers = _initial_guess_pfoertner_style()
    
    return best_centers

# EVOLVE-BLOCK-END
