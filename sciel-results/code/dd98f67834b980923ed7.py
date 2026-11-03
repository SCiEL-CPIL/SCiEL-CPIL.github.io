"""Optimal packing of 50 equal circles in a unit square."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 110.0  # seconds, safety margin before 120s limit


def _radius(positions):
    """Compute the maximum common radius for given circle centers."""
    # Distance to walls
    wall_dists = np.min([
        0.5 - np.min(positions[:, 0]),
        0.5 + np.min(positions[:, 1]),
        0.5 - np.max(positions[:, 0]),
        0.5 - np.max(positions[:, 1]),
    ])
    # Pairwise distances
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(wall_dists, min_pair_dist / 2.0)


def _soft_min_radius(positions, alpha=50.0):
    """Differentiable approximation of the radius using soft-min."""
    # Wall distances
    wall_dists = np.array([
        0.5 - np.min(positions[:, 0]),
        0.5 + np.min(positions[:, 1]),
        0.5 - np.max(positions[:, 0]),
        0.5 - np.max(positions[:, 1]),
    ])
    
    # Pairwise distances / 2
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    # Combine all constraints
    all_constraints = np.concatenate([wall_dists, dists[np.triu_indices(n, k=1)]])
    
    # Soft min via log-sum-exp (negative for min)
    # soft_min = -logsumexp(-alpha * x) / alpha
    # For numerical stability:
    a = alpha * all_constraints
    a_max = np.max(a)
    soft_min = -a_max / alpha - np.log(np.sum(np.exp(a - a_max))) / alpha
    return soft_min


def _objective_neg_soft_min(positions_flat, alpha=50.0):
    """Negative of soft-min radius for minimization."""
    positions = positions_flat.reshape(-1, 2)
    return -_soft_min_radius(positions, alpha)


def _gradient_neg_soft_min(positions_flat, alpha=50.0):
    """Gradient of negative soft-min radius (numerical)."""
    eps = 1e-7
    grad = np.zeros_like(positions_flat)
    f0 = _objective_neg_soft_min(positions_flat, alpha)
    for i in range(len(positions_flat)):
        positions_flat[i] += eps
        f1 = _objective_neg_soft_min(positions_flat, alpha)
        grad[i] = (f1 - f0) / eps
        positions_flat[i] -= eps
    return grad


def _grid_init(n_cols=7, n_rows=8):
    """Create a grid-like initial configuration."""
    positions = []
    for r in range(n_rows):
        for c in range(n_cols):
            if len(positions) < N:
                x = (c + 0.5) / n_cols - 0.5
                y = (r + 0.5) / n_rows - 0.5
                positions.append([x, y])
    return np.array(positions[:N])


def _random_init():
    """Random initial configuration."""
    return np.random.uniform(-0.45, 0.45, size=(N, 2))


def _jitter(positions, scale=0.01):
    """Add small random jitter."""
    return positions + np.random.normal(0, scale, size=positions.shape)


def construct():
    start_time = time.time()
    
    # Try multiple strategies and keep the best
    best_positions = None
    best_radius = 0.0
    
    # Strategy 1: Grid initialization with optimization
    init1 = _grid_init()
    
    # Strategy 2: Random initialization
    init2 = _random_init()
    
    # Strategy 3: Another grid with different aspect ratio
    init3 = _grid_init(n_cols=5, n_rows=10)
    
    inits = [init1, init2, init3]
    
    for idx, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Progressive alpha optimization
        alphas = [5.0, 10.0, 20.0, 50.0, 100.0]
        current = init.copy()
        
        for alpha in alphas:
            if time.time() - start_time > DEADLINE - 15:
                break
            
            # Try L-BFGS-B
            try:
                result = minimize(
                    _objective_neg_soft_min,
                    current.flatten(),
                    method='L-BFGS-B',
                    jac=_gradient_neg_soft_min,
                    args=(alpha,),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.success or result.fun < _objective_neg_soft_min(current.flatten(), alpha):
                    current = result.x.reshape(-1, 2)
            except Exception:
                pass
            
            # Clamp to square
            current[:, 0] = np.clip(current[:, 0], -0.49, 0.49)
            current[:, 1] = np.clip(current[:, 1], -0.49, 0.49)
        
        # Final refinement with high alpha
        if time.time() - start_time < DEADLINE - 10:
            try:
                result = minimize(
                    _objective_neg_soft_min,
                    current.flatten(),
                    method='L-BFGS-B',
                    jac=_gradient_neg_soft_min,
                    args=(200.0,),
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result.fun < _objective_neg_soft_min(current.flatten(), 200.0):
                    current = result.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Evaluate actual radius
        current[:, 0] = np.clip(current[:, 0], -0.5, 0.5)
        current[:, 1] = np.clip(current[:, 1], -0.5, 0.5)
        r = _radius(current)
        
        if r > best_radius:
            best_radius = r
            best_positions = current.copy()
        
        # If we have time, try simulated annealing perturbation
        if time.time() - start_time < DEADLINE - 20 and best_radius > 0:
            # Try a few SA-like perturbations from best
            for sa_iter in range(5):
                if time.time() - start_time > DEADLINE - 10:
                    break
                perturbed = _jitter(best_positions, scale=0.005)
                perturbed[:, 0] = np.clip(perturbed[:, 0], -0.49, 0.49)
                perturbed[:, 1] = np.clip(perturbed[:, 1], -0.49, 0.49)
                
                try:
                    result = minimize(
                        _objective_neg_soft_min,
                        perturbed.flatten(),
                        method='L-BFGS-B',
                        jac=_gradient_neg_soft_min,
                        args=(50.0,),
                        options={'maxiter': 100, 'ftol': 1e-10}
                    )
                    if result.success:
                        candidate = result.x.reshape(-1, 2)
                        candidate[:, 0] = np.clip(candidate[:, 0], -0.5, 0.5)
                        candidate[:, 1] = np.clip(candidate[:, 1], -0.5, 0.5)
                        r_cand = _radius(candidate)
                        if r_cand > best_radius:
                            best_radius = r_cand
                            best_positions = candidate.copy()
                except Exception:
                    pass
    
    # Final safety: ensure valid output
    if best_positions is None:
        best_positions = init1
    
    best_positions[:, 0] = np.clip(best_positions[:, 0], -0.5, 0.5)
    best_positions[:, 1] = np.clip(best_positions[:, 1], -0.5, 0.5)
    
    # Verify no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.allclose(best_positions[i], best_positions[j], atol=1e-10):
                best_positions[j, 0] += 1e-6
    
    return best_positions
