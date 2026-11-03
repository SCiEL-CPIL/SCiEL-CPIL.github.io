import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def _compute_radius(pos):
    """Compute the maximum valid radius for given positions."""
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Wall distances
    d_walls = HALF - np.abs(pos)
    min_wall = np.min(d_walls)
    
    return min(min_pair, min_wall)

def _repulsion_energy(flat_pos, power=4):
    """Repulsion energy: sum of 1/d^power for all pairs, plus wall penalties."""
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise repulsion
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Avoid division by zero
    dists_sq = np.maximum(dists_sq, 1e-12)
    energy = np.sum(dists_sq ** (-power / 2.0))
    
    # Wall penalty: push away from walls
    d_walls = HALF - np.abs(pos)
    wall_penalty = np.sum(np.maximum(0, -d_walls) ** 2) * 1000
    energy += wall_penalty
    
    return energy

def _repulsion_grad(flat_pos, power=4):
    """Gradient of repulsion energy."""
    pos = flat_pos.reshape(N, 3)
    grad = np.zeros_like(pos)
    
    # Pairwise: d/dx_i sum_{j!=i} 1/|x_i - x_j|^power
    # = -power * sum_{j!=i} (x_i - x_j) / |x_i - x_j|^(power+2)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1.0)  # avoid div by 0 on diagonal
    
    dists = np.sqrt(dists_sq)
    # weight[i,j] = -power / dists[i,j]^(power+2)
    weight = -power * dists_sq ** (-(power + 2) / 2.0)
    np.fill_diagonal(weight, 0.0)
    
    # grad[i] = sum_j weight[i,j] * diff[i,j]
    grad = np.sum(weight[:, :, np.newaxis] * diff, axis=1)
    
    # Wall gradient: d/dx_i sum max(0, -d_wall_i)^2
    d_walls = HALF - np.abs(pos)
    wall_mask = d_walls < 0
    if np.any(wall_mask):
        # d/dx_i (d_wall_i)^2 = 2*d_wall_i * d(d_wall_i)/d(x_i)
        # d(d_wall_i)/d(x_ik) = -sign(x_ik)
        signs = np.sign(pos)
        wall_grad = 2 * d_walls * 1000 * (-signs)
        wall_grad[~wall_mask] = 0
        grad += wall_grad
    
    return grad.flatten()

def _local_optimize(pos_init, power=4, maxiter=500, ftol=1e-12, gtol=1e-10, deadline=None):
    """Local optimization using repulsion energy."""
    flat_init = pos_init.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def obj(x):
        return _repulsion_energy(x, power)
    
    def jac(x):
        return _repulsion_grad(x, power)
    
    try:
        res = minimize(
            obj, flat_init, jac=jac,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
        )
        if deadline is not None and time.time() > deadline:
            return pos_init
        return res.x.reshape(N, 3)
    except Exception:
        return pos_init

def _grid_init(m):
    """Create a grid of m^3 points, take first N."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _random_init(rng, margin=0.05):
    """Random initialization inside cube with margin."""
    return rng.uniform(-HALF + margin, HALF - margin, size=(N, 3))

def _polish(pos, deadline):
    """High-precision polish pass with tighter tolerances."""
    if time.time() > deadline:
        return pos
    
    # Multiple rounds of tight optimization
    for _ in range(3):
        if time.time() > deadline:
            break
        pos = _local_optimize(pos, power=6, maxiter=1000, ftol=1e-14, gtol=1e-12, deadline=deadline)
    
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Multiple initialization strategies
    best_pos = None
    best_r = -np.inf
    
    # Strategy 1: Grid-based starts with different m values
    for m in [4, 5, 6]:
        if time.time() > deadline:
            break
        pos_init = _grid_init(m)
        # Add small random perturbation to break symmetry
        pos_init = pos_init + rng.normal(0, 0.01, size=(N, 3))
        pos_init = np.clip(pos_init, -HALF + 0.01, HALF - 0.01)
        
        pos_opt = _local_optimize(pos_init, power=4, maxiter=300, ftol=1e-10, gtol=1e-8, deadline=deadline)
        r = _compute_radius(pos_opt)
        
        if r > best_r:
            best_r = r
            best_pos = pos_opt.copy()
    
    # Strategy 2: Random starts
    n_random = 5
    for i in range(n_random):
        if time.time() > deadline:
            break
        pos_init = _random_init(rng, margin=0.05)
        pos_opt = _local_optimize(pos_init, power=4, maxiter=300, ftol=1e-10, gtol=1e-8, deadline=deadline)
        r = _compute_radius(pos_opt)
        
        if r > best_r:
            best_r = r
            best_pos = pos_opt.copy()
    
    # Strategy 3: Perturb best solution
    if best_pos is not None:
        for scale in [0.03, 0.05, 0.08]:
            if time.time() > deadline:
                break
            pos_perturbed = best_pos + rng.normal(0, scale, size=(N, 3))
            pos_perturbed = np.clip(pos_perturbed, -HALF + 0.01, HALF - 0.01)
            pos_opt = _local_optimize(pos_perturbed, power=4, maxiter=300, ftol=1e-10, gtol=1e-8, deadline=deadline)
            r = _compute_radius(pos_opt)
            
            if r > best_r:
                best_r = r
                best_pos = pos_opt.copy()
    
    # Final polish
    if best_pos is not None and time.time() < deadline:
        best_pos = _polish(best_pos, deadline)
        best_r = _compute_radius(best_pos)
    
    # Ensure valid output
    if best_pos is None:
        best_pos = _grid_init(5)
    
    # Validate
    if not np.all(np.isfinite(best_pos)):
        best_pos = _grid_init(5)
    
    return best_pos
