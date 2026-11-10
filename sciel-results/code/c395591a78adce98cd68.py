"""Physics-based circle packing: maximize minimum distance among 200 points in unit square."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
SEED = 42


def _compute_radius(centers):
    """Compute the packing radius: min of half pairwise distances and half distance to walls."""
    # Pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    # Distance to walls (square is [-0.5, 0.5]^2)
    dist_to_walls = np.min([
        0.5 - centers[:, 0],   # left wall
        centers[:, 0] + 0.5,   # right wall
        0.5 - centers[:, 1],   # bottom wall
        centers[:, 1] + 0.5    # top wall
    ], axis=0)
    min_wall = np.min(dist_to_walls)
    
    return min(min_pair, min_wall)


def _energy_and_grad(centers, r_target=0.035):
    """
    Energy function: penalize pairs closer than r_target and wall violations.
    We use a soft repulsive potential to push points apart.
    """
    centers = centers.reshape(-1, 2)
    
    # Pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    
    # Soft repulsion: for pairs with d < r_target, add penalty
    # Use a smooth potential: max(0, r_target - d)^2
    violations = np.maximum(0, r_target - d)
    pair_energy = np.sum(violations ** 2)
    
    # Wall violations
    wall_violations = []
    for axis in range(2):
        # Left wall: centers[:, axis] < -0.5
        lv = np.maximum(0, -0.5 - centers[:, axis])
        wall_violations.append(lv)
        # Right wall: centers[:, axis] > 0.5
        rv = np.maximum(0, centers[:, axis] - 0.5)
        wall_violations.append(rv)
    
    wall_violations = np.array(wall_violations)
    wall_energy = np.sum(wall_violations ** 2)
    
    total_energy = pair_energy + 10.0 * wall_energy
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    for i in range(N):
        for j in range(i + 1, N):
            dij = d[i, j]
            if dij < r_target and dij > 1e-12:
                # d/dxi max(0, r - dij)^2 = 2*(r - dij) * d/dxi(r - dij)
                # d/dxi(r - dij) = -(xi - xj)/dij for coordinate i
                factor = 2 * (r_target - dij) / dij
                diff = centers[i] - centers[j]
                grad[i] -= factor * diff
                grad[j] += factor * diff
    
    # Wall gradient
    for axis in range(2):
        # Left wall
        lv = np.maximum(0, -0.5 - centers[:, axis])
        grad[:, axis] += 20.0 * lv
        # Right wall
        rv = np.maximum(0, centers[:, axis] - 0.5)
        grad[:, axis] -= 20.0 * rv
    
    return total_energy, grad


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_radius = 0.0
    best_centers = None
    
    # Initial grid configuration
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initial_centers = grid[:N].copy()
    
    r_init = _compute_radius(initial_centers)
    best_radius = r_init
    best_centers = initial_centers.copy()
    
    # Try multiple random perturbations and optimize
    np.random.seed(SEED)
    
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Perturb the initial configuration
        noise_scale = 0.005 + 0.01 * (restart % 5)
        centers = initial_centers + np.random.randn(N, 2) * noise_scale
        
        # Clamp to stay roughly inside
        centers = np.clip(centers, -0.49, 0.49)
        
        # Use L-BFGS-B to minimize energy
        x0 = centers.flatten()
        
        def obj(x):
            c = x.reshape(N, 2)
            e, g = _energy_and_grad(c)
            return e, g.flatten()
        
        try:
            result = minimize(obj, x0, method='L-BFGS-B', jac=True,
                             options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8,
                                      'maxfun': 10000})
        except Exception:
            continue
        
        if time.time() > deadline:
            break
        
        opt_centers = result.x.reshape(N, 2)
        
        # Validate
        if not np.all(np.isfinite(opt_centers)):
            continue
        if np.any(opt_centers < -0.5 - 1e-9) or np.any(opt_centers > 0.5 + 1e-9):
            continue
        
        r_opt = _compute_radius(opt_centers)
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = opt_centers.copy()
    
    # Final local refinement using a tighter target
    if time.time() < deadline - 5.0 and best_centers is not None:
        # Try to improve by adjusting r_target to be slightly above current best
        r_target = best_radius * 1.001
        x0 = best_centers.flatten()
        
        def obj2(x):
            c = x.reshape(N, 2)
            e, g = _energy_and_grad(c, r_target=r_target)
            return e, g.flatten()
        
        try:
            result2 = minimize(obj2, x0, method='L-BFGS-B', jac=True,
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10,
                                       'maxfun': 5000})
            opt_centers2 = result2.x.reshape(N, 2)
            if np.all(np.isfinite(opt_centers2)) and \
               np.all(opt_centers2 >= -0.5 - 1e-9) and np.all(opt_centers2 <= 0.5 + 1e-9):
                r_opt2 = _compute_radius(opt_centers2)
                if r_opt2 > best_radius:
                    best_radius = r_opt2
                    best_centers = opt_centers2.copy()
        except Exception:
            pass
    
    return best_centers


if __name__ == "__main__":
    centers = construct()
    r = _compute_radius(centers)
    print(f"Packing radius: {r:.12f}")
    print(f"Best known:     0.036612798904")
    print(f"Ratio:          {r / 0.036612798904:.6f}")
