"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    # Pairwise distances
    n = len(centers)
    if n < 2:
        min_dist = np.inf
    else:
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        # Get upper triangle
        iu = np.triu_indices(n, k=1)
        min_dist = dists[iu].min()
    
    # Distance to walls
    wall_dists = np.minimum(0.5 - np.abs(centers), 0.5)  # min over coords
    min_wall = np.min(np.min(0.5 - np.abs(centers), axis=1))
    
    return min(min_dist / 2.0, min_wall)


def _repulsion_energy(centers, r_target=0.11):
    """Energy function to maximize minimum radius. Uses soft repulsion."""
    n = len(centers)
    # Pairwise repulsion
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    
    # Soft penalty for pairs closer than 2*r_target
    # Use a smooth approximation
    eps = 1e-6
    violations = np.maximum(2 * r_target - pair_dists, 0)
    pair_energy = np.sum(violations ** 2)
    
    # Wall constraint penalty
    wall_violations = np.maximum(0, np.abs(centers) + r_target - 0.5)
    wall_energy = np.sum(wall_violations ** 2)
    
    return pair_energy + wall_energy


def _repulsion_grad(centers, r_target=0.11):
    """Gradient of repulsion energy."""
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Avoid division by zero
    dists_safe = np.where(dists > 1e-10, dists, 1e-10)
    
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    violations = np.maximum(2 * r_target - pair_dists, 0)
    
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        if violations[idx] > 0:
            # Gradient of (2r - d)^2 with respect to centers
            d = dists[i, j]
            if d > 1e-10:
                factor = -2 * violations[idx] * (2 * r_target - d) / d
                grad[i] += factor * (centers[i] - centers[j])
                grad[j] -= factor * (centers[i] - centers[j])
    
    # Wall gradient
    wall_violations = np.maximum(0, np.abs(centers) + r_target - 0.5)
    signs = np.sign(centers)
    grad += 2 * wall_violations * signs
    
    return grad


def _local_optimize(centers, r_target, max_iter=200):
    """Local optimization using L-BFGS-B to push spheres apart."""
    n = len(centers)
    bounds = [(-0.5, 0.5)] * (n * 3)
    
    def objective(x):
        c = x.reshape(n, 3)
        return _repulsion_energy(c, r_target)
    
    def grad(x):
        c = x.reshape(n, 3)
        return _repulsion_grad(c, r_target).ravel()
    
    x0 = centers.ravel()
    result = minimize(objective, x0, jac=grad, method='L-BFGS-B', 
                      bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12})
    
    return result.x.reshape(n, 3)


def _generate_bcc_points():
    """Generate BCC-like initial points for better starting configuration."""
    # Use a body-centered cubic arrangement
    # For 100 spheres, we can use a 5x5x4 grid with BCC
    points = []
    
    # Simple approach: use a jittered grid
    m = 5  # 5x5x4 = 100
    for z in range(4):
        for y in range(5):
            for x in range(5):
                # Add some BCC offset
                if (x + y + z) % 2 == 0:
                    offset = 0.1
                else:
                    offset = -0.1
                px = (x + 0.5 + offset * 0.3) / 5 - 0.5
                py = (y + 0.5 + offset * 0.3) / 5 - 0.5
                pz = (z + 0.5 + offset * 0.3) / 4 - 0.5
                points.append([px, py, pz])
    
    return np.array(points[:100])


def _simulated_annealing(centers, r_target, n_iter=500, t_start=0.05, t_end=0.001):
    """Simulated annealing to escape local minima."""
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    n = len(current)
    rng = np.random.RandomState(42)
    
    for it in range(n_iter):
        t = t_start * (t_end / t_start) ** (it / n_iter)
        
        # Random perturbation
        idx = rng.randint(n)
        perturbation = rng.normal(0, t, size=3)
        new_center = current[idx].copy()
        new_center += perturbation
        
        # Keep within bounds
        new_center = np.clip(new_center, -0.5 + r_target, 0.5 - r_target)
        
        # Try the move
        trial = current.copy()
        trial[idx] = new_center
        trial_r = _compute_radius(trial)
        
        # Accept if better, or with probability exp((trial_r - current_r) / t)
        if trial_r >= current_r:
            current = trial
            current_r = trial_r
            if current_r > best_r:
                best = current.copy()
                best_r = current_r
        else:
            accept_prob = np.exp((trial_r - current_r) / max(t, 1e-10))
            if rng.random() < accept_prob:
                current = trial
                current_r = trial_r
    
    return best, best_r


def construct():
    """Construct centers of 100 equal spheres in a unit cube."""
    start_time = time.time()
    deadline = 140  # Leave 10s safety margin
    
    # Initial configuration using BCC-like points
    centers = _generate_bcc_points()
    
    # Multi-start local optimization
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Try to increase radius target iteratively
    r_target = best_radius + 0.001
    
    # First, do local optimization to improve current packing
    for iteration in range(20):
        if time.time() > deadline - 30:
            break
        
        # Local optimization
        optimized = _local_optimize(centers, r_target, max_iter=100)
        
        # Compute new radius
        new_radius = _compute_radius(optimized)
        
        if new_radius > best_radius:
            best_centers = optimized.copy()
            best_radius = new_radius
            centers = optimized.copy()
            # Increase target slightly
            r_target = best_radius + 0.001
        else:
            # Try simulated annealing to escape local minimum
            if time.time() > deadline - 20:
                break
            
            annealed, annealed_r = _simulated_annealing(centers, best_radius, 
                                                        n_iter=200, t_start=0.03, t_end=0.001)
            if annealed_r > best_radius:
                best_centers = annealed.copy()
                best_radius = annealed_r
                centers = annealed.copy()
                r_target = best_radius + 0.001
            else:
                # Try different initial perturbation
                perturbation = np.random.normal(0, 0.02, size=centers.shape)
                centers = np.clip(centers + perturbation, -0.5 + best_radius, 0.5 - best_radius)
    
    # Final polish: high-precision local optimization
    if time.time() < deadline - 5:
        final = _local_optimize(best_centers, best_radius, max_iter=300)
        final_radius = _compute_radius(final)
        if final_radius >= best_radius:
            best_centers = final
    
    return best_centers
# EVOLVE-BLOCK-END
