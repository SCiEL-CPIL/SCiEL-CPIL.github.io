"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

N = 50

_deadline = 0.0

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    """
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _violation(centers, r):
    """
    Compute total violation for radius r.
    Violation = sum of max(0, r - d/2) for pairs + sum of max(0, r - wall_dist) for walls.
    Lower is better. Zero means feasible.
    """
    centers = np.asarray(centers)
    n = len(centers)
    
    # Pairwise violations
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    pair_violation = np.sum(np.maximum(0, r - dists / 2.0))
    
    # Wall violations
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(wall_dists_x, wall_dists_y)
    wall_violation = np.sum(np.maximum(0, r - wall_dists))
    
    return pair_violation + wall_violation

def _grad_violation(centers, r):
    """
    Compute gradient of violation with respect to centers.
    """
    centers = np.asarray(centers)
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Pairwise gradients
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Only consider upper triangle to avoid double counting
    for i in range(n):
        for j in range(i+1, n):
            if dists[i, j] / 2.0 < r:
                d = dists[i, j]
                if d > 1e-12:
                    factor = 0.5 / d
                    grad[i] -= factor * (centers[i] - centers[j])
                    grad[j] += factor * (centers[i] - centers[j])
    
    # Wall gradients
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    
    for i in range(n):
        if wall_dists_x[i] < r:
            grad[i, 0] += np.sign(centers[i, 0])
        if wall_dists_y[i] < r:
            grad[i, 1] += np.sign(centers[i, 1])
    
    return grad

def _force_directed_layout(centers, r_target, max_iter=2000, lr=0.002):
    """
    Force-directed layout using repulsive forces between circles and attractive forces
    toward the square center. Force magnitude is inversely proportional to distance^2.
    """
    centers = centers.copy()
    n = len(centers)
    
    for it in range(max_iter):
        if time.time() > _deadline:
            break
        
        forces = np.zeros_like(centers)
        
        # Repulsive forces between pairs
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Only apply force when distance < 2*r_target (circles would overlap)
        # Use inverse square law for force magnitude
        for i in range(n):
            for j in range(i+1, n):
                d = dists[i, j]
                if d < 2 * r_target and d > 1e-12:
                    # Force proportional to 1/d^2, direction away from each other
                    force_mag = 1.0 / (d * d)
                    direction = diff[i, j] / d
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction
        
        # Attractive forces toward center (to keep circles inside and centered)
        for i in range(n):
            # Gentle attraction to center
            forces[i] -= 0.1 * centers[i]
        
        # Wall repulsion: if too close to wall, push inward
        for i in range(n):
            # Left wall
            if centers[i, 0] < -0.5 + r_target:
                forces[i, 0] += (r_target - (centers[i, 0] + 0.5)) * 10.0
            # Right wall
            if centers[i, 0] > 0.5 - r_target:
                forces[i, 0] -= (centers[i, 0] - (0.5 - r_target)) * 10.0
            # Bottom wall
            if centers[i, 1] < -0.5 + r_target:
                forces[i, 1] += (r_target - (centers[i, 1] + 0.5)) * 10.0
            # Top wall
            if centers[i, 1] > 0.5 - r_target:
                forces[i, 1] -= (centers[i, 1] - (0.5 - r_target)) * 10.0
        
        # Update positions
        centers += lr * forces
        centers = np.clip(centers, -0.499, 0.499)
        
        # Check convergence
        if np.max(np.abs(forces)) < 1e-8:
            break
    
    return centers

def _sa_search(centers, rng, budget_time):
    """
    Simulated annealing to improve packing.
    """
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    T_start = 0.03
    T_end = 1e-7
    max_steps = 1000000
    
    for step in range(max_steps):
        if time.time() - start_time > budget_time:
            break
        
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Random perturbation
        idx = rng.integers(0, N)
        step_size = 0.015 * (1.0 + 0.3 * np.cos(np.pi * step / max_steps))
        delta = rng.normal(0, step_size, 2)
        
        new = current.copy()
        new[idx] += delta
        new = np.clip(new, -0.5, 0.5)
        
        new_r = _compute_radius(new)
        
        if new_r > current_r:
            current = new
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            delta_r = new_r - current_r
            if delta_r > -5 * T:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    current = new
                    current_r = new_r
    
    return best, best_r

def _gradient_polish(centers, r_target, max_iter=500):
    """
    Use gradient descent to minimize violation for a target radius.
    """
    x = centers.flatten().copy()
    lr = 0.005
    
    for i in range(max_iter):
        if time.time() > _deadline:
            break
        
        v = _violation(x.reshape(-1, 2), r_target)
        if v < 1e-10:
            break
        
        g = _grad_violation(x.reshape(-1, 2), r_target).flatten()
        x -= lr * g
        x = np.clip(x, -0.5, 0.5)
    
    return x.reshape(-1, 2)

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 145.0  # 5s safety margin
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    rng = np.random.default_rng(42)
    best_centers = best_centers + rng.normal(0, 0.01, best_centers.shape)
    best_centers = np.clip(best_centers, -0.49, 0.49)
    
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: Force-directed layout to escape grid local minimum
    remaining = _deadline - time.time()
    if remaining > 20:
        # Use current radius as target for force-directed layout
        fd_centers = _force_directed_layout(best_centers, best_radius, max_iter=1000, lr=0.003)
        fd_radius = _compute_radius(fd_centers)
        if fd_radius > best_radius:
            best_centers = fd_centers
            best_radius = fd_radius
    
    # Phase 2: Simulated Annealing
    remaining = _deadline - time.time()
    sa_budget = min(70.0, remaining * 0.6)
    best_centers, best_radius = _sa_search(best_centers, rng, sa_budget)
    
    # Phase 3: Binary search on radius with violation minimization
    remaining = _deadline - time.time()
    if remaining > 30:
        lo = best_radius
        hi = 0.08  # Upper bound
        
        # Try to find better radius via binary search
        for _ in range(12):
            if time.time() > _deadline - 10:
                break
            
            mid = (lo + hi) / 2
            # Try to find feasible configuration for radius mid
            # Use gradient descent to minimize violation
            trial = best_centers.copy()
            trial = _gradient_polish(trial, mid, max_iter=200)
            
            trial_r = _compute_radius(trial)
            if trial_r >= mid - 1e-9:
                lo = mid
                best_centers = trial
                best_radius = trial_r
            else:
                hi = mid
    
    # Phase 4: Final SA polish
    remaining = _deadline - time.time()
    if remaining > 5:
        best_centers, best_radius = _sa_search(best_centers, rng, remaining - 1)
    
    # Final check
    if not np.all(np.isfinite(best_centers)):
        return grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
