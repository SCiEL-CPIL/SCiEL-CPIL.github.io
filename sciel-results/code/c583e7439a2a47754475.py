"""Optimized packing of 50 equal circles in a unit square."""
import numpy as np
import time
import random

N = 50

# EVOLVE-BLOCK-START

def _score(centres):
    """Compute the minimum radius: min of half-pair-distances and wall distances."""
    wall = 0.5 - np.abs(centres)
    min_wall = np.min(wall)
    
    n = len(centres)
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_wall, min_pair)


def _compute_forces(centres, r_target, stiffness=1.0):
    """
    Compute repulsion forces treating circles as hard disks with target radius r_target.
    Force on each circle pushes it away from others and walls that are too close.
    Uses a smooth penalty: force magnitude ~ max(0, (r_target - d/2))^2 * direction
    """
    n = len(centres)
    forces = np.zeros_like(centres)
    
    # Pairwise repulsion
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]  # (n,n,2)
    dist = np.sqrt(np.sum(diff**2, axis=2))  # (n,n)
    
    # Penalty: if dist < 2*r_target, push apart
    overlap = np.maximum(0.0, 2.0 * r_target - dist)
    # Force direction: normalize diff, scale by overlap^2 for smoothness
    # Avoid division by zero
    safe_dist = np.where(dist > 1e-12, dist, 1.0)
    force_mag = overlap ** 2 / safe_dist  # (n,n)
    np.fill_diagonal(force_mag, 0.0)
    
    # Sum forces from all pairs
    forces += np.sum(force_mag[:, :, np.newaxis] * diff / safe_dist[:, :, np.newaxis], axis=1)
    
    # Wall repulsion: push away from boundaries
    for axis in range(2):
        # Left wall (x = -0.5)
        dist_to_left = centres[:, axis] + 0.5
        overlap_left = np.maximum(0.0, r_target - dist_to_left)
        forces[:, axis] -= overlap_left ** 2  # push right (positive direction)
        
        # Right wall (x = +0.5)
        dist_to_right = 0.5 - centres[:, axis]
        overlap_right = np.maximum(0.0, r_target - dist_to_right)
        forces[:, axis] += overlap_right ** 2  # push left (negative direction)
    
    return forces * stiffness


def _force_based_optimize(centres, max_time=30.0, seed=42):
    """
    Force-based optimization: iteratively push circles apart using repulsion forces.
    Binary search on r_target to find the largest feasible radius.
    """
    np.random.seed(seed)
    centres = centres.copy()
    
    start_time = time.time()
    deadline = start_time + max_time
    
    # Binary search on radius
    r_lo = 0.05
    r_hi = 0.075
    
    # First, verify r_lo is feasible (should be)
    best_centres = centres.copy()
    best_score = _score(centres)
    
    # Run force relaxation for a given r_target
    def _relax(c, r_target, steps=2000, lr_init=0.001, lr_decay=0.999, seed_val=0):
        np.random.seed(seed_val)
        c = c.copy()
        lr = lr_init
        for step in range(steps):
            if time.time() > deadline:
                break
            f = _compute_forces(c, r_target, stiffness=1.0)
            c += lr * f
            # Add small noise to escape local minima
            c += lr * 0.1 * np.random.randn(*c.shape)
            # Clamp to square
            c = np.clip(c, -0.499, 0.499)
            lr *= lr_decay
        return c
    
    # Binary search: find largest r that can be achieved
    for iteration in range(20):
        if time.time() > deadline - 5.0:
            break
        
        r_mid = (r_lo + r_hi) / 2.0
        relaxed = _relax(centres, r_mid, steps=1500, lr_init=0.002, lr_decay=0.9995, seed_val=iteration)
        
        # Check if the relaxed config actually achieves radius >= r_mid
        actual_r = _score(relaxed)
        
        if actual_r >= r_mid * 0.999:
            # Feasible, try larger
            r_lo = r_mid
            if actual_r > best_score:
                best_score = actual_r
                best_centres = relaxed.copy()
        else:
            # Not feasible, try smaller
            r_hi = r_mid
            # Still keep the relaxed config if it improved
            if actual_r > best_score:
                best_score = actual_r
                best_centres = relaxed.copy()
    
    return best_centres, best_score


def _gradient_refine(centres, max_time=20.0):
    """
    Gradient-based local refinement: analytically compute the gradient of the score
    (which is the minimum of pair-distances/2 and wall-distances) and do steepest ascent.
    """
    np.random.seed(999)
    centres = centres.copy()
    current_score = _score(centres)
    best_centres = centres.copy()
    best_score = current_score
    
    start_time = time.time()
    deadline = start_time + max_time
    
    n = len(centres)
    step_size = 0.002
    
    while time.time() < deadline:
        # Compute gradient of the score function
        # The score is min over all constraints. We use subgradient: average of active constraints' gradients
        wall = 0.5 - np.abs(centres)  # (n, 2)
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_min = dist / 2.0
        
        current_score = min(np.min(wall), np.min(pair_min))
        
        # Find which constraints are active (within tolerance)
        tol = 1e-6
        grad = np.zeros_like(centres)
        
        # Wall constraints
        wall_flat = wall.flatten()
        for i in range(n):
            for k in range(2):
                if wall[i, k] <= current_score + tol:
                    # Gradient of (0.5 - |c[i,k]|) w.r.t. c[i,k]
                    if centres[i, k] > 0:
                        grad[i, k] -= 1.0  # decreasing c[i,k] increases distance from right wall
                    else:
                        grad[i, k] += 1.0  # increasing c[i,k] increases distance from left wall
        
        # Pair constraints
        for i in range(n):
            for j in range(i + 1, n):
                if pair_min[i, j] <= current_score + tol:
                    d = dist[i, j]
                    if d > 1e-12:
                        direction = (centres[i] - centres[j]) / d
                        grad[i] += 0.5 * direction
                        grad[j] -= 0.5 * direction
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad /= grad_norm
        
        # Take a step
        new_centres = centres + step_size * grad
        new_centres = np.clip(new_centres, -0.4999, 0.4999)
        
        new_score = _score(new_centres)
        
        if new_score >= current_score:
            centres = new_centres
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centres = centres.copy()
        else:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-7:
                break
    
    return best_centres, best_score


def _construct_grid():
    """Create initial grid configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _construct_honeycomb():
    """Create a honeycomb-like initial configuration for better packing."""
    centres = []
    for row in range(8):
        for col in range(8):
            if len(centres) >= N:
                break
            x = (col - 3.5) * 0.25
            y = (row - 3.5) * 0.2165
            if col % 2 == 1:
                y += 0.10825
            if -0.45 <= x <= 0.45 and -0.45 <= y <= 0.45:
                centres.append([x, y])
    
    if len(centres) < N:
        while len(centres) < N:
            centres.append([random.uniform(-0.45, 0.45), random.uniform(-0.45, 0.45)])
    
    return np.array(centres[:N])


def construct():
    """Construct the optimal packing of 50 equal circles in a unit square."""
    start_time = time.time()
    total_deadline = start_time + 145.0  # Leave 5s margin for safety
    
    # Initial configurations
    np.random.seed(42)
    initial_configs = [
        _construct_grid(),
        _construct_honeycomb(),
        np.random.uniform(-0.45, 0.45, size=(N, 2)),
    ]
    
    best_centres = None
    best_score = -np.inf
    
    # Phase 1: Force-based optimization from each initial config
    for idx, config in enumerate(initial_configs):
        if time.time() > total_deadline - 20.0:
            break
        
        fb_time = min(30.0, total_deadline - time.time() - 20.0)
        if fb_time <= 0:
            break
        
        centres, score = _force_based_optimize(config, max_time=fb_time, seed=42 + idx)
        
        if score > best_score:
            best_score = score
            best_centres = centres
    
    if best_centres is None:
        return _construct_grid()
    
    # Phase 2: Gradient-based refinement
    grad_time = min(20.0, total_deadline - time.time() - 5.0)
    if grad_time > 0:
        refined_centres, refined_score = _gradient_refine(best_centres, max_time=grad_time)
        if refined_score > best_score:
            best_score = refined_score
            best_centres = refined_centres
    
    # Phase 3: Final quick force-based pass with the refined config
    if time.time() < total_deadline - 3.0:
        final_time = min(15.0, total_deadline - time.time() - 3.0)
        if final_time > 0:
            final_centres, final_score = _force_based_optimize(best_centres, max_time=final_time, seed=777)
            if final_score > best_score:
                best_score = final_score
                best_centres = final_centres
    
    # Ensure all centres are strictly inside the square
    best_centres = np.clip(best_centres, -0.4999, 0.4999)
    
    return best_centres

# EVOLVE-BLOCK-END
