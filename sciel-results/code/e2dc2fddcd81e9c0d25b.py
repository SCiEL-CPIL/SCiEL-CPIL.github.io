"""Search-based approach for packing 50 equal circles in a unit square."""
import time
import numpy as np
import random

N = 50
SEED = 42
DEADLINE_FRACTION = 0.85  # Use 85% of the 150s budget as safety margin

def _compute_radius(centers):
    """Compute the largest common radius for given centers.
    
    Returns min( min_{i<j} |c_i - c_j|/2, min_i min_k (0.5 - |c_ik|) )
    """
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    
    if min_boundary <= 0:
        return 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _initial_grid():
    """Generate initial grid configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].astype(float)

def _hexagonal_initializer():
    """Generate a hexagonal packing pattern as initial configuration."""
    # Hexagonal packing: rows offset by half spacing
    # For 50 circles, use roughly 8 columns x 7 rows with hexagonal offset
    centers = []
    # Try different arrangements
    for cols in range(5, 11):
        rows = int(np.ceil(N / cols))
        if cols * rows < N:
            continue
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        # Hexagonal offset: odd rows shifted by half spacing
        for row in range(rows):
            for col in range(cols):
                if len(centers) >= N:
                    break
                x = (col + 0.5) * spacing_x - 0.5
                if row % 2 == 1:
                    x += spacing_x / 2
                y = (row + 0.5) * spacing_y - 0.5
                centers.append([x, y])
        if len(centers) >= N:
            break
    if len(centers) < N:
        # Fallback to grid
        return _initial_grid()
    return np.array(centers[:N]).astype(float)

def _random_initializer(n_points=N, seed=None):
    """Generate random initial positions within the square."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.5 + 0.05, 0.5 - 0.05, size=(n_points, 2))

def _force_relaxation(centers, n_iterations=5000, learning_rate=0.01, deadline=None):
    """Physics-based relaxation: circles repel each other and are pushed to maximize min distance.
    
    Uses a force model where:
    - Close pairs repel strongly
    - Boundary proximity adds repulsive force
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    lr = learning_rate
    for it in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        c = best_centers.copy()
        forces = np.zeros_like(c)
        
        # Pairwise repulsion: force proportional to 1/d^2 for close pairs
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Only consider pairs that are too close (within 2*r + margin)
        # Use a soft repulsion
        min_dist = 0.05  # Minimum distance threshold for strong repulsion
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d < min_dist:
                    # Strong repulsion
                    force_mag = (min_dist - d) / min_dist
                    direction = (c[i] - c[j]) / (d + 1e-10)
                    forces[i] += direction * force_mag
                    forces[j] -= direction * force_mag
        
        # Boundary repulsion
        for i in range(N):
            for k in range(2):
                if c[i, k] > 0:
                    dist_to_boundary = 0.5 - c[i, k]
                    if dist_to_boundary < 0.05:
                        forces[i, k] -= (0.05 - dist_to_boundary) / 0.05
                else:
                    dist_to_boundary = 0.5 + c[i, k]
                    if dist_to_boundary < 0.05:
                        forces[i, k] += (0.05 - dist_to_boundary) / 0.05
        
        # Update positions
        best_centers += lr * forces
        np.clip(best_centers, -0.5, 0.5, out=best_centers)
        
        new_radius = _compute_radius(best_centers)
        if new_radius > best_radius:
            best_radius = new_radius
        else:
            # Revert and reduce learning rate
            best_centers = c
            lr *= 0.95
        
        # Early stopping if converged
        if lr < 1e-6:
            break
    
    return best_centers, best_radius

def _local_search(centers, n_iterations=2000, deadline=None):
    """Improved local search: try moving pairs of points and single points."""
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    rng = np.random.RandomState(42)
    
    step_size = 0.02
    
    for i in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Randomly choose: move one point (70%) or two points (30%)
        if rng.random() < 0.7:
            idxs = [rng.randint(N)]
        else:
            idxs = list(rng.choice(N, size=2, replace=False))
        
        orig = best_centers[idxs].copy()
        
        # Try random perturbation
        for idx in idxs:
            perturbation = rng.normal(0, step_size, size=2)
            best_centers[idx] = orig[idxs.index(idx)] + perturbation
        
        np.clip(best_centers, -0.5, 0.5, out=best_centers)
        
        new_radius = _compute_radius(best_centers)
        
        if new_radius > best_radius:
            best_radius = new_radius
        else:
            best_centers[idxs] = orig
    
    return best_centers, best_radius

def construct():
    """Main function that searches for the best packing configuration."""
    start_time = time.time()
    total_time_budget = 150.0
    deadline = start_time + total_time_budget * DEADLINE_FRACTION
    
    best_centers = None
    best_radius = 0.0
    
    # Try multiple initialization strategies
    initializers = [
        _initial_grid,
        _hexagonal_initializer,
    ]
    
    for init_func in initializers:
        if time.time() > deadline:
            break
        
        centers = init_func()
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    # Try random initializations with force relaxation
    n_attempts = 15
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        seed = SEED + attempt * 1000
        centers = _random_initializer(seed=seed)
        
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
        
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
        
        # Use force relaxation
        n_iters = max(1000, int(5000 * remaining_time / 20.0))
        optimized, r_opt = _force_relaxation(centers, n_iterations=n_iters, deadline=deadline)
        
        r = _compute_radius(optimized)
        if r > best_radius:
            best_radius = r
            best_centers = optimized.copy()
    
    # If we have time, do local search on the best solution
    if time.time() < deadline - 3.0:
        remaining = deadline - time.time() - 3.0
        n_iters = max(500, int(5000 * remaining / 30.0))
        best_centers, best_radius = _local_search(best_centers, n_iterations=n_iters, deadline=deadline)
    
    # Final check
    final_radius = _compute_radius(best_centers)
    if final_radius <= 0:
        best_centers = _initial_grid()
    
    return best_centers

# EVOLVE-BLOCK-END
