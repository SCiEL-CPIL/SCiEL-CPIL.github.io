"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_radius(centers):
    """Compute the valid common radius for a set of centers in [-0.5, 0.5]^2."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dist_boundary = np.min(np.stack([
        centers[:, 0] + 0.5,
        0.5 - centers[:, 0],
        centers[:, 1] + 0.5,
        0.5 - centers[:, 1]
    ], axis=1), axis=1)
    r_boundary = np.min(dist_boundary)
    if r_boundary < 0:
        return 0.0
    # Pairwise distances
    # Use broadcasting for all pairs
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    r_pairs = np.min(dists) / 2.0
    return min(r_boundary, r_pairs)

def _objective(centers):
    """Negative radius for minimization."""
    return -_compute_radius(centers)

def _relaxation_step(centers, r, steps=100, lr=0.001):
    """Simple relaxation: push overlapping circles apart."""
    centers = centers.copy()
    for _ in range(steps):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 2 * r and dist > 1e-12:
                    # Push apart
                    push = (2 * r - dist) / dist * 0.5
                    centers[i] += push * d
                    centers[j] -= push * d
                    moved = True
        # Keep inside boundary
        for i in range(N):
            for k in range(2):
                if centers[i, k] < -0.5 + r:
                    centers[i, k] = -0.5 + r
                elif centers[i, k] > 0.5 - r:
                    centers[i, k] = 0.5 - r
    return centers

def _make_grid_start():
    """Create a grid-based initial configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _make_random_start(seed=None):
    """Create a random initial configuration inside the square."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))

def _optimize_from_start(centers, time_budget=100):
    """Optimize starting from given centers using Nelder-Mead with time budget."""
    start_time = time.time()
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    
    # Try multiple iterations of Nelder-Mead with different options
    options = {
        'maxiter': 2000,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True,
        'maxfev': 5000
    }
    
    for iteration in range(10):
        if time.time() - start_time > time_budget:
            break
        result = minimize(
            _objective,
            best_centers,
            method='Nelder-Mead',
            options=options
        )
        new_r = _compute_radius(result.x)
        if new_r > best_r:
            best_r = new_r
            best_centers = result.x.copy()
        else:
            # If no improvement, try a perturbation
            break
    
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Start with grid configuration
    best_centers = _make_grid_start()
    best_r = _compute_radius(best_centers)
    
    # Try multiple random starts
    num_starts = 20
    for i in range(num_starts):
        if time.time() > deadline:
            break
        
        # Mix of random and perturbed grid starts
        if i == 0:
            centers = _make_grid_start()
        elif i < 5:
            centers = _make_random_start(seed=i * 42 + 7)
        else:
            # Perturbed grid
            centers = _make_grid_start()
            rng = np.random.default_rng(i * 13 + 3)
            centers += rng.normal(0, 0.02, size=(N, 2))
            # Clip to valid range
            centers = np.clip(centers, -0.49, 0.49)
        
        # Relaxation step to resolve obvious overlaps
        initial_r = _compute_radius(centers)
        # Estimate a reasonable radius for relaxation
        relax_r = max(0.05, initial_r * 0.8)
        centers = _relaxation_step(centers, relax_r, steps=50, lr=0.001)
        
        # Optimize with Nelder-Mead
        time_remaining = deadline - time.time()
        if time_remaining < 2:
            break
        opt_time = min(5, time_remaining - 1)
        centers, r = _optimize_from_start(centers, time_budget=opt_time)
        
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Final polish if time remains
    if time.time() < deadline - 5:
        time_remaining = deadline - time.time()
        best_centers, best_r = _optimize_from_start(best_centers, time_budget=time_remaining)
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5 + 1e-10, 0.5 - 1e-10)
    final_r = _compute_radius(best_centers)
    
    return best_centers
# EVOLVE-BLOCK-END
