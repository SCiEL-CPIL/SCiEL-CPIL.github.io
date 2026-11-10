"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 300

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.linalg.norm(centers, axis=1)
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    # Use a chunked approach to save memory
    min_pair = np.inf
    for i in range(N):
        d = np.linalg.norm(centers[i+1:] - centers[i], axis=1)
        min_pair = min(min_pair, np.min(d))
    r_pair = min_pair / 2.0
    return min(r_boundary, r_pair)


def _score(centers):
    """Higher is better; returns the radius."""
    return _compute_radius(centers)


def _initial_hex_lattice():
    """Generate a hexagonal lattice point set, trimmed to the unit disk, sorted by radius."""
    # Start with a reasonable spacing
    # For N=300 in unit disk, area per circle ~ pi/300 ~ 0.0105, so spacing ~ sqrt(0.0105/0.866) ~ 0.11
    # Let's try spacing 0.12
    spacing = 0.12
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Sort by radius to prefer centers closer to origin (usually better for packing?)
            # Actually, for packing, we just need N points.
            # Let's return the first N points.
            return points[:N]
        spacing *= 0.98


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Initial solution
    centers = _initial_hex_lattice()
    best_score = _score(centers)
    best_centers = centers.copy()

    # Define the objective function for differential_evolution
    # We want to maximize the radius, so we minimize the negative radius.
    # However, the radius calculation is expensive and non-smooth.
    # A better approach: maximize the minimum distance.
    # Let's use a smooth surrogate or just the raw score.
    # The raw score is min(min_pair/2, min_dist_to_boundary).
    
    def objective(x):
        c = x.reshape(N, 2)
        # Simple penalty for being outside
        if np.any(np.linalg.norm(c, axis=1) > 1.0):
            return 1.0 - np.min(1.0 - np.linalg.norm(c, axis=1))
        return -_score(c)

    # Differential Evolution
    # Bounds: x in [-1, 1], y in [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Run DE with limited iterations to stay within time
    # DE is slow for high dimensions (600D).
    # Maybe use a local refinement approach instead?
    # The prompt mentions "multi-start repulsion+confinement relaxation" improved the score.
    # Let's implement a simple local refinement (gradient-free or gradient-based) on the hex lattice.
    
    # Let's try a simple iterative refinement:
    # 1. Compute current radius r.
    # 2. For each circle, try to move it slightly to increase the minimum distance.
    # This is essentially a local search.
    
    # Given the time limit and dimensionality, DE might be too slow.
    # Let's stick to a robust local search method.
    
    # Method: Repulsion + Confinement
    # Force on each circle i:
    # F_i = sum_{j != i} (c_i - c_j) / |c_i - c_j|^2  (repulsion)
    # F_i += -c_i / |c_i| * (1/|c_i| - 1)  (confinement, pushes inward if near boundary)
    # Actually, confinement should push inward if outside, or just keep inside.
    # Let's use a potential energy approach.
    
    # Let's try a simpler local search:
    # Randomly perturb points and accept if score improves.
    
    current_centers = best_centers.copy()
    current_score = best_score
    step_size = 0.01
    
    while time.time() < deadline:
        # Perturb a random point
        idx = np.random.randint(N)
        original = current_centers[idx].copy()
        current_centers[idx] += np.random.normal(0, step_size, 2)
        
        # Ensure it stays within bounds (roughly)
        norm = np.linalg.norm(current_centers[idx])
        if norm > 0.99:
            current_centers[idx] *= 0.99 / norm
            
        new_score = _score(current_centers)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = current_centers.copy()
        else:
            current_centers[idx] = original
            
        # Adaptive step size
        if new_score > current_score:
            step_size *= 1.05
        else:
            step_size *= 0.95
            step_size = max(step_size, 0.001)

    return best_centers

# EVOLVE-BLOCK-END
