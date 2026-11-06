"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist


def _score(centers):
    """Compute the score: min over all pairs of half-distance and min over walls."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distance
    wall = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall)
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_wall)


def _make_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)


def _relax_spheres(centers, n_iterations=5000, learning_rate=0.01, deadline=None):
    """
    Simple force-directed relaxation to maximize minimum distance.
    Uses gradient ascent on the minimum distance.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    for it in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute all pairwise distances
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        
        # Find the minimum distance pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx[0], min_idx[1]
        min_dist = dists[i, j]
        
        # Also check wall constraints
        wall = np.min(0.5 - np.abs(centers), axis=1)
        min_wall_idx = np.argmin(wall)
        min_wall = wall[min_wall_idx]
        
        current_score = min(min_dist / 2.0, min_wall)
        
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
        
        # Determine which constraint is active
        if min_dist / 2.0 < min_wall:
            # Pair constraint active: push i and j apart
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-15:
                # Random nudge
                diff = np.random.randn(3)
                norm = np.linalg.norm(diff)
            direction = diff / norm
            centers[i] += learning_rate * direction
            centers[j] -= learning_rate * direction
        else:
            # Wall constraint active: push the closest sphere inward
            idx = min_wall_idx
            # Find which wall is closest
            for k in range(3):
                if 0.5 - centers[idx, k] < min_wall + 1e-15 and centers[idx, k] >= 0:
                    # Close to +0.5 wall
                    centers[idx, k] -= learning_rate
                    break
                elif 0.5 + centers[idx, k] < min_wall + 1e-15 and centers[idx, k] <= 0:
                    # Close to -0.5 wall
                    centers[idx, k] += learning_rate
                    break
                else:
                    # Try other walls
                    if centers[idx, k] > 0 and 0.5 - centers[idx, k] < min_wall + 1e-15:
                        centers[idx, k] -= learning_rate
                        break
                    elif centers[idx, k] < 0 and 0.5 + centers[idx, k] < min_wall + 1e-15:
                        centers[idx, k] += learning_rate
                        break
        
        # Keep centers within bounds (with small margin)
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Decay learning rate
        learning_rate *= 0.9999
    
    return best_centers, best_score


def _local_search(centers, n_iterations=3000, deadline=None):
    """
    Local search: randomly perturb a sphere and accept if score improves.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    for it in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random sphere to move
        idx = np.random.randint(N)
        old_pos = centers[idx].copy()
        
        # Random perturbation
        step = 0.005 * (1.0 - it / n_iterations) + 0.001
        centers[idx] += np.random.randn(3) * step
        centers[idx] = np.clip(centers[idx], -0.4999, 0.4999)
        
        new_score = _score(centers)
        if new_score > best_score:
            best_score = new_score
            best_centers = centers.copy()
        else:
            centers[idx] = old_pos
    
    return best_centers, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with grid
    centers = _make_grid()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Phase 1: Force-directed relaxation
    centers, score = _relax_spheres(centers, n_iterations=10000, learning_rate=0.005, deadline=deadline)
    if score > best_score:
        best_score = score
        best_centers = centers.copy()
    
    # Phase 2: Local search
    centers, score = _local_search(centers, n_iterations=20000, deadline=deadline)
    if score > best_score:
        best_score = score
        best_centers = centers.copy()
    
    # Phase 3: More relaxation with smaller steps
    centers, score = _relax_spheres(centers, n_iterations=10000, learning_rate=0.002, deadline=deadline)
    if score > best_score:
        best_score = score
        best_centers = centers.copy()
    
    # Phase 4: Final local search
    centers, score = _local_search(centers, n_iterations=10000, deadline=deadline)
    if score > best_score:
        best_score = score
        best_centers = centers.copy()
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.49999, 0.49999)
    
    return best_centers
# EVOLVE-BLOCK-END
