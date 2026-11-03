"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial.distance import cdist
import time

N = 200

# EVOLVE-BLOCK-START

def _score(points):
    """Compute the packing radius for a set of points in [-0.5, 0.5]^2."""
    # Boundary constraint
    max_coord = np.max(np.abs(points), axis=1)
    boundary_min = 0.5 - max_coord
    # If any point is outside, return 0 (invalid)
    if np.any(boundary_min < -1e-12):
        return 0.0
    # Pairwise distances
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_dist = np.min(d)
    r = min(min_dist / 2.0, np.min(boundary_min))
    if r < 0:
        return 0.0
    return r


def _random_init():
    """Generate random initial points inside the square."""
    return np.random.uniform(-0.5, 0.5, size=(N, 2))


def _hex_init():
    """Generate a hexagonal lattice initialization clipped to the square."""
    # For N=200, a hexagonal packing with spacing s
    # Area per circle in hex packing: s^2 * sqrt(3)/2
    # Total area ~ 1, so N * s^2 * sqrt(3)/2 ~ 1 => s ~ sqrt(2/(N*sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    # Generate hex lattice points
    points = []
    # Number of rows/cols to cover the square
    n_rows = int(np.ceil(1.0 / (s * np.sqrt(3.0) / 2.0))) + 4
    n_cols = int(np.ceil(1.0 / s)) + 4
    for i in range(-n_rows, n_rows + 1):
        for j in range(-n_cols, n_cols + 1):
            x = j * s + (i % 2) * s / 2.0
            y = i * s * np.sqrt(3.0) / 2.0
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= N:
        # Randomly select N points
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        # Pad with random points if needed
        extra = np.random.uniform(-0.5, 0.5, size=(N - len(points), 2))
        return np.vstack([points, extra])


def _fd_gradient(points, r_target):
    """Compute gradient of the score using finite differences."""
    eps = 1e-6
    grad = np.zeros_like(points)
    base_score = _score(points)
    
    # Only compute gradient if base score is close to target
    # For efficiency, we'll use a simplified approach
    
    for i in range(N):
        for k in range(2):
            old = points[i, k]
            points[i, k] = old + eps
            s_plus = _score(points)
            points[i, k] = old - eps
            s_minus = _score(points)
            points[i, k] = old
            grad[i, k] = (s_plus - s_minus) / (2.0 * eps)
    return grad


def _project_to_square(points):
    """Project points to stay within the square."""
    points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
    points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
    return points


def _optimize(points, time_limit=100.0):
    """Run a simple gradient-based optimization with restarts."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Use a simple repulsion-based approach
    # Iteratively push overlapping circles apart
    
    for iteration in range(50000):
        if time.time() - start_time > time_limit:
            break
        
        # Compute pairwise distances
        d = cdist(points, points)
        np.fill_diagonal(d, np.inf)
        
        # Find minimum distance
        min_idx = np.unravel_index(np.argmin(d), d.shape)
        min_dist = d[min_idx]
        
        # If minimum distance is large enough, we're close to optimal
        # The target radius is about 0.0366, so min_dist should be about 0.0732
        if min_dist > 0.073:
            break
        
        # Push the closest pair apart
        i, j = min_idx
        diff = points[i] - points[j]
        if np.linalg.norm(diff) < 1e-10:
            diff = np.random.randn(2) * 1e-6
        diff = diff / np.linalg.norm(diff)
        
        # Move both points apart
        step = 0.001
        points[i] += diff * step
        points[j] -= diff * step
        
        # Also push against boundaries
        for k in range(N):
            for axis in range(2):
                if points[k, axis] > 0.49:
                    points[k, axis] -= 0.001
                elif points[k, axis] < -0.49:
                    points[k, axis] += 0.001
        
        # Update best
        current_score = _score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
    
    return best_points, best_score


def construct():
    """Construct the optimal packing of 200 circles in a unit square."""
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for safety
    
    # Try multiple initializations and optimization strategies
    best_points = None
    best_score = 0.0
    
    # Strategy 1: Hexagonal lattice initialization
    while time.time() - start_time < time_limit * 0.5:
        points = _hex_init()
        points, score = _optimize(points, time_limit=30.0)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        if time.time() - start_time > time_limit * 0.5:
            break
    
    # Strategy 2: Random initialization with repulsion
    while time.time() - start_time < time_limit * 0.8:
        points = _random_init()
        points, score = _optimize(points, time_limit=20.0)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        if time.time() - start_time > time_limit * 0.8:
            break
    
    # Final refinement on the best solution
    if best_points is not None and time.time() - start_time < time_limit - 5:
        remaining_time = time_limit - (time.time() - start_time) - 5
        if remaining_time > 5:
            refined_points, refined_score = _optimize(best_points, time_limit=remaining_time)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points
    
    if best_points is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
    
    # Final safety check: ensure points are within bounds
    best_points[:, 0] = np.clip(best_points[:, 0], -0.5, 0.5)
    best_points[:, 1] = np.clip(best_points[:, 1], -0.5, 0.5)
    
    return best_points

# EVOLVE-BLOCK-END
