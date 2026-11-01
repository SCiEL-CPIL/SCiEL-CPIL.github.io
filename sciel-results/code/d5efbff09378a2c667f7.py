"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import time


def _score(points):
    """Compute the score: min of (min pairwise dist / 2) and (min distance to boundary)."""
    # Clip to valid range first for safety in intermediate computations
    # But for scoring, we need the actual values
    # Boundary distance: for each point, min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
    # Since points should be in [-0.5, 0.5], this is min(0.5 - abs(x), 0.5 - abs(y))
    boundary_dists = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
    min_boundary = np.min(boundary_dists)
    
    # Pairwise distances
    if N < 2:
        return min_boundary
    d = pdist(points)
    min_pair = np.min(d) / 2.0
    
    return min(min_boundary, min_pair)


def _gradient(points, epsilon=1e-7):
    """Compute numerical gradient of the score function."""
    grad = np.zeros_like(points)
    base_score = _score(points)
    
    for i in range(N):
        for j in range(2):
            orig = points[i, j]
            points[i, j] = orig + epsilon
            s_plus = _score(points)
            points[i, j] = orig - epsilon
            s_minus = _score(points)
            points[i, j] = orig
            
            # Central difference
            grad[i, j] = (s_plus - s_minus) / (2 * epsilon)
    
    return grad


def _project(points):
    """Project points back into the square [-0.5, 0.5]^2."""
    return np.clip(points, -0.5, 0.5)


def _simulated_annealing(points, time_limit=100.0):
    """Simulated annealing to maximize the minimum distance."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(points)
    
    current_points = points.copy()
    current_score = best_score
    
    T = 0.05  # Initial temperature
    cooling = 0.9995
    max_iter = 500000
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Generate a random perturbation
        # Scale perturbation with temperature
        step_size = T * 0.1
        idx = np.random.randint(N)
        delta = np.random.randn(2) * step_size
        new_points = current_points.copy()
        new_points[idx] += delta
        new_points = _project(new_points)
        
        new_score = _score(new_points)
        
        # Accept if better or with probability exp((new - current) / T)
        if new_score >= current_score or np.random.rand() < np.exp((new_score - current_score) / max(T, 1e-10)):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        T *= cooling
        if T < 1e-8:
            break
    
    return best_points, best_score


def _local_optimization(points, time_limit=30.0):
    """Local optimization using gradient ascent with adaptive step size."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(points)
    
    current_points = points.copy()
    current_score = best_score
    
    step_size = 0.01
    patience = 0
    max_patience = 50
    
    for iteration in range(10000):
        if time.time() - start_time > time_limit:
            break
        if patience > max_patience:
            break
        
        # Compute gradient
        grad = _gradient(current_points)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-10:
            break
        
        # Gradient ascent step
        direction = grad / grad_norm
        new_points = _project(current_points + step_size * direction)
        new_score = _score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            patience = 0
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            step_size *= 0.8
            patience += 1
    
    return best_points, best_score


def _initial_grid():
    """Create an initial grid layout."""
    m = int(np.ceil(np.sqrt(N)))
    # Use a hexagonal-like offset for better initial packing
    points = []
    for row in range(m):
        for col in range(m):
            x = (col + 0.5) / m - 0.5
            y = (row + 0.5) / m - 0.5
            # Add small offset for hexagonal packing
            if row % 2 == 1:
                x += 0.5 / m
            points.append([x, y])
            if len(points) >= N:
                break
        if len(points) >= N:
            break
    
    return np.array(points[:N])


def _initial_hexagonal():
    """Create an initial hexagonal packing layout."""
    points = []
    # Hexagonal packing: rows offset
    dx = 1.0 / 10  # Approximate spacing
    dy = dx * np.sqrt(3) / 2
    
    for row in range(12):
        for col in range(12):
            x = (col + 0.5) * dx - 0.5
            y = (row + 0.5) * dy - 0.5
            if row % 2 == 1:
                x += dx / 2
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
            if len(points) >= N:
                break
        if len(points) >= N:
            break
    
    if len(points) < N:
        # Fall back to grid
        return _initial_grid()
    
    return np.array(points[:N])


def _force_directed_layout(points, time_limit=20.0):
    """Force-directed layout with repulsive forces."""
    start_time = time.time()
    current_points = points.copy()
    
    for iteration in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        forces = np.zeros_like(current_points)
        
        # Repulsive forces between all pairs
        for i in range(N):
            for j in range(i + 1, N):
                diff = current_points[i] - current_points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.random.randn(2) * 1e-5
                    dist = np.linalg.norm(diff)
                # Force proportional to 1/dist^2
                force_mag = 1.0 / (dist * dist)
                force_dir = diff / dist
                forces[i] += force_mag * force_dir
                forces[j] -= force_mag * force_dir
        
        # Attractive force toward center (to keep points in the square)
        for i in range(N):
            forces[i] -= 0.1 * current_points[i]
        
        # Apply forces with damping
        current_points += 0.01 * forces
        current_points = _project(current_points)
    
    return current_points


def construct():
    """Construct the circle packing configuration."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate multiple initial configurations
    initializations = []
    
    # Grid initialization
    initializations.append(_initial_grid())
    
    # Hexagonal initialization
    initializations.append(_initial_hexagonal())
    
    # Random initialization
    for _ in range(3):
        random_points = np.random.uniform(-0.45, 0.45, (N, 2))
        initializations.append(random_points)
    
    # Best result across all runs
    best_points = None
    best_score = -np.inf
    
    for init_points in initializations:
        if time.time() > deadline:
            break
        
        # Force-directed layout to get a better starting point
        fd_points = _force_directed_layout(init_points, time_limit=5.0)
        
        # Simulated annealing
        sa_points, sa_score = _simulated_annealing(fd_points, time_limit=20.0)
        
        # Local optimization
        lo_points, lo_score = _local_optimization(sa_points, time_limit=10.0)
        
        if lo_score > best_score:
            best_score = lo_score
            best_points = lo_points.copy()
        
        if time.time() > deadline - 20.0:
            break
    
    # Final polish if time remains
    if best_points is not None and time.time() < deadline - 5.0:
        final_points, final_score = _local_optimization(best_points, time_limit=15.0)
        if final_score > best_score:
            best_points = final_points
            best_score = final_score
    
    if best_points is None:
        # Fallback: return grid
        best_points = _initial_grid()
    
    # Ensure valid output
    best_points = np.clip(best_points, -0.5, 0.5)
    return best_points
# EVOLVE-BLOCK-END
