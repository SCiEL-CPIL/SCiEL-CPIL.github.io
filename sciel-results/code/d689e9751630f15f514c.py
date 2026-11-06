"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree


def _compute_score(points):
    """Compute the minimum radius for a set of points."""
    n = len(points)
    if n == 0:
        return 0.0
    dists = np.sqrt(np.sum(points**2, axis=1))
    boundary_r = np.min(1.0 - dists)
    if boundary_r < 0:
        return 0.0
    # Use KDTree for efficient nearest neighbor search
    tree = cKDTree(points)
    # Get all pairs within a reasonable distance
    # We need the minimum pair distance, so we can use query_pairs
    # But for 300 points, brute force is fine and faster than KDTree overhead
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    pair_r = min_pair_dist / 2.0
    return min(boundary_r, pair_r)


def _hexagonal_init(n, seed=42):
    """Generate n points in the unit disk using hexagonal lattice."""
    # Start with a hexagonal lattice and select n points closest to center
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    for _ in range(100):
        k = int(1.0 / spacing) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.sqrt(x**2 + y**2)
                if r <= 1.0 - spacing / 2.0:
                    points.append((x, y, r))
        points = np.array(points)
        if len(points) >= n:
            # Sort by radius and take first n
            idx = np.argsort(points[:, 2])[:n]
            return points[idx, :2]
        spacing *= 0.98
    # Fallback: random points
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(rng.uniform(0, 1, n))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _force_relaxation(points, max_iter=5000, deadline=None):
    """Force-based relaxation to improve packing."""
    points = points.copy()
    n = len(points)
    best_score = _compute_score(points)
    best_points = points.copy()
    
    # Adaptive step size
    step = 0.01
    min_step = 1e-8
    max_step = 0.1
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion: push apart if too close
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.sqrt(np.sum(diff**2))
                if dist < 1e-10:
                    # Random perturbation for coincident points
                    forces[i] += rng.uniform(-1, 1, 2) * 0.01
                    forces[j] -= rng.uniform(-1, 1, 2) * 0.01
                    continue
                # Desired minimum distance is determined by the current bottleneck
                # We want all pairs to have at least 2*r distance
                # For now, use a fixed target based on average spacing
                target_dist = 2.0 * best_score
                if dist < target_dist:
                    # Repulsive force proportional to overlap
                    force_mag = (target_dist - dist) / max(dist, 1e-10)
                    forces[i] += (diff / dist) * force_mag
                    forces[j] -= (diff / dist) * force_mag
        
        # Boundary force: push inward if too close to boundary
        for i in range(n):
            r = np.sqrt(np.sum(points[i]**2))
            if r > 1.0 - best_score:
                # Inward force
                if r > 1e-10:
                    forces[i] -= (points[i] / r) * (r - (1.0 - best_score))
        
        # Normalize forces
        force_norms = np.sqrt(np.sum(forces**2, axis=1))
        max_norm = np.max(force_norms)
        if max_norm > 0:
            forces /= max_norm
        
        # Update positions
        new_points = points + step * forces
        
        # Check if new configuration is valid
        new_score = _compute_score(new_points)
        
        if new_score >= best_score:
            points = new_points
            best_score = new_score
            best_points = points.copy()
            # Increase step size if improving
            step = min(step * 1.05, max_step)
        else:
            # Decrease step size
            step = max(step * 0.95, min_step)
            if step <= min_step * 1.1:
                # Try a few more iterations with small steps
                if it > max_iter * 0.9:
                    break
    
    return best_points, best_score


def _simulated_annealing(points, max_iter=10000, deadline=None):
    """Simulated annealing to escape local minima."""
    points = points.copy()
    n = len(points)
    best_score = _compute_score(points)
    best_points = points.copy()
    
    rng = np.random.RandomState(42)
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling = 0.999
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate a perturbation
        new_points = points.copy()
        idx = rng.randint(n)
        perturbation = rng.normal(0, 0.01, 2)
        new_points[idx] += perturbation
        
        # Keep within bounds
        r = np.sqrt(np.sum(new_points[idx]**2))
        if r > 1.0:
            new_points[idx] *= (1.0 / r)
        
        new_score = _compute_score(new_points)
        delta = new_score - best_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            points = new_points
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        
        T = max(T * cooling, T_min)
    
    return best_points, best_score


def construct():
    # Track time
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initialize with hexagonal lattice
    points = _hexagonal_init(N)
    score = _compute_score(points)
    
    # Phase 1: Force-based relaxation
    points, score = _force_relaxation(points, max_iter=3000, deadline=deadline)
    
    if time.time() > deadline:
        return points
    
    # Phase 2: More force-based relaxation with smaller steps
    points, score = _force_relaxation(points, max_iter=5000, deadline=deadline)
    
    if time.time() > deadline:
        return points
    
    # Phase 3: Simulated annealing for fine-tuning
    points, score = _simulated_annealing(points, max_iter=5000, deadline=deadline)
    
    return points
# EVOLVE-BLOCK-END
