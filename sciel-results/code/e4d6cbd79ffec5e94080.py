# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd


def calculate_score(points):
    """Calculate R = (d_min / d_max)^2 for 16 points in 2D."""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def hexagonal_init(n, scale=1.0):
    """Initialize points on a hexagonal grid pattern for good initial separation."""
    points = []
    rows = 4
    cols = 4
    dx = 1.0 * scale
    dy = dx * np.sqrt(3) / 2
    
    for i in range(rows):
        for j in range(cols):
            x = j * dx + (i % 2) * (dx / 2)
            y = i * dy
            points.append([x, y])
    
    return np.array(points[:n])


def random_init(n, scale=1.0, seed=None):
    """Initialize points randomly within a circle."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = rng.uniform(0.2, 1.0, n) * scale
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def simulated_annealing(points, start_time, max_time=580.0, n_restarts=1):
    """
    Refine points using simulated annealing to maximize min/max distance ratio.
    Uses repulsive forces to push close points apart while keeping spread.
    """
    n = len(points)
    points = points.copy().astype(float)
    best_points = points.copy()
    best_score = calculate_score(points)
    
    # Precompute index pairs for efficiency
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    # Annealing parameters
    T_start = 0.1
    T_end = 1e-6
    cooling_rate = 0.995
    T = T_start
    
    step_size_start = 0.3
    step_size_end = 0.001
    
    while time.time() - start_time < max_time and T > T_end:
        # Generate perturbation
        step_size = step_size_start * (T / T_start) + step_size_end
        perturb = np.random.randn(n, 2) * step_size
        new_points = points + perturb
        
        # Compute scores
        old_score = best_score
        new_score = calculate_score(new_points)
        
        # Accept if better or with probability exp((new-old)/T)
        if new_score >= old_score:
            points = new_points
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        else:
            delta = (new_score - old_score) / max(T, 1e-12)
            if np.random.random() < np.exp(delta):
                points = new_points
        
        T *= cooling_rate
    
    return best_points, best_score


def repulsive_relaxation(points, start_time, max_time=580.0):
    """
    Relax points using repulsive forces to maximize minimum distance
    while controlling maximum distance.
    """
    n = len(points)
    points = points.copy().astype(float)
    best_points = points.copy()
    best_score = calculate_score(points)
    
    # Precompute index pairs
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    # Force-based relaxation
    lr = 0.05
    iterations = 0
    
    while time.time() - start_time < max_time:
        # Center points
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Normalize scale
        max_dist = np.max(spd.pdist(points))
        if max_dist > 0:
            points = points / max_dist * 2.0  # keep diameter ~2
        
        # Compute pairwise distances
        dists = spd.pdist(points)
        d_min = np.min(dists)
        
        # Compute repulsive forces (stronger for close pairs)
        forces = np.zeros((n, 2))
        for k in range(len(idx_i)):
            i, j = idx_i[k], idx_j[k]
            diff = points[i] - points[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random push for coincident points
                diff = np.random.randn(2) * 0.01
                dist = 0.01
            # Inverse square repulsion, scaled
            force_mag = 1.0 / (dist * dist + 1e-6)
            force = force_mag * diff / dist
            forces[i] += force
            forces[j] -= force
        
        # Apply forces
        points = points + lr * forces
        
        # Re-center and re-normalize
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(spd.pdist(points))
        if max_dist > 0:
            points = points / max_dist * 2.0
        
        score = calculate_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        iterations += 1
        # Slow down learning rate
        lr *= 0.999
    
    return best_points, best_score


def local_refinement(points, start_time, max_time=580.0):
    """
    Fine-grained local optimization using coordinate descent.
    """
    n = len(points)
    points = points.copy().astype(float)
    best_points = points.copy()
    best_score = calculate_score(points)
    
    # Coordinate descent
    lr = 0.01
    while time.time() - start_time < max_time:
        improved = False
        for i in range(n):
            for dim in range(2):
                # Try small perturbations in each direction
                for delta in [lr, -lr, lr*0.5, -lr*0.5, lr*2, -lr*2]:
                    new_points = points.copy()
                    new_points[i, dim] += delta
                    score = calculate_score(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points.copy()
                        points = new_points
                        improved = True
        if not improved:
            lr *= 0.8
            if lr < 1e-8:
                break
    
    return best_points, best_score


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
    Returns np.ndarray of shape (16, 2).
    """
    start_time = time.time()
    max_time = 580.0  # Safety margin before 630s limit
    
    n = 16
    best_points = None
    best_score = 0.0
    
    # Phase 1: Multiple initializations with simulated annealing
    initializations = [
        ("hex", hexagonal_init(n, scale=1.0)),
        ("hex_rot", hexagonal_init(n, scale=1.0) @ np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])),
        ("rand_1", random_init(n, scale=1.0, seed=42)),
        ("rand_2", random_init(n, scale=1.0, seed=123)),
        ("rand_3", random_init(n, scale=1.0, seed=456)),
    ]
    
    # Allocate time per initialization
    time_per_init = max_time / (len(initializations) * 3)  # 3 phases per init
    
    for name, init_points in initializations:
        if time.time() - start_time > max_time - 30:
            break
        
        # Phase 1: Simulated annealing
        t_remaining = max_time - (time.time() - start_time) - 30
        sa_points, sa_score = simulated_annealing(
            init_points, start_time, 
            max_time=min(time_per_init, t_remaining)
        )
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
        
        if time.time() - start_time > max_time - 30:
            break
        
        # Phase 2: Repulsive relaxation
        t_remaining = max_time - (time.time() - start_time) - 30
        rel_points, rel_score = repulsive_relaxation(
            sa_points, start_time,
            max_time=min(time_per_init, t_remaining)
        )
        
        if rel_score > best_score:
            best_score = rel_score
            best_points = rel_points.copy()
        
        if time.time() - start_time > max_time - 30:
            break
        
        # Phase 3: Local refinement
        t_remaining = max_time - (time.time() - start_time) - 30
        ref_points, ref_score = local_refinement(
            rel_points, start_time,
            max_time=min(time_per_init, t_remaining)
        )
        
        if ref_score > best_score:
            best_score = ref_score
            best_points = ref_points.copy()
    
    # Final safety check
    if best_points is None:
        best_points = hexagonal_init(n)
    
    # Ensure valid output
    assert best_points.shape == (16, 2), f"Invalid shape: {best_points.shape}"
    assert np.all(np.isfinite(best_points)), "Non-finite values in points"
    
    return best_points


# EVOLVE-BLOCK-END
