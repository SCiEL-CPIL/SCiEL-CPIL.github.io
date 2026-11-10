# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time


def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _normalize(points):
    """Center at origin and scale so sum of squared norms = n."""
    points = points - np.mean(points, axis=0)
    norm_sq = np.sum(points ** 2)
    if norm_sq > 0:
        points = points * np.sqrt(len(points) / norm_sq)
    return points


def _repulsion_step(points, strength=0.01):
    """Move points apart based on inverse-square repulsion, weighted by closeness to min distance."""
    n = len(points)
    d = pdist(points)
    # Reconstruct distance matrix
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_mat[i, j] = d[j * (j - 1) // 2 + i]
            dist_mat[j, i] = dist_mat[i, j]
    
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return points
    
    # Force: repulsive, stronger for closer pairs
    forces = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random small perturbation for coincident points
                diff = np.random.randn(2) * 1e-6
                dist = np.linalg.norm(diff)
            # Weight by how close this distance is to d_min
            weight = 1.0 + 10.0 * (1.0 - dist / d_max) ** 2
            force = weight * diff / (dist ** 2 + 1e-10)
            forces[i] += force
            forces[j] -= force
    
    points = points + strength * forces
    return _normalize(points)


def _optimize_from_start(points, max_iter=5000, deadline=None):
    """Iterative optimization starting from given points."""
    best_points = points.copy()
    best_score = _compute_score(points)
    
    strength = 0.05
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try multiple strength values
        for s in [strength * 0.1, strength * 0.5, strength, strength * 2.0]:
            new_points = _repulsion_step(best_points, s)
            score = _compute_score(new_points)
            if score > best_score:
                best_score = score
                best_points = new_points
        
        # Decay strength
        strength *= 0.995
        
        if it % 100 == 0 and strength < 0.0001:
            break
    
    return best_points, best_score


def _circle_init(n, radius=1.0):
    """Points on a circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return _normalize(points)


def _two_circles_init(n, r1=0.5, r2=1.0):
    """Points on two concentric circles."""
    n1 = n // 2
    n2 = n - n1
    angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
    points = np.vstack([
        np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)]),
        np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    ])
    return _normalize(points)


def _grid_init(n):
    """4x4 grid."""
    side = int(np.ceil(np.sqrt(n)))
    x = np.linspace(-1.5, 1.5, side)
    y = np.linspace(-1.5, 1.5, side)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])[:n]
    return _normalize(points)


def _hexagonal_init(n):
    """Hexagonal lattice arrangement."""
    points = []
    # Center
    points.append([0.0, 0.0])
    # Rings of hexagonal lattice
    ring = 1
    while len(points) < n:
        for i in range(6):
            angle = i * np.pi / 3
            for j in range(ring):
                px = (ring - j) * np.cos(angle) + j * np.cos(angle + np.pi / 3)
                py = (ring - j) * np.sin(angle) + j * np.sin(angle + np.pi / 3)
                points.append([px, py])
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        ring += 1
    points = np.array(points[:n])
    return _normalize(points)


def _random_init(n, seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 2)
    return _normalize(points)


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580 second deadline, 20s safety margin
    
    n = 16
    best_points = None
    best_score = -1.0
    
    # Generate multiple initializations
    inits = []
    
    # Structured inits
    inits.append(_circle_init(n))
    inits.append(_two_circles_init(n, 0.4, 1.0))
    inits.append(_two_circles_init(n, 0.5, 1.0))
    inits.append(_two_circles_init(n, 0.6, 1.0))
    inits.append(_two_circles_init(n, 0.3, 0.8))
    inits.append(_grid_init(n))
    inits.append(_hexagonal_init(n))
    
    # Random inits with various seeds
    for seed in range(20):
        inits.append(_random_init(n, seed))
    
    # Evaluate and optimize each initialization
    for idx, init_points in enumerate(inits):
        if time.time() > deadline:
            break
        
        score = _compute_score(init_points)
        if score > best_score:
            best_score = score
            best_points = init_points.copy()
        
        # Optimize from this starting point
        opt_points, opt_score = _optimize_from_start(init_points, max_iter=3000, deadline=deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points
    
    # Final polish: more iterations from the best
    if best_points is not None and time.time() < deadline:
        opt_points, opt_score = _optimize_from_start(best_points, max_iter=5000, deadline=deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points
    
    # Ensure valid output
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _circle_init(n)
    
    return best_points.reshape(16, 2)


# EVOLVE-BLOCK-END
