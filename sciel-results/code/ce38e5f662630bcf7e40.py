"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 14
TIME_LIMIT = 150  # seconds
SAFETY_MARGIN = 5  # seconds

# Precompute all triples
def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

TRIPLES = _get_triple_indices(N)


def _compute_score(points):
    """Compute min triangle area / convex hull area."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area < 1e-12:
        return 0.0
    
    # Vectorized triangle area computation
    i, j, k = TRIPLES.T
    a = points[i]
    b = points[j]
    c = points[k]
    
    # Area = 0.5 * |cross(b-a, c-a)|
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area


def _initial_points():
    """Create initial points on a regular polygon-like configuration."""
    # Start with points on a circle (regular polygon)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points


def _perturb(points, rng, scale=0.1):
    """Perturb points slightly."""
    new_points = points.copy()
    new_points += rng.gauss(0, scale, size=new_points.shape)
    return new_points


def construct():
    """
    Construct 14 points in a planar convex region to maximize the minimum
    triangle area divided by the convex hull area.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # Initial configuration: regular polygon
    points = _initial_points()
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Simulated annealing / local search
    rng = np.random.RandomState(42)
    
    # Multi-start with different initial configurations
    initial_configs = []
    
    # 1. Regular polygon
    initial_configs.append(_initial_points())
    
    # 2. Regular polygon with slight random perturbation
    for seed in range(5):
        rng_init = np.random.RandomState(seed)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng_init.uniform(-0.1, 0.1, N)
        radii = 1.0 + rng_init.uniform(-0.05, 0.05, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initial_configs.append(pts)
    
    # 3. Random points in unit circle
    for seed in range(5):
        rng_init = np.random.RandomState(100 + seed)
        angles = rng_init.uniform(0, 2 * np.pi, N)
        radii = rng_init.uniform(0.3, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initial_configs.append(pts)
    
    # Evaluate all initial configurations
    for config in initial_configs:
        score = _compute_score(config)
        if score > best_score:
            best_score = score
            best_points = config.copy()
    
    # Local search from best configuration
    current_points = best_points.copy()
    current_score = best_score
    
    # Adaptive step size
    step_size = 0.1
    min_step_size = 1e-6
    
    while time.time() < deadline:
        # Generate a perturbation
        new_points = _perturb(current_points, rng, step_size)
        
        # Ensure points remain in a reasonable region
        # Normalize to unit area if needed
        new_score = _compute_score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
            
            # Increase step size slightly on improvement
            step_size = min(step_size * 1.1, 0.5)
        else:
            # Decrease step size
            step_size = max(step_size * 0.95, min_step_size)
        
        # Occasionally reset to best and try a larger perturbation
        if rng.random() < 0.01:
            current_points = best_points.copy()
            current_score = best_score
            step_size = 0.3
    
    return best_points
