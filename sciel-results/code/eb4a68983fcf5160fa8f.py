# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time


def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _initial_configurations(n=16, d=2, n_configs=20):
    """Generate diverse initial configurations."""
    configs = []
    
    # 4x4 grid
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    grid = (grid - grid.mean(axis=0)) / grid.std()
    configs.append(grid.copy())
    
    # Hexagonal lattice pattern
    hex_pts = []
    for row in range(4):
        for col in range(4):
            x = col + (0.5 if row % 2 else 0)
            y = row * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts = (hex_pts - hex_pts.mean(axis=0)) / hex_pts.std()
    configs.append(hex_pts.copy())
    
    # Concentric rings
    ring_configs = [
        [1, 3, 4, 8],  # center, ring1, ring2, ring3
        [2, 3, 5, 6],
        [1, 4, 5, 6],
        [0, 4, 6, 6],
    ]
    for rings in ring_configs:
        pts = []
        for r, count in enumerate(rings):
            if count == 0:
                continue
            if r == 0:
                pts.append([0, 0])
            else:
                angles = np.linspace(0, 2*np.pi, count, endpoint=False)
                for a in angles:
                    pts.append([r * np.cos(a), r * np.sin(a)])
        if len(pts) == n:
            pts = np.array(pts)
            pts = (pts - pts.mean(axis=0)) / pts.std()
            configs.append(pts.copy())
    
    # Random configurations
    rng = np.random.default_rng(42)
    for _ in range(n_configs - len(configs)):
        pts = rng.standard_normal((n, d))
        pts = (pts - pts.mean(axis=0)) / pts.std()
        configs.append(pts)
    
    return configs


def _local_optimize(points, n_iter=5000, seed=0):
    """Local optimization by moving individual points."""
    n = len(points)
    rng = np.random.default_rng(seed)
    points = points.copy()
    current_score = _compute_score(points)
    
    # Find the pair with minimum distance and the pair with maximum distance
    def _get_extreme_pairs(pts):
        d = pdist(pts)
        d_min_idx = np.argmin(d)
        d_max_idx = np.argmax(d)
        # Convert from pdist index to pair indices
        i, j = np.unravel_index(d_min_idx, (n, n-1))
        min_pair = (i, j+1) if j >= i else (j, i+1)
        # Fix: pdist uses upper triangle, so index mapping is different
        # Use a simpler approach
        return d_min_idx, d_max_idx
    
    for it in range(n_iter):
        # Pick a random point to move
        idx = rng.integers(n)
        orig = points[idx].copy()
        
        # Try to move point to increase the minimum distance
        # Strategy: move away from nearest neighbor, or toward reducing max distance
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        
        if d_max == 0:
            break
        
        # Find nearest neighbor of this point
        dists_to_point = np.sqrt(np.sum((points - points[idx])**2, axis=1))
        dists_to_point[idx] = np.inf
        nearest = np.argmin(dists_to_point)
        
        # Direction away from nearest neighbor
        direction = points[idx] - points[nearest]
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        
        # Also consider farthest point
        farthest = np.argmax(dists_to_point)
        
        # Try several step sizes
        best_move = None
        best_score = current_score
        
        for step_mult in [0.01, 0.05, 0.1, 0.2, 0.5]:
            for sign in [1, -1]:
                new_point = points[idx] + sign * step_mult * direction
                points[idx] = new_point
                new_score = _compute_score(points)
                if new_score > best_score:
                    best_score = new_score
                    best_move = new_point.copy()
        
        if best_move is not None:
            points[idx] = best_move
            current_score = best_score
        else:
            # Try random perturbation
            perturbation = rng.standard_normal(2) * 0.05
            points[idx] = orig + perturbation
            new_score = _compute_score(points)
            if new_score > current_score:
                current_score = new_score
            else:
                points[idx] = orig
    
    return points, current_score


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    # Generate initial configurations
    configs = _initial_configurations(n, d, n_configs=15)
    
    best_points = None
    best_score = -1.0
    
    # First pass: local optimize each config
    for i, config in enumerate(configs):
        if time.time() > deadline - 60:
            break
        points, score = _local_optimize(config, n_iter=3000, seed=i)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Second pass: simulated annealing from best point
    if best_points is not None and time.time() < deadline - 120:
        points = best_points.copy()
        current_score = best_score
        rng = np.random.default_rng(123)
        
        T = 0.1  # Initial temperature
        cooling = 0.9995
        min_T = 1e-6
        
        while T > min_T and time.time() < deadline - 30:
            # Pick random point
            idx = rng.integers(n)
            orig = points[idx].copy()
            
            # Random perturbation
            perturbation = rng.standard_normal(2) * T * 0.5
            points[idx] = orig + perturbation
            new_score = _compute_score(points)
            
            delta = new_score - current_score
            if delta > 0 or rng.random() < np.exp(delta / (T + 1e-10)):
                current_score = new_score
            else:
                points[idx] = orig
            
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
            
            T *= cooling
        
        # Final local optimization
        if time.time() < deadline - 10:
            points, score = _local_optimize(best_points, n_iter=5000, seed=999)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    # Ensure valid output
    if best_points is None:
        best_points = configs[0]
    
    # Normalize to ensure d_max is reasonable (scale doesn't matter for ratio)
    return best_points


# EVOLVE-BLOCK-END
