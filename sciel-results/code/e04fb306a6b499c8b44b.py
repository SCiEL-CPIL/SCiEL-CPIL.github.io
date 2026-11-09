"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize, minimize_scalar

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_M = len(_idx_i)  # 165

# Precompute all C(11,2) = 55 pair index pairs for packing
_pair_i, _pair_j = [], []
for i in range(N):
    for j in range(i + 1, N):
        _pair_i.append(i)
        _pair_j.append(j)
_pair_i = np.array(_pair_i)
_pair_j = np.array(_pair_j)
_M_pair = len(_pair_i)  # 55


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * np.min(cross)


def _min_area_and_idx(points):
    """Compute minimum triangle area and which triple achieves it."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    areas = 0.5 * cross
    idx = np.argmin(areas)
    return areas[idx], idx


def _min_pairwise_distance(points):
    """Compute minimum pairwise distance."""
    p1 = points[_pair_i]
    p2 = points[_pair_j]
    dists = np.sqrt(np.sum((p2 - p1) ** 2, axis=1))
    return np.min(dists)


def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    result = points.copy()
    for i in range(len(result)):
        x, y = result[i]
        if y < 0:
            y = 0
        if y > np.sqrt(3) * x + 1e-12:
            if x > 1e-12:
                y = np.sqrt(3) * x
            else:
                y = 0
        if y > np.sqrt(3) * (1 - x) + 1e-12:
            if 1 - x > 1e-12:
                y = np.sqrt(3) * (1 - x)
            else:
                y = 0
        result[i] = [x, y]
    return result


def _local_search(points, max_iter=2000, tol=1e-10):
    """
    Local search: iteratively move the point(s) in the minimum-area triangle
    to increase the minimum area.
    """
    points = points.copy()
    best_area = _min_area(points)
    
    for iteration in range(max_iter):
        min_a, min_idx = _min_area_and_idx(points)
        
        # Get the three points in the minimum triangle
        i, j, k = _idx_i[min_idx], _idx_j[min_idx], _idx_k[min_idx]
        
        improved = False
        # Try moving each of the three points
        for idx in [i, j, k]:
            best_local = points[idx].copy()
            best_local_area = min_a
            
            # Grid search around current position
            for dx in np.linspace(-0.05, 0.05, 11):
                for dy in np.linspace(-0.05, 0.05, 11):
                    trial = points.copy()
                    trial[idx] = [points[idx][0] + dx, points[idx][1] + dy]
                    trial = _clamp_to_triangle(trial)
                    trial_area = _min_area(trial)
                    if trial_area > best_local_area:
                        best_local_area = trial_area
                        best_local = trial[idx].copy()
            
            if best_local_area > min_a + 1e-12:
                points[idx] = best_local
                improved = True
        
        if not improved:
            break
        
        best_area = _min_area(points)
    
    return points


def _init_symmetric():
    """
    Initialize with a symmetric configuration.
    """
    points = np.zeros((N, 2))
    points[0] = [0.5, H / 3]
    eps = 0.15
    points[1] = [eps, H * eps]
    points[2] = [1 - eps, H * eps]
    points[3] = [0.5, H * (1 - 2 * eps)]
    points[4] = [0.5 * (1 - eps), H * 0.5 * (1 - eps)]
    points[5] = [0.5 + 0.5 * eps, H * 0.5 * eps]
    points[6] = [0.5 * eps, H * (1 - 0.5 * eps)]
    r = 0.25
    for i in range(3):
        angle = 2 * np.pi * i / 3
        x = 0.5 + r * np.cos(angle)
        y = H / 3 + r * H / (2 * 0.5) * np.sin(angle) * 0.5
        points[7 + i] = [x, y]
    return _clamp_to_triangle(points)


def _init_lattice():
    """Initialize with a triangular lattice pattern. Ensure N points."""
    points = []
    # Generate a denser grid to ensure we have at least N points
    n_rows = 5
    for row in range(n_rows):
        n_cols = n_rows - row
        y = H * (1 - row / (n_rows - 1)) * 0.95
        x_start = row * 0.5 / (n_rows - 1)
        for col in range(n_cols):
            x = x_start + col * 0.5 / (n_rows - 1)
            points.append([x, y])
    
    points = np.array(points)
    if len(points) < N:
        # Pad with random points if necessary (should not happen with n_rows=5)
        while len(points) < N:
            points = np.vstack([points, np.random.rand(2)])
    else:
        points = points[:N]
        
    return _clamp_to_triangle(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Bounds for DE: x in [0,1], y in [0,H]
    bounds = [(0, 1), (0, H)] * N
    
    # Stage 1: Maximize minimum pairwise distance (packing proxy)
    # This objective is smoother and helps find well-spread configurations
    def _obj_packing(params):
        pts = params.reshape(N, 2)
        pts = _clamp_to_triangle(pts)
        return -_min_pairwise_distance(pts)
    
    # Run DE for packing
    # We use multiple seeds/restarts if time permits, but DE is stochastic enough
    # Let's run one robust DE instance first
    try:
        # maxiter and popsize tuned for speed
        result = differential_evolution(
            _obj_packing,
            bounds,
            maxiter=100,
            popsize=30,
            tol=1e-8,
            seed=42,
            polish=False, # We will polish manually with local search on area
            init='latinhypercube'
        )
        
        de_points = _clamp_to_triangle(result.x.reshape(N, 2))
        
        # Stage 2: Polish with local search on min triangle area
        polished = _local_search(de_points, max_iter=3000)
        area = _min_area(polished)
        
        if area > best_score:
            best_score = area
            best_points = polished.copy()
            
    except Exception:
        pass
    
    # If time permits, try a few random restarts with packing DE
    if time.time() < deadline - 20 and best_points is not None:
        for seed in [123, 456]:
            if time.time() > deadline - 10:
                break
            try:
                result = differential_evolution(
                    _obj_packing,
                    bounds,
                    maxiter=50,
                    popsize=20,
                    tol=1e-8,
                    seed=seed,
                    polish=False,
                    init='latinhypercube'
                )
                de_points = _clamp_to_triangle(result.x.reshape(N, 2))
                polished = _local_search(de_points, max_iter=2000)
                area = _min_area(polished)
                
                if area > best_score:
                    best_score = area
                    best_points = polished.copy()
            except Exception:
                pass

    # Fallback to symmetric init if DE failed or didn't improve
    if best_points is None or best_score < _min_area(_init_symmetric()):
        sym = _init_symmetric()
        polished_sym = _local_search(sym, max_iter=3000)
        area = _min_area(polished_sym)
        if area > best_score:
            best_score = area
            best_points = polished_sym.copy()

    if best_points is None:
        best_points = _init_symmetric()
        
    return _clamp_to_triangle(best_points)
