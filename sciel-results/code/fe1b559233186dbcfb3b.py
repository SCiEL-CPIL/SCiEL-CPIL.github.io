# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools


def _compute_score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points using full vectorization."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Generate all triples i < j < k
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
                
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area == 0:
        return 0.0
    
    return min_area / A_hull


def _compute_score_fast(points: np.ndarray) -> float:
    """Fast score computation, returns 0 if invalid."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    if A_hull <= 0:
        return 0.0
    
    # Precompute all triples once
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
                
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area == 0:
        return 0.0
    
    return min_area / A_hull


def _find_worst_triangle(points: np.ndarray):
    """Find the triangle with minimum area. Returns (area, i, j, k)."""
    n = points.shape[0]
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
                
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    return areas[min_idx], i_idx[min_idx], j_idx[min_idx], k_idx[min_idx]


def _regular_polygon(n=13, radius=0.5, center=(0.5, 0.5)):
    """Generate points on a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])


def _hexagonal_initialization():
    """Generate a hexagonal-like arrangement."""
    pts = []
    # Center
    pts.append([0.5, 0.5])
    # Ring 1: 6 points
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([0.5 + 0.2 * np.cos(angle), 0.5 + 0.2 * np.sin(angle)])
    # Ring 2: 6 points
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        pts.append([0.5 + 0.38 * np.cos(angle), 0.5 + 0.38 * np.sin(angle)])
    return np.array(pts[:13])


def _grid_initialization():
    """Generate a 4x4 grid minus one corner."""
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([0.15 + 0.23 * i, 0.15 + 0.23 * j])
    return np.array(pts[:13])


def _concentric_initialization():
    """Concentric rings."""
    pts = [[0.5, 0.5]]
    for r, n_pts in [(0.25, 5), (0.42, 7)]:
        for i in range(n_pts):
            angle = 2 * np.pi * i / n_pts
            pts.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
    return np.array(pts[:13])


def _random_initialization(rng):
    """Random points in [0.1, 0.9]^2."""
    return rng.random((13, 2)) * 0.8 + 0.1


def _local_search(points: np.ndarray, deadline: float, max_steps=500):
    """Local search: perturb worst triangle vertices and nearby points."""
    n = 13
    best = points.copy()
    best_score = _compute_score_fast(best)
    if best_score == 0:
        return best
    
    step_size = 0.05
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    
    for step in range(max_steps):
        if time.time() > deadline - 2:
            break
        
        # Find worst triangle
        _, i, j, k = _find_worst_triangle(best)
        
        # Choose a point to move: one of the worst triangle vertices or a random point
        if rng.random() < 0.7:
            # Move one of the worst triangle vertices
            candidates = [i, j, k]
            pt_idx = candidates[rng.integers(0, 3)]
        else:
            pt_idx = rng.integers(0, n)
        
        # Try several perturbations
        for _ in range(10):
            candidate = best.copy()
            noise = rng.normal(0, step_size, size=2)
            candidate[pt_idx] += noise
            # Keep in bounds
            candidate[pt_idx] = np.clip(candidate[pt_idx], 0.01, 0.99)
            
            score = _compute_score_fast(candidate)
            if score > best_score:
                best = candidate
                best_score = score
                break
        else:
            # No improvement, reduce step size
            step_size *= 0.95
            if step_size < 0.001:
                break
    
    return best


def _simulated_annealing(points: np.ndarray, deadline: float, max_steps=3000):
    """Simulated annealing."""
    n = 13
    current = points.copy()
    current_score = _compute_score_fast(current)
    if current_score == 0:
        return current
    
    best = current.copy()
    best_score = current_score
    
    T = 0.02
    cooling = 0.999
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    
    for step in range(max_steps):
        if time.time() > deadline - 2:
            break
        
        # Random perturbation
        candidate = current.copy()
        n_perturb = rng.integers(1, 4)
        for _ in range(n_perturb):
            idx = rng.integers(0, n)
            noise = rng.normal(0, 0.03, size=2)
            candidate[idx] += noise
            candidate[idx] = np.clip(candidate[idx], 0.01, 0.99)
        
        cand_score = _compute_score_fast(candidate)
        
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
            if cand_score > best_score:
                best = candidate
                best_score = cand_score
        else:
            # Accept with probability exp((new - old) / T)
            delta = cand_score - current_score
            if delta < 0:
                prob = np.exp(delta / T) if T > 1e-10 else 0
                if rng.random() < prob:
                    current = candidate
                    current_score = cand_score
        
        T *= cooling
    
    return best


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Uses regular polygon initialization with radial perturbations, local search targeting worst triangles,
    and simulated annealing.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Initial configurations
    initial_configs = [
        _regular_polygon(13, 0.45, (0.5, 0.5)),
        _regular_polygon(13, 0.40, (0.5, 0.5)),
        _hexagonal_initialization(),
        _grid_initialization(),
        _concentric_initialization(),
    ]
    
    # Add some random initializations
    for seed in range(5):
        rng = np.random.default_rng(seed)
        initial_configs.append(_random_initialization(rng))
    
    # Evaluate and optimize each config
    for idx, config in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        score = _compute_score_fast(config)
        if score > best_score:
            best_score = score
            best_points = config.copy()
        
        # Local search
        if time.time() < deadline - 10:
            ls_result = _local_search(config, deadline, max_steps=300)
            ls_score = _compute_score_fast(ls_result)
            if ls_score > best_score:
                best_score = ls_score
                best_points = ls_result.copy()
        
        # Simulated annealing from best so far
        if time.time() < deadline - 15 and best_points is not None:
            sa_result = _simulated_annealing(best_points, deadline, max_steps=2000)
            sa_score = _compute_score_fast(sa_result)
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_result.copy()
    
    # Final local search refinement
    if best_points is not None and time.time() < deadline - 5:
        final_result = _local_search(best_points, deadline, max_steps=500)
        final_score = _compute_score_fast(final_result)
        if final_score > best_score:
            best_score = final_score
            best_points = final_result.copy()
    
    # Final safety
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(13, 0.45, (0.5, 0.5))
    
    return best_points


# EVOLVE-BLOCK-END
