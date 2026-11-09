# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools


def _compute_score_fast(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points using vectorized triangle areas."""
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
    
    # Vectorized computation of all triangle areas
    # For each pair (i, j), compute area with all k > j
    min_area = float('inf')
    
    # Precompute for speed: use the formula 0.5 * |cross(p_j - p_i, p_k - p_i)|
    # We can vectorize over k for each (i, j) pair
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            pj = points[j]
            # Vector over k from j+1 to n-1
            pk = points[j+1:]
            v1 = pj - pi
            v2 = pk - pi
            # Cross product in 2D: v1_x * v2_y - v1_y * v2_x
            cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
            areas = np.abs(cross) * 0.5
            min_area = min(min_area, float(np.min(areas)))
    
    if min_area <= 0:
        return 0.0
    
    return min_area / A_hull


def _compute_score_batch(points_list: list) -> list:
    """Compute scores for a batch of point configurations."""
    return [_compute_score_fast(p) for p in points_list]


def _simulated_annealing(points: np.ndarray, deadline: float, 
                          T_init: float = 0.1, T_min: float = 1e-6,
                          cooling_rate: float = 0.995, max_iters: int = 50000) -> np.ndarray:
    """Simulated annealing with adaptive step size for Heilbronn problem."""
    best_points = points.copy()
    current_points = points.copy()
    current_score = _compute_score_fast(current_points)
    best_score = current_score
    
    n = len(current_points)
    step_size = 0.05
    min_step = 1e-8
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Adaptive temperature
        T = T_init * (cooling_rate ** it)
        if T < T_min:
            T = T_min
        
        # Choose a random point and a random direction
        idx = np.random.randint(n)
        # Random perturbation in both x and y
        dx = np.random.randn() * step_size
        dy = np.random.randn() * step_size
        
        candidate = current_points.copy()
        candidate[idx, 0] += dx
        candidate[idx, 1] += dy
        
        # Keep within bounds [0, 1]
        if candidate[idx, 0] < 0 or candidate[idx, 0] > 1 or \
           candidate[idx, 1] < 0 or candidate[idx, 1] > 1:
            continue
        
        candidate_score = _compute_score_fast(candidate)
        
        # Acceptance criterion
        if candidate_score > current_score:
            current_points = candidate
            current_score = candidate_score
            if current_score > best_score:
                best_score = current_score
                best_points = candidate.copy()
        else:
            # Metropolis criterion
            delta = current_score - candidate_score
            if delta < 0:
                accept = True
            else:
                accept = np.random.random() < np.exp(-delta / max(T, 1e-10))
            if accept:
                current_points = candidate
                current_score = candidate_score
        
        # Adapt step size: reduce if many rejections
        # Simple adaptation: reduce step size slightly each iteration
        if it % 100 == 0:
            step_size *= 0.98
            if step_size < min_step:
                step_size = min_step
    
    return best_points


def _local_search(points: np.ndarray, deadline: float, max_iters: int = 10000) -> np.ndarray:
    """Local search with adaptive step size, focusing on improving the worst triangle."""
    best_points = points.copy()
    best_score = _compute_score_fast(best_points)
    
    n = len(best_points)
    step_x = 0.02
    step_y = 0.02
    min_step = 1e-8
    decay = 0.9
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Find the worst triangle and try to move its vertices
        # Compute all triangle areas to find the minimum
        min_area = float('inf')
        worst_triple = None
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    p = best_points[[i, j, k]]
                    v1 = p[1] - p[0]
                    v2 = p[2] - p[0]
                    area = abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2.0
                    if area < min_area:
                        min_area = area
                        worst_triple = (i, j, k)
        
        if worst_triple is None:
            break
        
        # Try to move each vertex of the worst triangle
        for idx in worst_triple:
            for dx in [-step_x, step_x]:
                candidate = best_points.copy()
                candidate[idx, 0] += dx
                if 0 <= candidate[idx, 0] <= 1:
                    score = _compute_score_fast(candidate)
                    if score > best_score:
                        best_points = candidate
                        best_score = score
                        improved = True
            for dy in [-step_y, step_y]:
                candidate = best_points.copy()
                candidate[idx, 1] += dy
                if 0 <= candidate[idx, 1] <= 1:
                    score = _compute_score_fast(candidate)
                    if score > best_score:
                        best_points = candidate
                        best_score = score
                        improved = True
        
        if not improved:
            step_x *= decay
            step_y *= decay
            if step_x < min_step and step_y < min_step:
                break
    
    return best_points


def _hexagonal_initialization(n: int) -> np.ndarray:
    """Create a hexagonal lattice initialization."""
    points = []
    spacing = 0.25
    row_height = spacing * np.sqrt(3) / 2
    
    row = 0
    while len(points) < n:
        y = 0.1 + row * row_height
        if row % 2 == 0:
            cols = 4
            x_start = 0.1
        else:
            cols = 3
            x_start = 0.1 + spacing / 2
        
        for col in range(cols):
            if len(points) >= n:
                break
            x = x_start + col * spacing
            points.append([x, y])
        row += 1
    
    return np.array(points[:n])


def _grid_initialization(n: int) -> np.ndarray:
    """Create a grid initialization."""
    points = []
    spacing = 0.2
    for i in range(5):
        for j in range(5):
            if len(points) >= n:
                break
            x = 0.1 + i * spacing
            y = 0.1 + j * spacing
            points.append([x, y])
    return np.array(points[:n])


def _concentric_initialization(n: int) -> np.ndarray:
    """Create a concentric circle initialization."""
    points = [[0.5, 0.5]]
    for r in [0.15, 0.3, 0.4]:
        num_pts = int(2 * np.pi * r / 0.2)
        for angle in np.linspace(0, 2 * np.pi, num_pts, endpoint=False):
            if len(points) >= n:
                break
            x = 0.5 + r * np.cos(angle)
            y = 0.5 + r * np.sin(angle)
            points.append([x, y])
    return np.array(points[:n])


def _random_initialization(n: int, seed: int) -> np.ndarray:
    """Create a random initialization."""
    rng = np.random.default_rng(seed=seed)
    return rng.random((n, 2)) * 0.8 + 0.1


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Uses multiple initialization strategies with simulated annealing and local search.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 590  # Safety margin: 10s before 600s limit
    
    best_points = None
    best_score = 0.0
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # Structured initializations
    initial_configs.append(_hexagonal_initialization(n))
    initial_configs.append(_grid_initialization(n))
    initial_configs.append(_concentric_initialization(n))
    
    # Random initializations with different seeds
    for seed in range(20):
        initial_configs.append(_random_initialization(n, seed))
    
    # Phase 1: Quick evaluation and selection of top candidates
    scored_configs = []
    for config in initial_configs:
        if time.time() > deadline - 300:
            break
        score = _compute_score_fast(config)
        scored_configs.append((score, config))
    
    # Sort by score and keep top 10
    scored_configs.sort(key=lambda x: -x[0])
    top_configs = [config for _, config in scored_configs[:10]]
    
    # Phase 2: Simulated annealing on top candidates
    for config in top_configs:
        if time.time() > deadline - 100:
            break
        
        # Run simulated annealing
        sa_result = _simulated_annealing(config, deadline - 100, 
                                           T_init=0.05, T_min=1e-6,
                                           cooling_rate=0.997, max_iters=20000)
        sa_score = _compute_score_fast(sa_result)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_result.copy()
    
    # Phase 3: Local search refinement on the best point
    if best_points is not None:
        ls_result = _local_search(best_points, deadline - 20, max_iters=5000)
        ls_score = _compute_score_fast(ls_result)
        
        if ls_score > best_score:
            best_score = ls_score
            best_points = ls_result.copy()
    
    # Phase 4: Additional random restarts if time permits
    if best_points is not None and time.time() < deadline - 60:
        for restart in range(15):
            if time.time() > deadline - 30:
                break
            
            # Perturb the best point
            perturbed = best_points.copy()
            noise = np.random.randn(*perturbed.shape) * 0.08
            perturbed += noise
            np.clip(perturbed, 0, 1, out=perturbed)
            
            # Quick local search
            quick_ls = _local_search(perturbed, deadline - 30, max_iters=2000)
            quick_score = _compute_score_fast(quick_ls)
            
            if quick_score > best_score:
                best_score = quick_score
                best_points = quick_ls.copy()
    
    # Final safety check
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _grid_initialization(n)
    
    return best_points


# EVOLVE-BLOCK-END
