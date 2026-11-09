# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

def _score_fast(points):
    """Compute A_min / A_hull for a set of points using vectorized operations."""
    if len(points) < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull <= 1e-12:
        return 0.0
    
    n = len(points)
    # Vectorized triangle area computation using meshgrid
    indices = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(indices, indices, indices, indexing='ij')
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Only keep unique triples i < j < k
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_flat = i_flat[mask]
    j_flat = j_flat[mask]
    k_flat = k_flat[mask]
    
    p_i = points[i_flat]
    p_j = points[j_flat]
    p_k = points[k_flat]
    
    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    min_area = np.min(areas)
    if min_area <= 1e-12:
        return 0.0
    return min_area / a_hull

def _score(points):
    """Compute A_min / A_hull for a set of points."""
    return _score_fast(points)

def _negative_score_flat(flat_points):
    """Objective function for differential evolution (minimize negative score)."""
    points = flat_points.reshape(13, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    s = _score_fast(points)
    if s <= 0:
        return 1e10
    return -s

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area using differential evolution.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    n = 13
    best_points = None
    best_score = 0.0
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Try multiple initializations to find good starting points
    initializations = []
    
    # 1. Hexagonal lattice pattern
    hex_pts = np.zeros((n, 2))
    hex_pts[0] = [0.0, 0.0]
    for i in range(6):
        angle = i * np.pi / 3
        hex_pts[i + 1] = [np.cos(angle), np.sin(angle)]
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        hex_pts[i + 7] = [np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)]
    initializations.append(hex_pts)
    
    # 2. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    initializations.append(reg_pts)
    
    # 3. Grid-based
    grid_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0],
        [0.5, 0.5],
        [1.5, 0.5],
        [0.5, 1.5],
        [1.5, 1.5],
    ], dtype=float)
    initializations.append(grid_pts)
    
    # 4. Random points in unit square
    rng = np.random.default_rng(seed=12345)
    for _ in range(5):
        initializations.append(rng.random((n, 2)) * 2 - 1)
    
    # Evaluate initializations
    for init_pts in initializations:
        if time.time() > deadline - 60:
            break
        s = _score_fast(init_pts)
        update_best(init_pts, s)
    
    # Run differential evolution from the best initialization
    if time.time() < deadline - 60:
        # Define bounds for each coordinate
        bounds = [(-2, 2)] * (2 * n)
        
        # Use the best initialization as the initial population center
        best_init = best_points.flatten()
        
        # Create initial population around the best point
        pop_size = 30
        initial_pop = np.array([best_init + rng.normal(size=2*n) * 0.5 for _ in range(pop_size)])
        # Make sure all points in initial population are valid
        for i in range(pop_size):
            pts = initial_pop[i].reshape(n, 2)
            if not np.all(np.isfinite(pts)):
                initial_pop[i] = best_init + rng.normal(size=2*n) * 0.1
        
        # Run differential evolution with time limit
        try:
            result = differential_evolution(
                _negative_score_flat,
                bounds,
                maxiter=100,
                popsize=pop_size,
                tol=1e-6,
                mutation=(0.5, 1.5),
                recombination=0.7,
                seed=42,
                polish=True,
                init='sobol',
                x0=best_init,
                updating='immediate',
                callback=None
            )
            
            if result.success or result.fun < 0:
                de_points = result.x.reshape(n, 2)
                s = _score_fast(de_points)
                if s > 0:
                    update_best(de_points, s)
        except Exception as e:
            pass
    
    # Local refinement using coordinate descent from the best point found
    if best_points is not None and time.time() < deadline - 30:
        current_points = best_points.copy()
        current_score = best_score
        
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for base_step in step_sizes:
            if time.time() > deadline - 30:
                break
            
            improved = True
            while improved:
                improved = False
                for i in range(n):
                    if time.time() > deadline - 30:
                        break
                    
                    # Try multiple directions
                    for _ in range(20):
                        direction = rng.normal(size=2)
                        norm = np.linalg.norm(direction)
                        if norm < 1e-10:
                            continue
                        direction /= norm
                        
                        for mult in [1.0, 0.5, 0.25, 2.0]:
                            step = base_step * mult
                            trial_points = current_points.copy()
                            trial_points[i] = current_points[i] + step * direction
                            s = _score_fast(trial_points)
                            if s > current_score:
                                current_points = trial_points
                                current_score = s
                                improved = True
                                break
                        if improved:
                            break
                
                if time.time() > deadline - 30:
                    break
            
            update_best(current_points, current_score)
    
    # Final validation
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Fallback
        fallback = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
        ], dtype=float)
        return fallback
    
    final_score = _score_fast(best_points)
    if final_score <= 0:
        fallback = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
        ], dtype=float)
        return fallback
    
    return best_points

# EVOLVE-BLOCK-END
