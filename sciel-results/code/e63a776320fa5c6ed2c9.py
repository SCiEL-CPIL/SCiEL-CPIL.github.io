# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

def _compute_score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points using vectorized operations."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Vectorized triangle area computation
    # For each triple (i,j,k), area = |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)| / 2
    # We can compute all triangle areas at once using broadcasting
    
    # Get all combinations of 3 indices
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing='ij'
    )
    # Filter to only i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    # Compute areas vectorized
    x1 = points[i_idx, 0]
    y1 = points[i_idx, 1]
    x2 = points[j_idx, 0]
    y2 = points[j_idx, 1]
    x3 = points[k_idx, 0]
    y3 = points[k_idx, 1]
    
    areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def _local_search(points: np.ndarray, deadline: float, max_iters: int = 50000) -> np.ndarray:
    """
    Perform local search using coordinate descent with adaptive step sizes.
    """
    n = points.shape[0]
    points = points.copy()
    current_score = _compute_score(points)
    
    # Adaptive step sizes for each coordinate
    step_sizes = np.ones((n, 2)) * 0.01
    
    # Directions to try: 12 directions (8 cardinal/diagonal + 4 more)
    directions = []
    for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        directions.append(np.array([np.cos(angle), np.sin(angle)]))
    directions = np.array(directions)
    
    iter_count = 0
    stuck_count = 0
    
    while iter_count < max_iters:
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            for d in directions:
                # Try moving point i in direction d
                for step_mult in [1.0, 0.5, 0.1, 2.0]:
                    new_points = points.copy()
                    new_points[i] = points[i] + step_sizes[i] * step_mult * d
                    
                    if not np.all(np.isfinite(new_points)):
                        continue
                    
                    new_score = _compute_score(new_points)
                    
                    if new_score > current_score:
                        points = new_points
                        current_score = new_score
                        improved = True
                        # Reduce step size slightly for next iteration
                        step_sizes[i] *= 0.9
                        break
                else:
                    # No improvement in this direction, shrink step
                    step_sizes[i] *= 0.8
                    if step_sizes[i, 0] < 1e-8:
                        step_sizes[i] = 0.01 * np.ones(2)
            
            if not improved and stuck_count > 100:
                # Reset stuck count occasionally
                stuck_count = 0
                # Perturb a random point
                idx = np.random.randint(n)
                points[idx] += np.random.normal(0, 0.01, 2)
                current_score = _compute_score(points)
        
        if not improved:
            stuck_count += 1
        else:
            stuck_count = 0
        
        iter_count += 1
    
    return points

def _nelder_mead(points: np.ndarray, deadline: float, max_iter: int = 2000) -> np.ndarray:
    """
    Perform Nelder-Mead optimization on the flattened point coordinates.
    """
    n = points.shape[0]
    flat = points.flatten().copy()
    
    def objective(x_flat):
        pts = x_flat.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return -1e10
        return -_compute_score(pts)  # Minimize negative score
    
    # Initial simplex
    simplex = [flat.copy()]
    for i in range(2 * n):
        perturbed = flat.copy()
        perturbed[i] += 0.01
        simplex.append(perturbed)
    
    # Nelder-Mead implementation
    simplex = np.array(simplex)
    f_simplex = np.array([objective(s) for s in simplex])
    
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink
    
    iter_count = 0
    while iter_count < max_iter:
        if time.time() > deadline:
            break
        
        # Sort simplex by function value
        order = np.argsort(f_simplex)
        simplex = simplex[order]
        f_simplex = f_simplex[order]
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = objective(x_r)
        
        if f_r < f_simplex[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_simplex[-1] = f_e
            else:
                simplex[-1] = x_r
                f_simplex[-1] = f_r
        elif f_r < f_simplex[-2]:
            simplex[-1] = x_r
            f_simplex[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = objective(x_c)
            if f_c < f_simplex[-1]:
                simplex[-1] = x_c
                f_simplex[-1] = f_c
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_simplex[i] = objective(simplex[i])
        
        iter_count += 1
    
    return simplex[0].reshape(n, 2)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget
    
    n = 13
    best_points = None
    best_score = -1.0
    
    # Strategy 1: Regular polygon + center point
    # 12 points on a regular dodecagon + 1 at center
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    radius = 1.0
    poly_points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    initial1 = np.vstack([poly_points, center])
    
    # Strategy 2: Two concentric hexagons (6 + 6) + 1 center
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    r1 = 1.0
    r2 = 0.6
    hex1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    hex2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    center2 = np.array([[0.0, 0.0]])
    initial2 = np.vstack([hex1, hex2, center2])
    
    # Strategy 3: Random with some structure
    rng = np.random.default_rng(seed=42)
    initial3 = rng.random((n, 2)) * 2 - 1
    
    # Strategy 4: Points on a circle with varying radii
    angles4 = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii4 = np.ones(n)
    radii4[::3] = 0.7  # Every third point closer to center
    initial4 = np.column_stack([radii4 * np.cos(angles4), radii4 * np.sin(angles4)])
    
    strategies = [initial1, initial2, initial3, initial4]
    
    for initial in strategies:
        if time.time() > deadline - 30:
            break
        
        # Evaluate initial
        score = _compute_score(initial)
        if score > best_score:
            best_score = score
            best_points = initial.copy()
        
        # Local search
        if time.time() > deadline - 60:
            break
        ls_points = _local_search(initial, deadline - 30, max_iters=10000)
        ls_score = _compute_score(ls_points)
        if ls_score > best_score:
            best_score = ls_score
            best_points = ls_points.copy()
        
        # Nelder-Mead refinement
        if time.time() > deadline - 30:
            break
        nm_points = _nelder_mead(ls_points, deadline - 15, max_iter=1000)
        nm_score = _compute_score(nm_points)
        if nm_score > best_score:
            best_score = nm_score
            best_points = nm_points.copy()
        
        # Another local search round
        if time.time() > deadline - 15:
            break
        ls2_points = _local_search(nm_points, deadline - 5, max_iters=5000)
        ls2_score = _compute_score(ls2_points)
        if ls2_score > best_score:
            best_score = ls2_score
            best_points = ls2_points.copy()
    
    # Final safety: ensure valid output
    if best_points is None:
        return initial1
    
    # Ensure all points are finite
    if not np.all(np.isfinite(best_points)):
        return initial1
    
    # Verify convex hull works
    try:
        hull = ConvexHull(best_points)
    except Exception:
        return initial1
    
    return best_points

# EVOLVE-BLOCK-END
