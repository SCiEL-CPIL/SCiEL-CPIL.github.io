# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the raw score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Check for duplicates or near-duplicates which might cause issues
    # though ConvexHull handles them, we want robust scoring.
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Compute all triangle areas vectorized
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 0:
        return 0.0
        
    return min_area / A_hull

def _smoothed_objective(params: np.ndarray, beta: float = 50.0) -> float:
    """
    Objective function for minimization (negative score).
    Uses a soft-min approximation for the minimum triangle area to make it differentiable.
    f(x) = - (soft_min(areas) / A_hull)
    soft_min(a) = -1/beta * log(sum(exp(-beta * a_i)))
    This is differentiable everywhere and approximates min(a) as beta -> inf.
    """
    points = params.reshape(13, 2)
    
    # Basic validity check: ensure points are in a reasonable range
    # If points fly away, score is likely bad, but we rely on bounds in minimize.
    
    n = points.shape[0]
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 1e6 # Large penalty
    
    if A_hull <= 1e-10:
        return 1e6
        
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min calculation
    # To avoid overflow in exp, shift by max
    min_area_approx = -1.0 / beta * np.log(np.sum(np.exp(-beta * areas)))
    
    score = min_area_approx / A_hull
    return -score

def _generate_initializations():
    """
    Generate diverse initial configurations for the optimizer.
    """
    inits = []
    
    # 1. Regular 13-gon (points on a circle)
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    pts = np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])
    inits.append(pts)
    
    # 2. Hexagonal packing approximation (3-4-4-2 rows)
    pts = []
    # Row 0: 3 pts
    for x in [0.2, 0.5, 0.8]: pts.append([x, 0.8])
    # Row 1: 4 pts
    for x in [0.1, 0.36, 0.63, 0.9]: pts.append([x, 0.53])
    # Row 2: 4 pts
    for x in [0.1, 0.36, 0.63, 0.9]: pts.append([x, 0.26])
    # Row 3: 2 pts
    for x in [0.3, 0.7]: pts.append([x, 0.0])
    inits.append(np.array(pts))
    
    # 3. Concentric rings (1 center, 6 inner, 6 outer)
    pts = [[0.5, 0.5]]
    r1 = 0.2
    for i in range(6):
        a = np.pi / 3 * i
        pts.append([0.5 + r1 * np.cos(a), 0.5 + r1 * np.sin(a)])
    r2 = 0.4
    for i in range(6):
        a = np.pi / 3 * i + np.pi / 6
        pts.append([0.5 + r2 * np.cos(a), 0.5 + r2 * np.sin(a)])
    inits.append(np.array(pts))
    
    # 4. Random points (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    for _ in range(10):
        pts = rng.random((13, 2)) * 0.8 + 0.1
        inits.append(pts)
        
    return inits

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area using gradient-based optimization.
    """
    start_time = time.time()
    deadline = 600  # seconds
    
    inits = _generate_initializations()
    
    best_points = None
    best_score = -np.inf
    
    # Bounds: keep points in [0, 1] x [0, 1]
    bounds = [(0.01, 0.99)] * 26
    
    # We will run L-BFGS-B on each initialization.
    # Since L-BFGS-B is deterministic given an init, we don't need many restarts 
    # if we have diverse inits.
    
    for i, init_pts in enumerate(inits):
        if time.time() - start_time > deadline - 30:
            break
            
        init_params = init_pts.flatten()
        
        try:
            # Use L-BFGS-B for constrained optimization
            res = minimize(
                _smoothed_objective, 
                init_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success:
                opt_points = res.x.reshape(13, 2)
                score = _compute_score(opt_points)
                
                if score > best_score:
                    best_score = score
                    best_points = opt_points
                    
        except Exception:
            continue
            
    # If optimization failed for all, fallback to best init
    if best_points is None:
        best_points = inits[0]
        
    # Final sanity check: ensure no NaNs and shape is correct
    if not np.all(np.isfinite(best_points)):
        # Fallback to a simple grid if something went wrong
        best_points = np.array([
            [0.2, 0.8], [0.5, 0.8], [0.8, 0.8],
            [0.1, 0.53], [0.36, 0.53], [0.63, 0.53], [0.9, 0.53],
            [0.1, 0.26], [0.36, 0.26], [0.63, 0.26], [0.9, 0.26],
            [0.3, 0.0], [0.7, 0.0]
        ])
        
    return best_points

# EVOLVE-BLOCK-END
