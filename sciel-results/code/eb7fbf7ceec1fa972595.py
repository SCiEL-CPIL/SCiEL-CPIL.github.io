import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time

N = 14

# Precompute triple indices for vectorized area calculation
_i_list, _j_list, _k_list = [], [], []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _i_list.append(_i)
            _j_list.append(_j)
            _k_list.append(_k)
_TRIPLE_I = np.array(_i_list)
_TRIPLE_J = np.array(_j_list)
_TRIPLE_K = np.array(_k_list)

def _compute_min_area(points):
    """Vectorized minimum triangle area for all triples."""
    p1 = points[_TRIPLE_I]
    p2 = points[_TRIPLE_J]
    p3 = points[_TRIPLE_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return np.min(0.5 * np.abs(cross))

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def get_score(points):
    """Score = min triangle area / convex hull area."""
    ha = _hull_area(points)
    if ha <= 1e-12:
        return 0.0
    ma = _compute_min_area(points)
    if ma < 1e-15:
        return 0.0
    return ma / ha

def objective_full(x_flat):
    """Objective function for 2D points (minimize negative score)."""
    points = x_flat.reshape(N, 2)
    score = get_score(points)
    return -score

def objective_angles(angles):
    """Objective function for points on a unit circle (minimize negative score)."""
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack((x, y))
    score = get_score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # 1. Initialize with regular polygon
    regular_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    regular_points = np.column_stack((np.cos(regular_angles), np.sin(regular_angles)))
    regular_score = get_score(regular_points)
    best_score = regular_score
    best_points = regular_points.copy()
    
    # 2. Global search on angles (14D)
    # Differential Evolution is good for this smooth-ish 1D manifold problem
    # Bounds for angles: [0, 2*pi]
    # We fix the first angle to 0 to break rotational symmetry? 
    # Actually, DE handles symmetries okay, but fixing one point reduces dimensionality.
    # Let's fix point 0 at angle 0.
    # So we optimize angles 1 to 13.
    
    def obj_fixed_angles(a):
        full_angles = np.concatenate(([0.0], a))
        # Ensure sorted order? The hull area and triangle areas are invariant to permutation 
        # IF we consider the set of points. But the "min triangle" is over all triples.
        # So the order doesn't matter for the score.
        # However, for numerical stability and to avoid duplicate configurations, 
        # sorting isn't strictly necessary for the score calculation.
        x = np.cos(full_angles)
        y = np.sin(full_angles)
        points = np.column_stack((x, y))
        score = get_score(points)
        return -score
    
    bounds = [(0, 2*np.pi)] * (N-1)
    
    try:
        if time.time() < deadline - 60.0:
            de_result = differential_evolution(
                obj_fixed_angles, 
                bounds, 
                seed=42, 
                maxiter=100, 
                tol=1e-6,
                popsize=15,
                workers=-1 # Use all cores if possible, but might be overhead for small N
            )
            
            if de_result.fun < -regular_score:
                best_angles = np.concatenate(([0.0], de_result.x))
                # Sort angles to ensure consistent ordering for refinement
                best_angles = np.sort(best_angles)
                best_points_circle = np.column_stack((np.cos(best_angles), np.sin(best_angles)))
                score_circle = get_score(best_points_circle)
                
                if score_circle > best_score:
                    best_score = score_circle
                    best_points = best_points_circle.copy()
    except Exception:
        pass

    # 3. Local Refinement in 2D (28D)
    # Start from the best circular configuration and allow points to move radially/angly
    # Use Powell's method which is robust for non-smooth objectives (min function)
    
    initializations = [best_points.flatten()]
    
    # Also try a few perturbed versions of the regular polygon in 2D
    rng = np.random.default_rng(42)
    for i in range(5):
        noise = rng.normal(0, 0.1, (N, 2))
        init = regular_points + noise
        initializations.append(init.flatten())
        
    # And the DE result if it was different
    if 'best_points_circle' in locals():
         initializations.append(best_points_circle.flatten())

    options = {'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-8}
    
    for idx, init_flat in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
            
        try:
            result = minimize(objective_full, init_flat, method='Powell', options=options)
            
            if result.fun < 0:
                points = result.x.reshape(N, 2)
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
        except Exception:
            pass

    # 4. Final check and return
    if best_points is None:
        best_points = regular_points

    return best_points
