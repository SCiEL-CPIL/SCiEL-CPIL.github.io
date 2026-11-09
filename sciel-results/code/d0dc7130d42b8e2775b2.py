# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time
import itertools

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for validity: finite, non-collinear
    if not np.all(np.isfinite(points)):
        return 0.0
        
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
        
    if A_hull <= 1e-12:
        return 0.0
        
    # Vectorized area calculation for all triangles
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the Heilbronn ratio.
    Strategy:
    1. Use Differential Evolution (global search) on a smooth surrogate objective
       (negative log-sum-exp of areas normalized by hull area) to find a good basin.
    2. Use Nelder-Mead (local search) on the EXACT non-smooth objective to polish.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # 50s safety margin for 630s limit
    
    # Pre-compute triangle indices for speed
    idx = np.array(list(itertools.combinations(range(n), 3)))
    i1, i2, i3 = idx[:, 0], idx[:, 1], idx[:, 2]
    
    def get_areas(pts):
        """Vectorized area calculation."""
        p1 = pts[i1]
        p2 = pts[i2]
        p3 = pts[i3]
        v1 = p2 - p1
        v2 = p3 - p1
        det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(det)

    def get_hull_area(pts):
        """Calculate convex hull area."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 0.0

    def smooth_objective(x):
        """
        Smooth surrogate for global search.
        Minimizes -log(min_area) + log(hull_area) approximately.
        We use -logsumexp(-areas) as a smooth approximation of min(areas).
        Objective to minimize: - ( -logsumexp(-areas) / A_hull ) 
        which is equivalent to maximizing the smooth min area ratio.
        
        Actually, to be safe with DE, let's minimize:
        f(x) = - ( logsumexp(-areas) * A_hull ) ... no, that's complex.
        
        Let's stick to: Minimize - (smooth_min_area / A_hull)
        smooth_min_area = -logsumexp(-areas)
        """
        pts = x.reshape((n, 2))
        areas = get_areas(pts)
        
        # Avoid log(0)
        areas = np.maximum(areas, 1e-12)
        
        # Calculate hull area
        A_hull = get_hull_area(pts)
        if A_hull < 1e-12:
            return 1e6 # Penalty for degenerate hull
            
        # Smooth minimum area approximation
        # -logsumexp(-areas) is a smooth lower bound on min(areas)
        neg_areas = -areas
        max_neg = np.max(neg_areas)
        lse_neg = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
        smooth_min_area = -lse_neg
        
        # We want to maximize smooth_min_area / A_hull
        # So we minimize - (smooth_min_area / A_hull)
        return -(smooth_min_area / A_hull)

    def exact_objective(x):
        """
        Exact objective for local polishing.
        Minimizes - (A_min / A_hull)
        """
        pts = x.reshape((n, 2))
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    # Define bounds for the search space. 
    # Normalizing to [-1, 1] box is a good start, but the hull area varies.
    # DE works in a bounded space. Let's search in [-1, 1] x [-1, 1].
    bounds = [(-1, 1)] * (2 * n)
    
    best_points = None
    best_score = 0.0
    
    # 1. Global Search with Differential Evolution
    # DE can be slow. We limit maxiter and popsize.
    # popsize=10, maxiter=100 is ~1000 evaluations.
    # Each evaluation is ~286 triangle areas + hull. ~1ms? 
    # 1000 * 1ms = 1s. So we can afford more.
    # Let's try popsize=20, maxiter=200. ~4000 evals.
    
    if time.time() < deadline - 120: # Leave 2 min for local polish
        try:
            de_result = differential_evolution(
                smooth_objective, 
                bounds, 
                maxiter=200, 
                popsize=20, 
                seed=42,
                tol=1e-6,
                polish=False # We will do our own polish
            )
            
            if de_result.success or de_result.fun < 1e5:
                candidate_pts = de_result.x.reshape((n, 2))
                candidate_score = compute_score(candidate_pts)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_pts.copy()
                    
        except Exception:
            pass

    # 2. Local Polishing with Nelder-Mead on the exact objective
    # Start from the best point found by DE, and also from a few diverse initial points
    # to ensure we don't get stuck if DE failed.
    
    inits = []
    if best_points is not None:
        inits.append(best_points.flatten())
    
    # Add some diverse inits
    rng = np.random.default_rng(seed=123)
    
    # Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init_regular = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(init_regular.flatten())
    
    # Random
    init_random = rng.uniform(-1, 1, (n, 2))
    inits.append(init_random.flatten())
    
    # Concentric
    center = np.array([[0, 0]])
    angles_inner = np.linspace(0, 2*np.pi, 6, endpoint=False)
    angles_outer = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    inner_pts = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    outer_pts = np.column_stack([1.0 * np.cos(angles_outer), 1.0 * np.sin(angles_outer)])
    init_concentric = np.vstack([center, inner_pts, outer_pts])
    inits.append(init_concentric.flatten())
    
    # Run Nelder-Mead on each init
    for x0 in inits:
        if time.time() > deadline - 30:
            break
            
        try:
            nm_result = minimize(
                exact_objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if nm_result.fun < 1e5: # Valid
                candidate_pts = nm_result.x.reshape((n, 2))
                candidate_score = compute_score(candidate_pts)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_pts.copy()
                    
        except Exception:
            pass

    # Fallback if nothing worked
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points

# EVOLVE-BLOCK-END
