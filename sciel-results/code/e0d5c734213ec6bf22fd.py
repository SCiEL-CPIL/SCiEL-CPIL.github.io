import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute triple indices for vectorization
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    """
    Compute areas of all triangles formed by triples of points.
    Returns array of shape (num_triangles,)
    """
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        if len(points) < 3:
            return 0.0
        # Check for degenerate cases
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min_triangle_area / convex_hull_area
    """
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _soft_min_value(values, mu):
    """
    Smooth approximation of the minimum of values.
    As mu -> 0, this approaches min(values).
    """
    if mu <= 1e-10:
        return np.min(values)
    # Log-sum-exp trick for stability
    scaled = values / mu
    max_scaled = np.max(scaled)
    if max_scaled > 500:
        # Prevent overflow in exp
        return mu * (max_scaled + np.log(np.mean(np.exp(scaled - max_scaled))))
    else:
        return mu * (max_scaled + np.log(np.mean(np.exp(scaled - max_scaled))))

def _soft_score(points, mu):
    """
    Smooth approximation of the score.
    """
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    soft_min = _soft_min_value(areas, mu)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area

def _neg_soft_score(points_flat, mu):
    """
    Negative soft score for minimization.
    """
    points = points_flat.reshape(N, 2)
    s = _soft_score(points, mu)
    if s <= 0:
        return 1e10
    return -s

def _regular_polygon(n, radius=1.0, angle_offset=0.0):
    angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _optimize_with_annealing(initial_points, deadline, start_mu=0.05, end_mu=1e-6, steps=10):
    """
    Optimize the points using an annealing schedule on the soft-min parameter mu.
    """
    current_points = initial_points.copy()
    best_points = initial_points.copy()
    best_score = _score(best_points)
    
    # Log-spaced mu values
    mus = np.logspace(np.log10(start_mu), np.log10(end_mu), steps)
    
    # Allocate time for each step
    elapsed = time.time()
    total_deadline = deadline
    time_per_step = (total_deadline - elapsed - 5.0) / steps
    
    for i, mu in enumerate(mus):
        if time.time() > total_deadline - 5.0:
            break
        
        # Define objective for this mu
        def obj(x):
            return _neg_soft_score(x, mu)
        
        x0 = current_points.flatten()
        
        try:
            # Use Powell or BFGS. Powell is derivative-free and robust for this non-smooth-ish landscape.
            # BFGS might be faster if gradients were available, but we don't have them easily.
            # Let's stick with Powell but with tight tolerances.
            res = minimize(
                obj, 
                x0, 
                method='Powell',
                options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            new_points = res.x.reshape(N, 2)
            current_points = new_points
            
            # Evaluate true score
            s = _score(current_points)
            if s > best_score:
                best_score = s
                best_points = current_points.copy()
                
        except Exception:
            pass
            
    return best_points, best_score

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish by 140s.
    deadline = start_time + 135.0
    
    best_points = None
    best_score = 0.0
    
    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # 1. Initial Seeds
    # Regular 13-gon
    seed_regular = _regular_polygon(N, 1.0)
    _update(seed_regular)
    
    # Slightly perturbed regular 13-gon to break symmetry if needed
    np.random.seed(42)
    perturbations = np.random.normal(0, 0.05, (N, 2))
    seed_perturbed = seed_regular + perturbations
    _update(seed_perturbed)
    
    # 2. Annealing Optimization from Regular Polygon
    if time.time() < deadline - 20:
        try:
            pts, s = _optimize_with_annealing(seed_regular, deadline, start_mu=0.05, end_mu=1e-5, steps=8)
            _update(pts)
        except Exception:
            pass

    # 3. Annealing Optimization from Perturbed Seed
    if time.time() < deadline - 20:
        try:
            pts, s = _optimize_with_annealing(seed_perturbed, deadline, start_mu=0.03, end_mu=1e-5, steps=8)
            _update(pts)
        except Exception:
            pass
            
    # 4. Final Polish with very low mu on the best result found so far
    if best_points is not None and time.time() < deadline - 5:
        try:
            # Use a single very low mu step to fine-tune
            def obj_final(x):
                return _neg_soft_score(x, 1e-8)
            
            res = minimize(
                obj_final,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            cand = res.x.reshape(N, 2)
            _update(cand)
        except Exception:
            pass
            
    # 5. Fallback
    if best_points is None:
        best_points = seed_regular
        
    return best_points
