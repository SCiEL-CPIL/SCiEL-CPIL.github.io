"""
Strategy: Geometrically constrained optimization using a differentiable soft-min objective.
Instead of optimizing raw coordinates which can lead to points falling inside the convex hull
(and thus not contributing to the hull area, or creating very small triangles with hull vertices),
we enforce that all points are on the convex hull by sorting them by polar angle around their
centroid. This reduces the effective search space and ensures the convex hull area is well-defined 
and maximized relative to the point spread.

We use a smooth approximation of the minimum (soft-min) of the triangle areas to make the 
objective differentiable, allowing the use of L-BFGS-B, which is much faster and more robust 
for smooth objectives than Nelder-Mead or Differential Evolution for high-dimensional problems.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

def _all_triangle_areas(points):
    """
    Compute areas of all triangles formed by the points.
    Returns a 1D array of shape (C(N,3),).
    """
    n = points.shape[0]
    # Generate all unique triples
    # For N=14, C(14,3) = 364. This is small enough for explicit loops or vectorization.
    # Vectorized approach for speed in the objective function.
    # We can use np.triu_indices to get pairs, but for triples it's easier to just loop 
    # or use a precomputed index array.
    # Let's precompute the indices for the 364 triangles to avoid overhead.
    pass

# Precompute triangle indices for N=14
def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices)

TRI_INDICES = _get_triangle_indices(N)

def _compute_triangle_areas(points):
    """
    Vectorized computation of triangle areas.
    points: (N, 2)
    Returns: (C(N,3),) array of areas
    """
    # points[i], points[j], points[k]
    p_i = points[TRI_INDICES[:, 0]]  # (364, 2)
    p_j = points[TRI_INDICES[:, 1]]  # (364, 2)
    p_k = points[TRI_INDICES[:, 2]]  # (364, 2)
    
    # Vectors
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product (2D)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    # Area is 0.5 * |cross|
    areas = 0.5 * np.abs(cross)
    return areas

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score is min_triangle_area / convex_hull_area"""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _compute_triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min(x, k=100.0):
    """
    Smooth approximation of the minimum.
    min(x) ≈ -1/k * log(sum(exp(-k * x)))
    This is differentiable.
    """
    # To avoid overflow/underflow, shift by min
    # However, for k=100, exp(-100 * x) can be very small.
    # Standard stable soft-min:
    # m = min(x)
    # soft_min = m - (1/k) * log(sum(exp(-k * (x - m))))
    m = np.min(x)
    # If all x are equal, sum(exp(0)) = len(x), log(len(x))/k is small.
    # If one x is much smaller, it dominates.
    exp_terms = np.exp(-k * (x - m))
    log_sum = np.log(np.sum(exp_terms))
    return m - (1.0 / k) * log_sum

def _objective(flat_points, k=100.0):
    """
    Objective function for minimization.
    We want to maximize min_area / hull_area.
    So we minimize - (soft_min(areas) / hull_area).
    
    To make it more stable and differentiable, we can minimize:
    - soft_min(areas) + lambda * hull_area
    But the ratio is the specific metric.
    
    Let's minimize: - (soft_min(areas) / hull_area)
    
    Note: The soft-min is a lower bound on the true min.
    As k -> inf, soft_min -> min.
    """
    points = flat_points.reshape(N, 2)
    
    # Enforce convexity by sorting by angle around centroid
    # This ensures all points are on the hull (unless collinear/degenerate)
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sort_idx = np.argsort(angles)
    points = points[sort_idx]
    
    # If points are identical or degenerate, return a large penalty
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-6:
        return 1e10
        
    areas = _compute_triangle_areas(points)
    
    # Use soft min to make it differentiable
    sm = _soft_min(areas, k=100.0)
    
    # We want to maximize sm / hull_area
    # So minimize -sm / hull_area
    return -sm / hull_area

def _generate_initializations():
    """
    Generate diverse initial configurations.
    1. Regular polygon
    2. Regular polygon with slight perturbations
    3. Random convex points (sorted by angle)
    """
    inits = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(reg_poly.flatten())
    
    # 2. Perturbed regular polygons
    for i in range(5):
        np.random.seed(i)
        noise = np.random.normal(0, 0.1, (N, 2))
        perturbed = reg_poly + noise
        # Sort by angle to ensure convexity order
        centroid = np.mean(perturbed, axis=0)
        angles_p = np.arctan2(perturbed[:, 1] - centroid[1], perturbed[:, 0] - centroid[0])
        sort_idx = np.argsort(angles_p)
        perturbed = perturbed[sort_idx]
        inits.append(perturbed.flatten())
        
    # 3. Random convex points
    for i in range(5):
        np.random.seed(100 + i)
        # Generate random angles and radii
        angles_r = np.sort(np.random.uniform(0, 2*np.pi, N))
        radii_r = np.random.uniform(0.8, 1.2, N)
        points_r = np.column_stack([radii_r * np.cos(angles_r), radii_r * np.sin(angles_r)])
        inits.append(points_r.flatten())
        
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = -np.inf
    best_points = None
    
    inits = _generate_initializations()
    
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                _objective, 
                init, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline:
                break
                
            candidate_points = res.x.reshape(N, 2)
            
            # Ensure the final points are sorted by angle for consistent output (though not strictly required for score)
            centroid = np.mean(candidate_points, axis=0)
            angles = np.arctan2(candidate_points[:, 1] - centroid[1], candidate_points[:, 0] - centroid[0])
            sort_idx = np.argsort(angles)
            candidate_points = candidate_points[sort_idx]
            
            # Calculate true score (using hard min, not soft min)
            true_score = _score(candidate_points)
            
            if true_score > best_score:
                best_score = true_score
                best_points = candidate_points
                
        except Exception:
            pass
            
    # Fallback to regular polygon if optimization failed or didn't improve
    if best_points is None or best_score < _score(_generate_initializations()[0].reshape(N, 2)):
        # Use the regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
