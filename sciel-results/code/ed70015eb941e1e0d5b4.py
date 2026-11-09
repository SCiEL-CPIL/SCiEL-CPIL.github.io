# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 590.0  # 10s safety margin before 600s limit


def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by triples of points."""
    n = len(points)
    if n < 3:
        return 0.0
    indices = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))


def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        # Check for collinearity/degeneracy
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    """Compute A_min / A_hull."""
    if not np.all(np.isfinite(points)):
        return 0.0
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 1e-12 or a_min <= 1e-12:
        return 0.0
    return a_min / a_hull


def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])


def _lloyd_relax(points, n_iter=50, tol=1e-10):
    """
    Lloyd's algorithm (Voronoi centroid iteration) for interior points.
    The outer hull points are fixed; interior points are moved to the centroid
    of their Voronoi cell (clipped to the convex hull).
    """
    n = len(points)
    points = points.copy()
    
    # Identify hull points
    try:
        hull = ConvexHull(points)
        hull_indices = set(hull.vertices)
    except Exception:
        return points
    
    interior_indices = [i for i in range(n) if i not in hull_indices]
    
    if len(interior_indices) == 0:
        return points
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 30:
            break
            
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(points)
        except Exception:
            break
        
        # For each interior point, find its Voronoi cell centroid
        # We approximate by averaging the centroids of adjacent triangles
        # weighted by distance, or more simply, we can use the fact that
        # in a Delaunay triangulation, the Voronoi vertex is the circumcenter.
        # Lloyd's algorithm moves a point to the centroid of its Voronoi cell.
        # A simpler approximation: move each interior point towards the
        # centroid of its neighboring points in the Delaunay graph.
        
        # Build neighbor list from Delaunay
        neighbors = {i: set() for i in range(n)}
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(len(simplex)):
                    if i != j:
                        neighbors[simplex[i]].add(simplex[j])
        
        new_points = points.copy()
        max_change = 0.0
        
        for idx in interior_indices:
            nbrs = list(neighbors[idx])
            if len(nbrs) > 0:
                # Move to centroid of neighbors
                target = np.mean(points[nbrs], axis=0)
                # Damped step
                new_pos = 0.5 * points[idx] + 0.5 * target
                new_points[idx] = new_pos
                max_change = max(max_change, np.linalg.norm(new_pos - points[idx]))
        
        points = new_points
        
        if max_change < tol:
            break
        
        # Ensure no points go outside the hull (optional, but good for stability)
        # Actually, Lloyd's algorithm naturally keeps points inside if started inside
        # and the hull is fixed. But numerical errors might push them out.
        # We can re-project if needed, but for now we trust the convex combination.
        
    return points


def _objective_lloyd(x):
    """
    Objective: minimize -score.
    Uses Lloyd's relaxation internally to improve the configuration before scoring.
    This makes the objective smoother in the space of hull vertices.
    """
    if time.time() > _DEADLINE - 10:
        return 1e10
        
    points = x.reshape(-1, 2)
    
    # Apply Lloyd relaxation
    relaxed = _lloyd_relax(points, n_iter=20, tol=1e-8)
    
    # Score the relaxed configuration
    s = _score(relaxed)
    if s < 1e-12:
        return 1e10
    return -s


def _refine_with_powell(points, max_iter=200):
    """
    Refine a configuration using Powell's method on the exact objective.
    """
    if time.time() > _DEADLINE - 30:
        return points
        
    x0 = points.flatten()
    
    def obj(x):
        if time.time() > _DEADLINE - 10:
            return 1e10
        pts = x.reshape(-1, 2)
        s = _score(pts)
        if s < 1e-12:
            return 1e10
        return -s
    
    try:
        res = minimize(obj, x0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-8, 'ftol': 1e-8})
        return res.x.reshape(-1, 2)
    except Exception:
        return points


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize
    the area of the smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(seed=42)
    
    # Strategy: 
    # 1. Start with configurations that have a clear hull (e.g., regular polygon or concentric circles)
    # 2. Use Lloyd's algorithm to relax interior points
    # 3. Use Powell's method to fine-tune all points
    
    initial_configs = []
    
    # Config 1: Regular 13-gon
    initial_configs.append(_regular_polygon(13, radius=1.0))
    
    # Config 2: 7 outer + 6 inner (concentric)
    for r_inner in [0.3, 0.4, 0.5]:
        outer = _regular_polygon(7, radius=1.0)
        inner = _regular_polygon(6, radius=r_inner)
        # Rotate inner to avoid alignment
        angle = np.pi / 6
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        inner = inner @ rot.T
        config = np.vstack([outer, inner])
        initial_configs.append(config)
    
    # Config 3: Random points
    for _ in range(5):
        pts = rng.uniform(-1, 1, size=(13, 2))
        initial_configs.append(pts)
    
    # Process each initial configuration
    for config in initial_configs:
        if time.time() > _DEADLINE - 60:
            break
        
        # Step 1: Lloyd relaxation
        relaxed = _lloyd_relax(config, n_iter=50, tol=1e-8)
        
        # Step 2: Powell refinement
        refined = _refine_with_powell(relaxed, max_iter=200)
        
        # Step 3: Evaluate
        score = _score(refined)
        if score > best_score:
            best_score = score
            best_points = refined.copy()
        
        # Also try refining the original config directly
        if time.time() > _DEADLINE - 30:
            break
        refined_direct = _refine_with_powell(config, max_iter=100)
        score_direct = _score(refined_direct)
        if score_direct > best_score:
            best_score = score_direct
            best_points = refined_direct.copy()
    
    # If no valid result found, fall back to regular polygon
    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)
    
    # Final verification and cleanup
    best_points = np.asarray(best_points, dtype=np.float64)
    if not np.all(np.isfinite(best_points)):
        best_points = rng.uniform(-1, 1, size=(13, 2))
    
    return best_points

# EVOLVE-BLOCK-END
