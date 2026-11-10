import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _compute_circumradii(points):
    """Compute circumradius of each facet."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.array([]), None
    
    facets = hull.simplices
    n_facets = len(facets)
    if n_facets == 0:
        return np.array([]), None
    
    circumradii = np.zeros(n_facets)
    circumcenters = np.zeros((n_facets, 3))
    
    for i, tri in enumerate(facets):
        p0, p1, p2 = points[tri[0]], points[tri[1]], points[tri[2]]
        
        # Normal to the triangle plane
        n = np.cross(p0 - p1, p0 - p2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            circumradii[i] = np.pi
            continue
        
        c = n / n_norm
        # Ensure c is on the same side as the origin relative to the triangle? 
        # The circumcenter is the intersection of the sphere and the line through the origin 
        # perpendicular to the triangle plane.
        # There are two candidates: c and -c.
        # The correct one is the one that is "inside" the triangle's cap.
        # Check dot product with the triangle vertices.
        # If the triangle is small, the circumcenter is close to the centroid.
        # Centroid g = (p0+p1+p2)/3
        # If dot(c, g) > 0, c is likely the correct one (same hemisphere as the triangle).
        g = (p0 + p1 + p2) / 3.0
        if np.dot(c, g) < 0:
            c = -c
            
        circumcenters[i] = c
        
        # Circumradius is angular distance from c to p0
        dot_val = np.clip(np.dot(c, p0), -1.0, 1.0)
        circumradii[i] = np.arccos(dot_val)
        
    return circumradii, circumcenters

def _lloyd_step(points):
    """Perform one Lloyd's algorithm step."""
    circumradii, circumcenters = _compute_circumradii(points)
    if len(circumradii) == 0:
        return points
        
    hull = ConvexHull(points)
    facets = hull.simplices
    
    # For each point, collect the circumcenters of the facets it belongs to
    # and weight them by the area of the facet (or just count them)
    new_points = np.zeros_like(points)
    
    # Calculate area of each facet (spherical area)
    # Area of spherical triangle is 2 * arctan(|det(p0,p1,p2)| / (1 + p0.p1 + p1.p2 + p2.p0))
    # Or simpler: area = 2 * atan( |p0 x p1| . p2 / (1 + p0.p1 + p1.p2 + p2.p0) )
    areas = np.zeros(len(facets))
    for i, tri in enumerate(facets):
        p0, p1, p2 = points[tri[0]], points[tri[1]], points[tri[2]]
        # Spherical excess formula or vector formula
        # Vector formula for area of spherical triangle with vertices on unit sphere:
        # Area = 2 * atan( |det([p0,p1,p2])| / (1 + p0.p1 + p1.p2 + p2.p0) )
        det_val = np.dot(np.cross(p0, p1), p2)
        denom = 1.0 + np.dot(p0, p1) + np.dot(p1, p2) + np.dot(p2, p0)
        if denom < 1e-10:
            areas[i] = 0
        else:
            areas[i] = 2.0 * np.arctan(abs(det_val) / denom)
            
    # Initialize accumulators
    acc = np.zeros_like(points)
    weights = np.zeros(N)
    
    for i, tri in enumerate(facets):
        w = areas[i]
        if w < 1e-10:
            w = 1.0 # Fallback
        # Add the circumcenter of this facet to the accumulator for each vertex
        c = circumcenters[i]
        for j in tri:
            acc[j] += w * c
            weights[j] += w
            
    # Normalize
    for i in range(N):
        if weights[i] > 1e-10:
            new_points[i] = acc[i] / weights[i]
        else:
            new_points[i] = points[i]
            
    return _normalize(new_points)

def _optimize_lloyd(points, time_budget):
    """Optimize using Lloyd's algorithm."""
    start_time = time.time()
    current_points = points.copy()
    
    while time.time() - start_time < time_budget:
        next_points = _lloyd_step(current_points)
        
        # Check if points moved significantly
        diff = np.linalg.norm(next_points - current_points)
        current_points = next_points
        
        if diff < 1e-6:
            break
            
    return current_points

def _polish_nelder(points, time_budget):
    """Polish using Nelder-Mead on the max circumradius."""
    start_time = time.time()
    
    def objective(x):
        pts = _normalize(x.reshape(N, 3))
        radii, _ = _compute_circumradii(pts)
        if len(radii) == 0:
            return np.pi
        return np.max(radii)
        
    x0 = points.flatten()
    
    try:
        res = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if res.fun < objective(x0):
            return _normalize(res.x.reshape(N, 3))
    except Exception:
        pass
        
    return points

def _fibonacci_spiral(N, offset=0.5):
    i = np.arange(N) + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_points = None
    best_score = np.pi
    
    starts = [
        _fibonacci_spiral(N, 0.5),
        _fibonacci_spiral(N, 0.0),
        np.random.RandomState(42).randn(N, 3),
        np.random.RandomState(123).randn(N, 3),
        np.random.RandomState(456).randn(N, 3)
    ]
    
    for start in starts:
        if time.time() > deadline - 20:
            break
            
        start_points = _normalize(start)
        
        # Phase 1: Lloyd's Algorithm (Global-ish refinement)
        t_start = time.time()
        lloyd_budget = min(30, deadline - t_start - 20) # Leave 20s for polish and next starts
        if lloyd_budget > 5:
            points_lloyd = _optimize_lloyd(start_points, lloyd_budget)
        else:
            points_lloyd = start_points
            
        # Evaluate Lloyd result
        radii, _ = _compute_circumradii(points_lloyd)
        score_lloyd = np.max(radii) if len(radii) > 0 else np.pi
        
        # Phase 2: Nelder-Mead Polish
        t_start = time.time()
        polish_budget = min(30, deadline - t_start - 10)
        if polish_budget > 5:
            points_polished = _polish_nelder(points_lloyd, polish_budget)
        else:
            points_polished = points_lloyd
            
        # Evaluate Polished result
        radii, _ = _compute_circumradii(points_polished)
        score_polished = np.max(radii) if len(radii) > 0 else np.pi
        
        # Update best
        if score_polished < best_score:
            best_score = score_polished
            best_points = points_polished.copy()
            
        if score_lloyd < best_score:
            best_score = score_lloyd
            best_points = points_lloyd.copy()
            
    if best_points is None:
        best_points = _fibonacci_spiral(N)
        
    return _normalize(best_points)
