"""Rewritten program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 130

def _covering_radius_fast(points):
    """Fast computation of covering radius using convex hull circumradii.
    Returns covering radius in degrees."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms < 1e-15):
        return 90.0
    points = points / norms
    
    # Check for coincident points
    dists = 1.0 - points @ points.T
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-10:
        return 90.0
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0
    
    # Check origin is inside hull (not in closed hemisphere)
    # For each facet, check that origin is on the interior side
    for simplex in hull.simplices:
        facet = points[simplex]
        # Outward normal
        n = np.cross(facet[1] - facet[0], facet[2] - facet[0])
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            return 90.0
        n = n / n_norm
        # Origin should be on interior side: n·0 < n·facet[0] (since n points outward)
        if np.dot(n, facet[0]) < 1e-10:
            return 90.0
    
    # Compute max circumradius over all facets
    max_rad = 0.0
    for simplex in hull.simplices:
        p0, p1, p2 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Angular side lengths
        a = math.acos(np.clip(np.dot(p1, p2), -1.0, 1.0))
        b = math.acos(np.clip(np.dot(p0, p2), -1.0, 1.0))
        c = math.acos(np.clip(np.dot(p0, p1), -1.0, 1.0))
        
        # Spherical circumradius formula:
        # For a spherical triangle with sides a, b, c, the circumradius R satisfies:
        # tan(R) = sin(a)sin(b)sin(c) / (sin(a)sin(b)cos(c) + sin(b)sin(c)cos(a) + sin(c)sin(a)cos(b))
        # But this can be unstable. Use the formula via the circumcenter.
        
        # The circumcenter C on the sphere satisfies C·p0 = C·p1 = C·p2 = cos(R)
        # C is parallel to the normal of the plane through p0,p1,p2
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            # Degenerate - use max distance to centroid
            centroid = (p0 + p1 + p2) / 3.0
            centroid = centroid / np.linalg.norm(centroid)
            max_d = max(math.acos(np.clip(np.dot(centroid, p), -1.0, 1.0)) for p in [p0, p1, p2])
            max_rad = max(max_rad, math.degrees(max_d))
            continue
        n = n / n_norm
        
        # Two candidates: n and -n
        # The correct circumcenter is the one where C·p_i is the same for all i
        # and C is on the "interior" side (toward origin for convex hull)
        for C in [n, -n]:
            d0 = np.dot(C, p0)
            d1 = np.dot(C, p1)
            d2 = np.dot(C, p2)
            # Check equidistance
            if abs(d0 - d1) < 1e-8 and abs(d0 - d2) < 1e-8:
                # This is a valid circumcenter
                R = math.acos(np.clip(d0, -1.0, 1.0))
                # Verify it's the correct one (should be on interior side)
                # For convex hull, the circumcenter should satisfy C·p0 > 0 
                # (circumradius < 90 degrees for good coverings)
                if d0 > 0.5:  # R < 60 degrees
                    max_rad = max(max_rad, math.degrees(R))
                break
        else:
            # Numerical issue - use approximate method
            # Sample the circumcenter by solving the linear system
            # C·(p0-p1) = 0, C·(p0-p2) = 0, |C| = 1
            # C is parallel to (p0-p1) x (p0-p2) = n (already have this)
            # Use the one with larger dot product with centroid
            centroid = (p0 + p1 + p2) / 3.0
            if np.dot(n, centroid) > 0:
                C = n
            else:
                C = -n
            d0 = np.dot(C, p0)
            R = math.acos(np.clip(d0, -1.0, 1.0))
            max_rad = max(max_rad, math.degrees(R))
    
    return max_rad


def _refine(points, deadline, tol=1e-10):
    """Refine points using adaptive-step hill climbing until deadline."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms < 1e-15):
        return points, 90.0
    points = points / norms
    
    best_points = points.copy()
    best_radius = _covering_radius_fast(best_points)
    step_size = 0.05
    
    while time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() >= deadline:
                break
            # Generate diverse directions
            for _ in range(30):
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
                
                new_point = points[i] + step_size * direction
                new_norm = np.linalg.norm(new_point)
                if new_norm < 1e-10:
                    continue
                new_point = new_point / new_norm
                
                test_points = points.copy()
                test_points[i] = new_point
                
                radius = _covering_radius_fast(test_points)
                if radius < best_radius - tol:
                    points = test_points
                    best_points = test_points.copy()
                    best_radius = radius
                    improved = True
                    # Increase step if improving
                    step_size = min(step_size * 1.5, 0.2)
                else:
                    # Decrease step if not improving
                    step_size = max(step_size * 0.7, 0.005)
        
        if not improved:
            # Try smaller steps
            step_size = max(step_size * 0.5, 0.002)
            if step_size < 0.003:
                break
    
    return best_points, best_radius


def _fibonacci_start(n=N):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(n=N, seed=None):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    best_points = None
    best_radius = 90.0
    
    # Multiple restarts with different starting configurations
    np.random.seed(42)
    
    # Start 1: Fibonacci spiral
    points = _fibonacci_start(N)
    refined, radius = _refine(points.copy(), deadline)
    if radius < best_radius:
        best_radius = radius
        best_points = refined.copy()
    
    # Start 2: Random
    if time.time() < deadline - 5.0:
        points = _random_start(N, seed=123)
        refined, radius = _refine(points.copy(), deadline)
        if radius < best_radius:
            best_radius = radius
            best_points = refined.copy()
    
    # Start 3: Another random with different seed
    if time.time() < deadline - 5.0:
        points = _random_start(N, seed=456)
        refined, radius = _refine(points.copy(), deadline)
        if radius < best_radius:
            best_radius = radius
            best_points = refined.copy()
    
    # Final polish if time remains
    if time.time() < deadline - 2.0 and best_points is not None:
        refined, radius = _refine(best_points.copy(), deadline, tol=1e-12)
        if radius < best_radius:
            best_radius = radius
            best_points = refined.copy()
    
    # Ensure valid output
    if best_points is None:
        best_points = _fibonacci_start(N)
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
