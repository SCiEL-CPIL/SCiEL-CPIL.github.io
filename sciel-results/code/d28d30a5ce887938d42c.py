"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import math

def _to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius (in radians) using convex hull.
    The covering radius is the maximum circumradius over all facets of the convex hull.
    """
    pts = _to_sphere(pts)
    
    # Check for coincident points
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-8:
                return np.pi  # invalid - coincident points
    
    # Check if all points are in a closed hemisphere
    # This is equivalent to checking if the origin is inside the convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # invalid hull
    
    # Check if origin is inside the convex hull
    # We can check this by seeing if the hull contains the origin
    # A simpler check: if the sum of outward normals weighted by area is zero-ish,
    # or we can use linear programming / scipy
    # Actually, for a convex hull of points on the unit sphere, the origin is inside
    # iff the points are NOT all in a closed hemisphere.
    # Let's check: if all points have positive dot product with some direction, they're in a hemisphere.
    # We can check if the origin is in the convex hull by solving a linear program,
    # but a simpler heuristic: check if the minimum of the maximum projection is positive.
    
    # Alternative: the covering radius computation via circumcircles of facets.
    # For each triangular facet of the convex hull, compute the circumcircle.
    # The circumradius of the triangle (on the sphere) gives the angular radius.
    # The covering radius is the maximum of these circumradii.
    
    # Get the triangular facets
    simplices = hull.simplices  # shape (M, 3) for 3D
    
    max_circumradius = 0.0
    
    for simplex in simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute the circumcenter of the spherical triangle
        # The circumcenter is the point equidistant (angularly) from a, b, c.
        # This is the normalized cross product of (b-a) and (c-a) ... no.
        # The circumcenter of three points on a sphere is along the line through
        # the center of the sphere perpendicular to the plane of the triangle,
        # but that's only if the triangle is acute in some sense.
        
        # Better: find the point p such that |p-a| = |p-b| = |p-c| and |p| = 1.
        # This means p is equidistant from a, b, c in Euclidean space, so p lies on
        # the line through the circumcenter of the planar triangle abc and the origin.
        
        # Compute the circumcenter of the planar triangle in 3D.
        # The circumcenter is the intersection of the perpendicular bisectors.
        
        # Vector approach:
        # p = alpha * a + beta * b + gamma * c where the circumcenter condition holds.
        # Actually, the circumcenter of triangle abc (as a planar figure) lies on the line
        # from the origin through the point equidistant from a, b, c.
        
        # Let's compute it directly.
        # The circumcenter C of triangle with vertices a, b, c satisfies:
        # |C - a|^2 = |C - b|^2 = |C - c|^2
        # This gives: 2 C . (b - a) = |b|^2 - |a|^2 = 0 (since on unit sphere)
        #            2 C . (c - a) = |c|^2 - |a|^2 = 0
        # So C is perpendicular to (b-a) and (c-a), meaning C is along the normal to the plane.
        # But C must also satisfy the circumradius condition.
        
        # Actually, since a, b, c are on the unit sphere, the circumcenter of the planar triangle
        # is at distance d from the origin, and the circumradius R of the planar triangle satisfies
        # R^2 + d^2 = 1 (if the circumcenter is on the same side).
        # Wait, that's not quite right either.
        
        # Let me use a direct formula.
        # The circumcenter of three points on the unit sphere:
        # n = (b - a) × (c - a)  -- normal to the plane
        # The circumcenter is at position t * n_normalized where t is chosen so that
        # the distance from the circumcenter to a equals the distance from the circumcenter to the origin's projection... 
        
        # Let me just solve it numerically for each facet.
        # Find p on unit sphere minimizing max(|p-a|, |p-b|, |p-c|)
        # This is equivalent to finding the point equidistant from a, b, c on the sphere.
        
        # The equidistant point satisfies:
        # (p - a) . (p - a) = (p - b) . (p - b)
        # => 2 p . (b - a) = |b|^2 - |a|^2 = 0
        # Similarly 2 p . (c - a) = 0
        # So p is perpendicular to (b-a) and (c-a), i.e., p is along ±n where n = (b-a) × (c-a).
        
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            # Degenerate triangle
            max_circumradius = max(max_circumradius, np.pi)
            continue
        n = n / n_norm
        
        # The circumcenter is at ±n. We need to check which one is the correct circumcenter
        # (the one that gives the smaller circumradius, or the one inside the spherical triangle).
        # The covering radius for the facet is the angular distance from the circumcenter to a, b, or c.
        
        # Try both signs
        best_radius = np.pi
        for sign in [1.0, -1.0]:
            p = sign * n
            # Check if p is on the correct side (inside the spherical triangle)
            # The circumcenter should be such that the angular distances to a, b, c are equal
            # and less than pi/2 for the "small" circumcircle.
            
            da = np.arccos(np.clip(np.dot(p, a), -1, 1))
            db = np.arccos(np.clip(np.dot(p, b), -1, 1))
            dc = np.arccos(np.clip(np.dot(p, c), -1, 1))
            
            # Check if distances are approximately equal
            if abs(da - db) < 1e-6 and abs(da - dc) < 1e-6:
                # This is a valid circumcenter
                if da < best_radius:
                    best_radius = da
        
        max_circumradius = max(max_circumradius, best_radius)
    
    return max_circumradius

def _covering_radius_fast(pts):
    """
    Faster covering radius computation.
    Uses the fact that the covering radius is the maximum over all points on the sphere
    of the distance to the nearest point in the configuration.
    This is equivalent to the maximum circumradius of the Delaunay triangulation on the sphere,
    which corresponds to the convex hull facets.
    """
    pts = _to_sphere(pts)
    
    # Check for coincident points
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-8:
                return np.pi
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    
    simplices = hull.simplices
    
    max_circumradius = 0.0
    
    for simplex in simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            max_circumradius = max(max_circumradius, np.pi)
            continue
        n = n / n_norm
        
        best_radius = np.pi
        for sign in [1.0, -1.0]:
            p = sign * n
            da = np.arccos(np.clip(np.dot(p, a), -1, 1))
            db = np.arccos(np.clip(np.dot(p, b), -1, 1))
            dc = np.arccos(np.clip(np.dot(p, c), -1, 1))
            
            if abs(da - db) < 1e-5 and abs(da - dc) < 1e-5:
                if da < best_radius:
                    best_radius = da
        
        max_circumradius = max(max_circumradius, best_radius)
    
    return max_circumradius

def _objective(pts):
    """Objective function: covering radius in radians."""
    return _covering_radius_fast(pts)

def _fibonacci_spiral(n, offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n):
    """Generate n random points on the sphere."""
    pts = np.random.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _local_optimize(pts, max_iter=200, step_size=0.1):
    """
    Local optimization using gradient-free method.
    We use a simple perturb-and-evaluate approach.
    """
    best_pts = pts.copy()
    best_val = _covering_radius_fast(best_pts)
    
    for it in range(max_iter):
        # Perturb a random point
        idx = np.random.randint(len(best_pts))
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_pts = best_pts.copy()
        new_pts[idx] = best_pts[idx] + step_size * direction
        new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        new_val = _covering_radius_fast(new_pts)
        
        if new_val < best_val:
            best_pts = new_pts
            best_val = new_val
            step_size *= 1.05  # slightly increase step size
        else:
            step_size *= 0.9  # decrease step size
    
    return best_pts, best_val

def _simulated_annealing(pts, max_iter=500, T0=0.5, T_end=0.01):
    """
    Simulated annealing for covering radius minimization.
    """
    current_pts = pts.copy()
    current_val = _covering_radius_fast(current_pts)
    best_pts = current_pts.copy()
    best_val = current_val
    
    T = T0
    cooling = (T_end / T0) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        T *= cooling
        
        # Perturb a random point
        idx = np.random.randint(len(current_pts))
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_pts = current_pts.copy()
        new_pts[idx] = current_pts[idx] + 0.1 * direction
        new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        new_val = _covering_radius_fast(new_pts)
        
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_pts = new_pts
            current_val = new_val
            if current_val < best_val:
                best_pts = current_pts.copy()
                best_val = current_val
    
    return best_pts, best_val

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses multiple restarts with local optimization and simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_pts = None
    best_val = np.pi
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    val = _covering_radius_fast(pts)
    if val < best_val:
        best_val = val
        best_pts = pts.copy()
    
    # Local optimization from Fibonacci spiral
    opt_pts, opt_val = _local_optimize(pts, max_iter=100, step_size=0.05)
    if opt_val < best_val:
        best_val = opt_val
        best_pts = opt_pts.copy()
    
    # Try simulated annealing
    sa_pts, sa_val = _simulated_annealing(best_pts, max_iter=300, T0=0.3, T_end=0.01)
    if sa_val < best_val:
        best_val = sa_val
        best_pts = sa_pts.copy()
    
    # Multiple random restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random start
        pts_r = _random_points(N)
        val_r = _covering_radius_fast(pts_r)
        
        # Local optimization
        opt_pts_r, opt_val_r = _local_optimize(pts_r, max_iter=50, step_size=0.05)
        
        # Simulated annealing
        sa_pts_r, sa_val_r = _simulated_annealing(opt_pts_r, max_iter=200, T0=0.2, T_end=0.01)
        
        if sa_val_r < best_val:
            best_val = sa_val_r
            best_pts = sa_pts_r.copy()
    
    # Final local optimization
    if time.time() < deadline - 5.0:
        final_pts, final_val = _local_optimize(best_pts, max_iter=100, step_size=0.03)
        if final_val < best_val:
            best_val = final_val
            best_pts = final_pts.copy()
    
    return _to_sphere(best_pts)
# EVOLVE-BLOCK-END
