"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _spherical_distance(a, b):
    dot = np.dot(a, b)
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)


def _covering_radius(points):
    """Compute the covering radius in radians.
    The covering radius is the maximum over all facets of the convex hull
    of the circumradius of that facet (on the sphere).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: if hull fails, use a grid-based estimate
        # This should rarely happen
        theta = np.linspace(0, np.pi, 100)
        phi = np.linspace(0, 2 * np.pi, 200)
        T, P = np.meshgrid(theta, phi)
        grid = np.column_stack([
            np.sin(T) * np.cos(P),
            np.sin(T) * np.sin(P),
            np.cos(T)
        ]).reshape(-1, 3)
        max_dist = 0.0
        for g in grid:
            dists = np.arccos(np.clip(points @ g, -1, 1))
            max_dist = max(max_dist, np.min(dists))
        return max_dist

    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = points[simplex]
        # Compute circumradius of spherical triangle
        # The circumcenter is the point equidistant from a, b, c on the sphere
        # For a spherical triangle, the circumradius R satisfies:
        # cos(R) = (cos(ab)cos(bc)cos(ca) + 2cos(ab)cos(bc)cos(ca)) / (1 - cos^2(ab) - cos^2(bc) - cos^2(ca) + 2cos(ab)cos(bc)cos(ca))
        # But a simpler approach: find the normal to the plane through a,b,c,
        # then the circumcenter is along that normal (or its negative) at the
        # correct distance.
        
        # Plane normal
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        
        # The circumcenter lies on the line through the origin in direction n
        # (or -n). We need to find which sign and what distance.
        # The circumcenter O satisfies |O - a| = |O - b| = |O - c| = R
        # and |O| = 1 (on the sphere).
        # Let O = t * n. Then |t*n - a|^2 = R^2
        # t^2 - 2t(n·a) + |a|^2 = R^2
        # t^2 - 2t(n·a) + 1 = R^2
        # Similarly for b and c.
        # Subtracting: 2t(n·a - n·b) = 0 => t(n·a - n·b) = 0
        # This means either t=0 (degenerate) or n·a = n·b = n·c.
        # For a triangle not containing the origin in its convex hull, n·a = n·b = n·c = d
        # Then t^2 - 2td + 1 = R^2
        # The circumradius is the angle between O and a: cos(R) = O·a = t*n·a = t*d
        # Also |O - a|^2 = R^2 => 1 - 2td + 1 = R^2 => R^2 = 2 - 2td
        # And cos(R) = td, so R^2 = 2 - 2cos(R) => sin^2(R) = 1 - cos^2(R) = 1 - (td)^2
        # 2 - 2td = 1 - t^2d^2 => t^2d^2 - 2td + 1 = 0 => (td - 1)^2 = 0 => td = 1
        # Wait, that gives cos(R) = 1, R = 0. That's wrong.
        
        # Let me reconsider. The circumcenter on the sphere is the point O on the unit sphere
        # such that the angular distance from O to a, b, c is equal.
        # O = t*n where |t*n| = 1 => t = ±1.
        # So O is either n or -n.
        # The angular distance from n to a is arccos(n·a).
        # We need arccos(n·a) = arccos(n·b) = arccos(n·c).
        # This means n·a = n·b = n·c (since arccos is monotonic).
        # If n·a = n·b = n·c = d, then the circumradius is arccos(d) if d > 0,
        # or arccos(d) which is > pi/2 if d < 0. But the circumcenter should be
        # on the same side as the triangle, so we pick the sign that gives
        # the smaller angle.
        
        da = np.dot(n, a)
        db = np.dot(n, b)
        dc = np.dot(n, c)
        
        # Check if they're approximately equal
        if abs(da - db) > 1e-6 or abs(db - dc) > 1e-6:
            # The plane doesn't pass through the origin symmetrically; 
            # use the other sign
            da = -da
            db = -db
            dc = -dc
            if abs(da - db) > 1e-6 or abs(db - dc) > 1e-6:
                # Degenerate or numerical issue; skip
                continue
        
        d = (da + db + dc) / 3.0
        d = np.clip(d, -1.0, 1.0)
        r = np.arccos(d)
        # The circumradius should be the smaller angle
        if r > np.pi / 2:
            r = np.pi - r  # This shouldn't happen if we picked the right sign
        max_r = max(max_r, r)
    
    return max_r


def _objective_and_grad(x, points_flat, N):
    """Objective: covering radius. x is the flattened 3N vector of points."""
    pts = x.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    pts = pts / norms
    
    r = _covering_radius(pts)
    return r


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # 115s deadline, 5s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_r = _covering_radius(points)
    best_points = points.copy()
    
    # Iterative refinement: repeatedly find worst facet and adjust its vertices
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Find the facet with the largest circumradius
        worst_idx = -1
        worst_r = 0.0
        worst_simplex = None
        for simplex in hull.simplices:
            a, b, c = points[simplex]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            da = np.dot(n, a)
            db = np.dot(n, b)
            dc = np.dot(n, c)
            if abs(da - db) > 1e-6 or abs(db - dc) > 1e-6:
                da, db, dc = -da, -db, -dc
                if abs(da - db) > 1e-6 or abs(db - dc) > 1e-6:
                    continue
            d = (da + db + dc) / 3.0
            d = np.clip(d, -1.0, 1.0)
            r = np.arccos(d)
            if r > worst_r:
                worst_r = r
                worst_idx = len(hull.simplices)  # not used directly
                worst_simplex = simplex
        
        if worst_simplex is None:
            break
        
        a_idx, b_idx, c_idx = worst_simplex
        a, b, c = points[a_idx], points[b_idx], points[c_idx]
        
        # Compute the circumcenter of this triangle
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            break
        n = n / nn
        da = np.dot(n, a)
        if abs(da - np.dot(n, b)) > 1e-6:
            n = -n
            da = -da
        
        circumcenter = n  # on the unit sphere
        
        # Move each vertex slightly toward the circumcenter to reduce the circumradius
        # Use a small step size
        step = 0.05
        for idx in [a_idx, b_idx, c_idx]:
            p = points[idx]
            # Move toward circumcenter on the sphere
            # Geodesic: p -> cos(t)*p + sin(t)*component
            # Simpler: linear interpolation in 3D then normalize
            new_p = (1 - step) * p + step * circumcenter
            new_p = _normalize(new_p)
            points[idx] = new_p
        
        # Recompute covering radius
        current_r = _covering_radius(points)
        if current_r < best_r:
            best_r = current_r
            best_points = points.copy()
        else:
            # Revert if it got worse
            points = best_points.copy()
            # Try smaller step
            step *= 0.5
            if step < 1e-6:
                break
    
    # Final polish with scipy minimize on the worst facet
    if time.time() < deadline - 5.0:
        # Get the current best points
        points = best_points.copy()
        
        for polish_iter in range(20):
            if time.time() > deadline - 2.0:
                break
            
            try:
                hull = ConvexHull(points)
            except Exception:
                break
            
            worst_r = 0.0
            worst_simplex = None
            for simplex in hull.simplices:
                a, b, c = points[simplex]
                n = np.cross(b - a, c - a)
                nn = np.linalg.norm(n)
                if nn < 1e-12:
                    continue
                n = n / nn
                da = np.dot(n, a)
                db = np.dot(n, b)
                dc = np.dot(n, c)
                if abs(da - db) > 1e-6 or abs(db - dc) > 1e-6:
                    da, db, dc = -da, -db, -dc
                    if abs(da - db) > 1e-6 or abs(db - dc) > 1e-6:
                        continue
                d = (da + db + dc) / 3.0
                d = np.clip(d, -1.0, 1.0)
                r = np.arccos(d)
                if r > worst_r:
                    worst_r = r
                    worst_simplex = simplex
            
            if worst_simplex is None:
                break
            
            a_idx, b_idx, c_idx = worst_simplex
            a, b, c = points[a_idx], points[b_idx], points[c_idx]
            
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                break
            n = n / nn
            if abs(np.dot(n, a) - np.dot(n, b)) > 1e-6:
                n = -n
            circumcenter = n
            
            # Try to optimize the three vertices
            def obj(x):
                pts = points.copy()
                pts[a_idx] = _normalize(x[0:3])
                pts[b_idx] = _normalize(x[3:6])
                pts[c_idx] = _normalize(x[6:9])
                return _covering_radius(pts)
            
            x0 = np.concatenate([a, b, c])
            res = minimize(obj, x0, method='Nelder-Mead', 
                          options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.success and res.fun < best_r:
                points[a_idx] = _normalize(res.x[0:3])
                points[b_idx] = _normalize(res.x[3:6])
                points[c_idx] = _normalize(res.x[6:9])
                best_r = res.fun
                best_points = points.copy()
            else:
                break
    
    # Ensure all points are normalized
    best_points = np.array([_normalize(p) for p in best_points])
    
    return best_points
