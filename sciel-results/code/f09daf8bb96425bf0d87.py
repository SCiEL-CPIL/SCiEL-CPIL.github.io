"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START

def _covering_radius(points):
    """Compute the covering radius (in radians) of points on the unit sphere."""
    try:
        hull = ConvexHull(points)
        # The covering radius is the maximum circumradius of the facets
        # that face away from the origin (i.e., the origin is on the inside)
        # Actually, for a set of points on the unit sphere, the covering radius
        # is the maximum over all facets of the circumradius of the spherical triangle
        # (or polygon) formed by the facet vertices, but only for facets where
        # the origin is inside the convex hull.
        
        # More precisely: for each facet (triangle), compute the circumcenter
        # on the sphere. The angular distance from the circumcenter to each vertex
        # is the circumradius. We want the max over all facets.
        
        # A facet is a triangle with vertices p1, p2, p3.
        # The circumcenter on the unit sphere is the normalized cross product approach,
        # or we can compute it as follows:
        # The circumcenter c satisfies |c - pi|^2 = R^2 for i=1,2,3 and |c|=1.
        # This gives: 2 c·pi = 1 + R^2 for all i.
        # So c·p1 = c·p2 = c·p3 = (1+R^2)/2.
        
        # Let's use the formula: for a spherical triangle with vertices on unit sphere,
        # the circumradius R satisfies:
        # cos(R) = (a*b*c + 2*sqrt(a*b*c) ... ) -- this is complex.
        
        # Simpler: solve the linear system.
        # c·p1 = c·p2 = c·p3 = t
        # |c|^2 = 1
        # So (p1 - p2)·c = 0, (p1 - p3)·c = 0, and c·p1 = t
        # c is perpendicular to (p1-p2) and (p1-p3), so c is parallel to (p1-p2) x (p1-p3)
        
        max_r = 0.0
        for simplex in hull.simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            
            # Compute circumcenter on sphere
            # c is parallel to (p2-p1) x (p3-p1)
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            cnorm = np.linalg.norm(cross)
            if cnorm < 1e-12:
                continue
            c = cross / cnorm
            
            # Ensure c is on the correct side (away from origin relative to facet)
            # The facet normal should point outward. The origin is inside the hull,
            # so the facet's circumcenter on the sphere should be the point on the
            # great circle arc that is farthest from all three vertices.
            # We need to check: is the origin on the same side as c?
            # The facet plane has normal n = cross(v1, v2). The origin is inside,
            # so n·origin < 0 means n points outward... 
            # Actually, let's just compute the distance and take the appropriate one.
            
            # The circumradius is the angular distance from c to p1 (or p2 or p3).
            # But c might be on the "wrong" side. If c·p1 < 0, the circumcenter
            # is actually -c, and the radius is pi - arccos(c·p1).
            dot1 = np.clip(np.dot(c, p1), -1.0, 1.0)
            r = np.arccos(dot1)
            
            # Check if -c gives a smaller radius (i.e., c is on the wrong side)
            dot1_neg = np.clip(-dot1, -1.0, 1.0)
            r_neg = np.arccos(dot1_neg)
            
            # The covering radius for this facet is the minimum of r and r_neg
            # because the circumcenter is the point on the great circle equidistant
            # from all three vertices, and it could be on either side.
            r_facet = min(r, r_neg)
            
            # But wait, we need the point on the sphere farthest from the three vertices.
            # That's the circumcenter. The angular distance from the circumcenter to
            # each vertex is the circumradius. We want the max over all facets.
            # The circumcenter is unique up to sign, and we want the one that is
            # "outside" the triangle, i.e., the one that is farthest from the centroid.
            
            # Let's use the correct formula: the circumradius of a spherical triangle.
            # For vertices with pairwise angular distances a, b, c:
            # cos(R) = (cos(a)cos(b)cos(c) + 2*cos(a)*cos(b)*cos(c)) / (2*sqrt(1-cos^2(a))...)
            # This is getting complicated. Let's stick with the linear algebra approach.
            
            # The point c (normalized) is equidistant from p1, p2, p3.
            # The angular distance from c to p1 is arccos(c·p1).
            # If c·p1 > 0, the distance is arccos(c·p1).
            # If c·p1 < 0, the distance is arccos(c·p1) which is > pi/2, but the
            # actual circumcenter on the sphere should be the one closer to the triangle.
            # So we take the minimum distance to the three vertices.
            
            max_r = max(max_r, r_facet)
        
        return max_r
    except Exception:
        return np.pi  # Fallback: return maximum possible


def _objective(params):
    """Objective function: covering radius of the configuration."""
    points = params.reshape(-1, 3)
    return _covering_radius(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # Local optimization using L-BFGS-B with smooth approximation
    # We'll use a soft-max approximation of the max circumradius
    
    def _smooth_objective(params, beta=50.0):
        """Smooth approximation of the covering radius using log-sum-exp."""
        points = params.reshape(-1, 3)
        # Renormalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        points = points / norms
        
        try:
            hull = ConvexHull(points)
        except Exception:
            return 1.0  # Large penalty for invalid hull
        
        radii = []
        for simplex in hull.simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            cnorm = np.linalg.norm(cross)
            if cnorm < 1e-12:
                continue
            c = cross / cnorm
            
            dot1 = np.clip(np.dot(c, p1), -1.0, 1.0)
            r = np.arccos(dot1)
            r_neg = np.arccos(np.clip(-dot1, -1.0, 1.0))
            r_facet = min(r, r_neg)
            radii.append(r_facet)
        
        if not radii:
            return 1.0
        
        radii = np.array(radii)
        # Soft max
        max_r = np.max(radii)
        log_sum_exp = max_r + np.log(np.sum(np.exp(radii - max_r)))
        return log_sum_exp / beta
    
    # Optimize
    best_params = points.flatten().copy()
    
    for restart in range(5):
        if time.time() > deadline - 30.0:
            break
        
        # Add small perturbation for restarts
        if restart > 0:
            noise = np.random.randn(N, 3) * 0.01
            params = best_points.flatten() + noise
        else:
            params = best_params.copy()
        
        # Normalize after perturbation
        pts = params.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        pts = pts / norms
        params = pts.flatten()
        
        try:
            result = minimize(
                _smooth_objective,
                params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < _smooth_objective(params):
                new_points = result.x.reshape(-1, 3)
                norms = np.linalg.norm(new_points, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                new_points = new_points / norms
                
                new_radius = _covering_radius(new_points)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = new_points
                    best_params = result.x
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _smooth_objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < _smooth_objective(best_params):
                new_points = result.x.reshape(-1, 3)
                norms = np.linalg.norm(new_points, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                new_points = new_points / norms
                
                new_radius = _covering_radius(new_points)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = new_points
        except Exception:
            pass
    
    # Ensure points are on unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
