"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _covering_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere."""
    try:
        hull = ConvexHull(pts, qhull_options='QJ')
    except Exception:
        return 180.0
    max_r = 0.0
    for f in hull.simplices:
        # Each facet is a triangle (3 points)
        a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
        # Centroid of the facet (for the circumcenter in 3D, we need the sphere through a,b,c)
        # The circumcenter of triangle abc on the unit sphere is the point on the sphere
        # equidistant from a, b, c. This is the intersection of the sphere with the line
        # through the origin perpendicular to the plane of abc... actually the circumcenter
        # of the spherical triangle is the point on the sphere that is the circumcenter.
        # For three points on the unit sphere, the circumcenter (if not antipodal) is
        # proportional to a cross product combination. Actually, the circumcenter of the
        # spherical triangle is the point p on the sphere such that |p-a|=|p-b|=|p-c|.
        # This means p·a = p·b = p·c. So p is perpendicular to (a-b) and (a-c).
        # So p is along the cross product (a-b) x (a-c), normalized.
        v1 = a - b
        v2 = a - c
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-15:
            # Degenerate
            continue
        n = n / nn
        # The circumcenter is either n or -n; we need the one that is "outside" the convex hull
        # Actually for a convex hull facet, the circumcenter of the spherical triangle 
        # (the point farthest from all three vertices on that spherical triangle's circumcircle)
        # is the point on the sphere in the direction of the outward normal of the facet.
        # The outward normal of the facet: the centroid of the facet is (a+b+c)/3, 
        # and the outward direction is from origin through this centroid.
        centroid = (a + b + c) / 3.0
        # The circumcenter should be in the direction of the centroid (outward)
        if np.dot(n, centroid) < 0:
            n = -n
        # The angular distance from the circumcenter to any of a,b,c is the circumradius
        # cos(angle) = n·a
        cos_ang = np.clip(n[0]*a[0] + n[1]*a[1] + n[2]*a[2], -1.0, 1.0)
        ang = np.arccos(cos_ang)
        if ang > max_r:
            max_r = ang
    return np.degrees(max_r)


def _refine(pts, time_budget):
    """Iteratively refine points to reduce covering radius using facet circumcenter attraction."""
    start = time.time()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Soft-min based gradient-free refinement
    # For each point, we want to move it toward the "worst" region it covers.
    # We use an iterative scheme: for each point, find the facet circumcenters it influences,
    # and move toward the worst one.
    
    lr = 0.02
    n_iter = 2000
    
    for it in range(n_iter):
        if time.time() - start > time_budget:
            break
        
        # Compute covering radius and identify worst facets
        try:
            hull = ConvexHull(pts, qhull_options='QJ')
        except Exception:
            break
        
        # For each point, compute a "pressure" from the facets it belongs to
        # We'll use a simple approach: for each facet, compute its circumcenter,
        # and push the three vertices toward it slightly.
        
        updates = np.zeros_like(pts)
        for f in hull.simplices:
            a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
            v1 = a - b
            v2 = a - c
            n = np.cross(v1, v2)
            nn = np.linalg.norm(n)
            if nn < 1e-15:
                continue
            n = n / nn
            centroid = (a + b + c) / 3.0
            if np.dot(n, centroid) < 0:
                n = -n
            # Move each vertex slightly toward the circumcenter
            for idx in f:
                p = pts[idx]
                # Direction from p to n
                d = n - p
                d_norm = np.linalg.norm(d)
                if d_norm > 1e-15:
                    updates[idx] += d / d_norm
        
        # Apply updates with learning rate
        pts = pts + lr * updates
        # Renormalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Check improvement
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    return best_pts, best_cr


def construct():
    t0 = time.time()
    deadline = 140.0  # Safety margin before 150s
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Refine
    if time.time() - t0 < deadline - 30:
        best_pts, best_cr = _refine(pts, deadline - time.time() - 20)
    
    # Final check
    if time.time() - t0 < deadline - 10:
        cr = _covering_radius(best_pts)
        if cr < best_cr:
            best_cr = cr
    
    return best_pts
# EVOLVE-BLOCK-END
