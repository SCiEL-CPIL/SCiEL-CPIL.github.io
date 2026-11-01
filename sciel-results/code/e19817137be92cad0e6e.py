"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import warnings
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50
DEADLINE = 120.0
SAFETY_MARGIN = 5.0  # seconds to spare

# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius in radians using convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # invalid
    
    facets = hull.simplices
    # For each facet (triangle), compute the circumcenter and circumradius
    max_r = 0.0
    for simplex in facets:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Circumcenter of triangle on unit sphere
        # Using the formula: circumcenter is along the normal to the plane
        # For points on unit sphere, circumradius = |a-b| / (2*sin(angle at circumcenter))
        # Actually, for a spherical triangle, the circumradius is the angular distance
        # from the circumcenter to each vertex.
        
        # Compute the normal to the triangle
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        # The circumcenter of the spherical triangle is in the direction of the normal
        # (for small triangles) or we need to compute it properly.
        
        # For a triangle on the unit sphere, the circumcenter can be found as:
        # It's the point on the sphere equidistant from all three vertices.
        # This is the intersection of the sphere with the plane through the origin
        # perpendicular to... actually it's more complex.
        
        # Simpler approach: the circumradius is the maximum angular distance
        # from any point in the facet to the nearest vertex.
        # The worst point is the circumcenter of the spherical triangle.
        
        # For a planar triangle projected onto the sphere, we can compute the
        # circumcenter in 3D and then project to the sphere.
        
        # 3D circumcenter of triangle abc:
        d = 2 * (a * np.dot(b, c) + b * np.dot(c, a) + c * np.dot(a, b)
                 - a * np.dot(b, b) - b * np.dot(c, c) - c * np.dot(a, a))
        # This formula is for the circumcenter in the plane. Let's use a different approach.
        
        # Use the fact that for points on the unit sphere, the circumradius
        # of the spherical triangle equals the angle at the center subtended by
        # the circumcenter-to-vertex distance.
        
        # Compute the circumcenter in the plane of the triangle:
        # Let u = b - a, v = c - a
        # Circumcenter = a + alpha*u + beta*v where:
        # 2*u.dot(alpha*u + beta*v) = |u|^2
        # 2*v.dot(alpha*u + beta*v) = |v|^2
        
        u = b - a
        v = c - a
        uu = np.dot(u, u)
        vv = np.dot(v, v)
        uv = np.dot(u, v)
        
        det = uu * vv - uv * uv
        if abs(det) < 1e-12:
            continue
        
        alpha = (vv * uu - uv * vv) / (2 * det)
        beta = (uu * vv - uv * uu) / (2 * det)
        
        circum_3d = a + alpha * u + beta * v
        circum_norm = np.linalg.norm(circum_3d)
        if circum_norm < 1e-12:
            continue
        circum_sphere = circum_3d / circum_norm
        
        # The circumradius is the angular distance from circumcenter to any vertex
        cos_angle = np.clip(np.dot(circum_sphere, a), -1.0, 1.0)
        r = np.arccos(cos_angle)
        max_r = max(max_r, r)
    
    return max_r


def _soft_max_covering_radius(pts, beta=100.0):
    """Smooth surrogate for the covering radius using log-sum-exp."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 10.0
    
    facets = hull.simplices
    radii = []
    for simplex in facets:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        u = b - a
        v = c - a
        uu = np.dot(u, u)
        vv = np.dot(v, v)
        uv = np.dot(u, v)
        
        det = uu * vv - uv * uv
        if abs(det) < 1e-12:
            continue
        
        alpha = (vv * uu - uv * vv) / (2 * det)
        beta_coeff = (uu * vv - uv * uu) / (2 * det)
        
        circum_3d = a + alpha * u + beta_coeff * v
        circum_norm = np.linalg.norm(circum_3d)
        if circum_norm < 1e-12:
            continue
        circum_sphere = circum_3d / circum_norm
        
        cos_angle = np.clip(np.dot(circum_sphere, a), -1.0, 1.0)
        r = np.arccos(cos_angle)
        radii.append(r)
    
    if not radii:
        return 10.0
    
    radii = np.array(radii)
    # Soft max
    r_max = np.max(radii)
    if r_max < 1e-10:
        return 0.0
    return r_max + np.log(np.sum(np.exp(beta * (radii - r_max)))) / beta


def _lloyd_iteration(pts, max_iter=100, tol=1e-8):
    """Lloyd iteration: move points toward worst facet circumcenters."""
    pts = _normalize(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        facets = hull.simplices
        # Compute circumcenters for all facets
        circumcenters = []
        radii = []
        for simplex in facets:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            u = b - a
            v = c - a
            uu = np.dot(u, u)
            vv = np.dot(v, v)
            uv = np.dot(u, v)
            
            det = uu * vv - uv * uv
            if abs(det) < 1e-12:
                continue
            
            alpha = (vv * uu - uv * vv) / (2 * det)
            beta_coeff = (uu * vv - uv * uu) / (2 * det)
            
            circum_3d = a + alpha * u + beta_coeff * v
            circum_norm = np.linalg.norm(circum_3d)
            if circum_norm < 1e-12:
                continue
            circum_sphere = circum_3d / circum_norm
            
            cos_angle = np.clip(np.dot(circum_sphere, a), -1.0, 1.0)
            r = np.arccos(cos_angle)
            circumcenters.append(circum_sphere)
            radii.append(r)
        
        if not circumcenters:
            break
        
        circumcenters = np.array(circumcenters)
        radii = np.array(radii)
        
        # Assign each point to its nearest circumcenter (Voronoi-like)
        # Actually, in Lloyd's algorithm for sphere covering, we move each point
        # toward the centroid of its Voronoi cell. But a simpler approach:
        # For each point, find the facet with the largest circumradius that
        # "belongs" to it, and move toward that circumcenter.
        
        # Alternative: compute the average of circumcenters weighted by radius
        # for each point based on which facets it's part of.
        
        # Simpler: for each point, compute the weighted average of all circumcenters
        # weighted by 1/distance^2, then move toward that.
        
        new_pts = np.zeros_like(pts)
        for i in range(N):
            # Weight by inverse squared distance to each circumcenter
            dists = np.linalg.norm(circumcenters - pts[i], axis=1)
            dists = np.maximum(dists, 1e-8)
            weights = radii / (dists ** 2)
            weights = weights / np.sum(weights)
            new_pts[i] = np.sum(weights[:, None] * circumcenters, axis=0)
        
        new_pts = _normalize(new_pts)
        
        # Check convergence
        diff = np.linalg.norm(new_pts - pts)
        pts = new_pts
        
        if diff < tol:
            break
    
    return pts


def _polish(pts, max_iter=500, tol=1e-10):
    """High-precision polish using gradient descent on soft-max covering radius."""
    pts = _normalize(pts)
    
    def objective(x):
        x = x.reshape(N, 3)
        return _soft_max_covering_radius(x, beta=500.0)
    
    # Use BFGS for local optimization
    result = minimize(objective, pts.flatten(), method='BFGS', 
                      options={'maxiter': max_iter, 'gtol': tol})
    
    if result.success:
        return _normalize(result.x.reshape(N, 3))
    else:
        return pts


def _generate_initial(seed=0):
    """Generate initial configuration using Fibonacci spiral with random perturbation."""
    rng = np.random.RandomState(seed)
    
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add small random perturbation
    pts = pts + rng.normal(0, 0.01, size=(N, 3))
    pts = _normalize(pts)
    
    return pts


def construct():
    start_time = time.time()
    
    best_pts = None
    best_radius = np.pi
    
    # Try multiple initial configurations with different seeds
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - SAFETY_MARGIN:
            break
        
        pts = _generate_initial(seed)
        
        # Lloyd iteration
        pts = _lloyd_iteration(pts, max_iter=50)
        r = _covering_radius(pts)
        
        if r < best_radius:
            best_radius = r
            best_pts = pts.copy()
        
        # Polish if time permits
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - SAFETY_MARGIN - 20:
            pts_polished = _polish(pts, max_iter=200, tol=1e-9)
            r_polished = _covering_radius(pts_polished)
            
            if r_polished < best_radius:
                best_radius = r_polished
                best_pts = pts_polished.copy()
    
    # Final polish on the best result if time permits
    elapsed = time.time() - start_time
    if best_pts is not None and elapsed < DEADLINE - SAFETY_MARGIN - 10:
        final_pts = _polish(best_pts, max_iter=300, tol=1e-10)
        r_final = _covering_radius(final_pts)
        
        if r_final < best_radius:
            best_radius = r_final
            best_pts = final_pts.copy()
    
    if best_pts is None:
        # Fallback: Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
