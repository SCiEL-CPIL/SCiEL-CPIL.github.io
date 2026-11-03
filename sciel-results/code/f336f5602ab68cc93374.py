"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _covering_radius(points):
    """Compute the covering radius in degrees for a set of unit vectors."""
    try:
        hull = ConvexHull(points)
        # For each facet, compute the circumradius
        # The facet is defined by its vertices
        max_circum = 0.0
        for simplex in hull.simplices:
            facet_pts = points[simplex]
            # Compute circumcenter of the facet
            # For a triangle, circumcenter is the intersection of perpendicular bisectors
            a, b, c = facet_pts
            # Using the formula for circumradius of a triangle
            # R = abc / (4 * Area)
            # But we need the angular distance from sphere center to circumcenter projected
            # Actually, the covering radius contribution from a facet is the angular distance
            # from the facet's circumcenter (on the sphere) to the farthest vertex
            
            # Compute the normal to the facet
            v1 = b - a
            v2 = c - a
            normal = np.cross(v1, v2)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-12:
                continue
            normal = normal / norm_n
            
            # The circumcenter of the triangle on the sphere is along the normal direction
            # Find the point on the sphere in the direction of the normal that is closest to all three vertices
            # Actually, for a spherical triangle, the covering radius is the maximum angular distance
            # from the circumcenter to the vertices
            
            # Compute the angular distances from the normal direction to each vertex
            # The circumcenter on the sphere is the point that minimizes the maximum distance
            # to the three vertices. For a triangle, this is the circumcenter.
            
            # Compute circumradius using the formula
            # For points on unit sphere, the circumradius R satisfies:
            # cos(R) = (cos(ab) + cos(bc) + cos(ca)) / (1 + 2*cos(ab)*cos(bc)*cos(ca))
            # Wait, that's not quite right. Let me use a different approach.
            
            # The circumcenter is the point x on the unit sphere such that x·a = x·b = x·c
            # This means x is in the null space of [a-b; b-c]
            # So x is parallel to the cross product (a-b) x (b-c)
            
            d1 = a - b
            d2 = b - c
            circum_dir = np.cross(d1, d2)
            circum_norm = np.linalg.norm(circum_dir)
            if circum_norm < 1e-12:
                continue
            circum_dir = circum_dir / circum_norm
            
            # Check if this is the correct direction (should be on the same side as the facet normal)
            # The circumcenter should be on the same side as the outward normal
            # For a convex hull, the outward normal points away from the center
            # The circumcenter should be in the direction of the normal if the triangle is acute
            # Let's just compute the angular distances
            
            # Angular distance from circum_dir to each vertex
            cos_a = np.dot(circum_dir, a)
            cos_b = np.dot(circum_dir, b)
            cos_c = np.dot(circum_dir, c)
            
            # Clamp to [-1, 1]
            cos_a = np.clip(cos_a, -1.0, 1.0)
            cos_b = np.clip(cos_b, -1.0, 1.0)
            cos_c = np.clip(cos_c, -1.0, 1.0)
            
            # The angular distance is arccos of the dot product
            # But we need the maximum of these three distances
            # The covering radius contribution is the maximum angular distance
            ang_a = np.arccos(cos_a)
            ang_b = np.arccos(cos_b)
            ang_c = np.arccos(cos_c)
            
            circum_rad = max(ang_a, ang_b, ang_c)
            max_circum = max(max_circum, circum_rad)
        
        return np.degrees(max_circum)
    except Exception:
        return 180.0


def _lloyd_relax(points, max_iter=200, tol=1e-10):
    """Lloyd relaxation: move each point to the centroid of its Voronoi cell."""
    pts = points.copy()
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # For each point, find its Voronoi cell centroid
        # This is complex, so use a simpler approach:
        # Move each point towards the centroid of its nearest neighbors' Voronoi region
        
        # Alternative: use the fact that the optimal covering has each point at the
        # centroid of its Voronoi cell. We can approximate this by moving each point
        # to the centroid of the points in its Voronoi cell.
        
        # Simpler approach: gradient-based optimization
        # Compute the covering radius and its gradient
        # This is complex, so let's use a different approach
        
        # Use a simple heuristic: for each point, compute the centroid of the
        # spherical Voronoi cell by averaging the facet circumcenters that contain it
        new_pts = pts.copy()
        for i in range(N):
            # Find all facets that contain point i
            cell_circumcenters = []
            for simplex in hull.simplices:
                if i in simplex:
                    facet_pts = pts[simplex]
                    a, b, c = facet_pts
                    d1 = a - b
                    d2 = b - c
                    circum_dir = np.cross(d1, d2)
                    circum_norm = np.linalg.norm(circum_dir)
                    if circum_norm > 1e-12:
                        circum_dir = circum_dir / circum_norm
                        cell_circumcenters.append(circum_dir)
            
            if len(cell_circumcenters) > 0:
                centroid = np.mean(cell_circumcenters, axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-12:
                    centroid = centroid / centroid_norm
                    # Move towards the centroid
                    new_pts[i] = pts[i] + 0.1 * (centroid - pts[i])
                    new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
        
        # Check convergence
        diff = np.linalg.norm(new_pts - pts)
        pts = new_pts
        if diff < tol:
            break
    
    return pts


def _optimize_local(points, max_iter=1000, tol=1e-12):
    """Local optimization using gradient descent on the covering radius."""
    pts = points.copy()
    
    # Use scipy's minimize with a smooth approximation
    def objective(x):
        pts_x = x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts_x, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts_x = pts_x / norms
        
        # Compute covering radius using a smooth approximation
        # Use the maximum of the circumradii of all facets
        try:
            hull = ConvexHull(pts_x)
            max_circum = 0.0
            for simplex in hull.simplices:
                facet_pts = pts_x[simplex]
                a, b, c = facet_pts
                d1 = a - b
                d2 = b - c
                circum_dir = np.cross(d1, d2)
                circum_norm = np.linalg.norm(circum_dir)
                if circum_norm < 1e-12:
                    continue
                circum_dir = circum_dir / circum_norm
                
                cos_a = np.clip(np.dot(circum_dir, a), -1.0, 1.0)
                cos_b = np.clip(np.dot(circum_dir, b), -1.0, 1.0)
                cos_c = np.clip(np.dot(circum_dir, c), -1.0, 1.0)
                
                ang_a = np.arccos(cos_a)
                ang_b = np.arccos(cos_b)
                ang_c = np.arccos(cos_c)
                
                circum_rad = max(ang_a, ang_b, ang_c)
                max_circum = max(max_circum, circum_rad)
            
            return max_circum
        except Exception:
            return 1.0  # Large penalty
    
    # Use Nelder-Mead for local optimization
    try:
        result = minimize(objective, pts.flatten(), method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol,
                                  'adaptive': True})
        if result.success or result.fun < objective(pts.flatten()):
            pts_new = result.x.reshape(N, 3)
            norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            pts = pts_new / norms
    except Exception:
        pass
    
    return pts


def _fibonacci_sphere(n, offset=0.5):
    """Generate points on the sphere using the Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _icosahedron_base():
    """Generate icosahedron vertices and subdivide to get 50 points."""
    # Icosahedron has 12 vertices, 20 faces
    # We need 50 points, so we can use the icosahedron and add more points
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Start with icosahedron and add random points, then optimize
    # For 50 points, we can use a mixture
    return verts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_pts = None
    best_radius = 180.0
    
    # Multiple initializations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_sphere(N, 0.5))
    
    # 2. Fibonacci spiral with different offset
    initializations.append(_fibonacci_sphere(N, 0.3))
    
    # 3. Random with fixed seed
    initializations.append(_random_sphere(N, seed=42))
    
    # 4. Random with another seed
    initializations.append(_random_sphere(N, seed=123))
    
    # 5. Icosahedron-based
    ico = _icosahedron_base()
    # Add more points to get to 50
    rng = np.random.RandomState(7)
    extra = rng.randn(N - 12, 3)
    extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
    init_ico = np.vstack([ico, extra])
    initializations.append(init_ico)
    
    for init in initializations:
        if time.time() > deadline:
            break
        
        # Local optimization
        try:
            optimized = _optimize_local(init, max_iter=500, tol=1e-10)
            radius = _covering_radius(optimized)
            if radius < best_radius:
                best_radius = radius
                best_pts = optimized
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # If we have time, do another round with the best point
    if best_pts is not None and time.time() < deadline - 20:
        try:
            refined = _optimize_local(best_pts, max_iter=1000, tol=1e-12)
            radius = _covering_radius(refined)
            if radius < best_radius:
                best_radius = radius
                best_pts = refined
        except Exception:
            pass
    
    # Final check
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N, 0.5)
    
    # Ensure all points are normalized
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
