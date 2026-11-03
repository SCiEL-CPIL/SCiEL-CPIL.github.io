"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import random

def _covering_radius_deg(pts):
    """Compute covering radius in degrees. Returns inf if invalid."""
    if pts.shape != (N, 3):
        return np.inf
    if not np.all(np.isfinite(pts)):
        return np.inf
    # Normalize
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms < 1e-10):
        return np.inf
    pts = pts / norms[:, np.newaxis]
    
    # Check not in a closed hemisphere: origin should be inside convex hull
    try:
        hull = ConvexHull(pts)
        # Check if origin is inside the hull
        # A point is inside a convex hull if for all facets, the dot product
        # with the outward normal is <= 0 (origin side)
        # Facets are triangles; compute normals
        facets = hull.simplices
        normals = []
        for tri in facets:
            v1 = pts[tri[0]]
            v2 = pts[tri[1]]
            v3 = pts[tri[2]]
            n = np.cross(v2 - v1, v3 - v1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            # Ensure outward: origin should be on the negative side
            # If dot(n, v1) > 0, flip
            if np.dot(n, v1) > 0:
                n = -n
            normals.append(n)
        if len(normals) == 0:
            return np.inf
        normals = np.array(normals)
        # Origin inside if all dot(normal, origin) <= 0, i.e., all normals point away
        # Actually, for a convex polytope containing origin, for each facet with outward normal n,
        # we need n · x <= 0 for all x in polytope, specifically n · 0 = 0 <= 0 is trivially true.
        # Better check: the covering radius is the max over facets of the angular radius of the spherical cap.
        # For a facet with vertices v1, v2, v3, the circumcenter on the sphere is the direction of the plane normal.
        # The covering radius contribution is the angle between the facet center and the vertices.
        
        # Let's compute the max angular distance from any point on sphere to nearest point.
        # This is equivalent to: for each facet of the convex hull, the farthest point on that facet's
        # spherical cap is the facet center (circumcenter of the triangle on the sphere).
        # The angular distance is the angle between the facet center and any vertex.
        
        max_r = 0.0
        for tri in facets:
            v1, v2, v3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            # Compute circumcenter of the spherical triangle
            # This is the normalized cross product of (v2-v1) and (v3-v1)? No.
            # The circumcenter of a spherical triangle is in the direction of the vector that is equidistant from all three.
            # For a plane triangle, circumcenter C satisfies |C-v1|=|C-v2|=|C-v3|.
            # On the sphere, we can find the point that maximizes the minimum distance to the three vertices,
            # but actually for the covering radius, we need the center of the largest empty circle.
            # For a facet of the convex hull, the covering radius is determined by the circumradius of the triangle.
            
            # Compute the circumradius of the triangle on the sphere.
            # The circumcenter direction is proportional to the cross product of the edge vectors? 
            # Let's use the formula: the circumcenter of a spherical triangle with vertices a, b, c
            # is in the direction of a * (b × c) + b * (c × a) + c * (a × b) ? 
            # Actually, a simpler approach: the circumcenter is the intersection of the perpendicular bisectors.
            # In 3D, for points on the unit sphere, the circumcenter direction is:
            # n = (b × c) * |b × c| + (c × a) * |c × a| + (a × b) * |a × b| ?
            # Let me use a different approach: minimize the maximum distance to the three points.
            # But for speed, let's compute the circumradius directly.
            
            # The circumradius R of a spherical triangle satisfies:
            # sin(R) = |a × b| / (2 * |a × b|) ... no.
            # Let's compute the circumcenter as the normalized vector:
            # c = a * sin(B) + b * sin(A) + c * sin(C) where A, B, C are angles?
            
            # Simpler: use the fact that the circumcenter lies in the plane of the triangle.
            # The plane is spanned by a, b, c. The circumcenter in the plane is:
            # Let's find the point x = αa + βb + γc such that |x-a| = |x-b| = |x-c| and x is on the unit sphere.
            # This is complex. Let's use a numerical approach for the circumcenter.
            
            # Actually, for the covering radius, we can just compute the maximum over the facet of the
            # distance to the nearest point. The farthest point on the spherical triangle from all three vertices
            # is either the circumcenter (if it lies inside the triangle) or one of the vertices.
            
            # For efficiency, let's compute the circumcenter direction as:
            # n = cross(b - a, c - a)  # normal to the plane
            # Then the circumcenter is in the plane, so it's a linear combination of a, b, c.
            
            # Let's just compute the maximum distance to the three vertices for a set of candidate points.
            # But this is getting complex. Let's use a simpler method:
            # The covering radius is the maximum over all points on the sphere of the distance to the nearest point.
            # This can be computed by finding the Voronoi diagram on the sphere, but that's complex.
            
            # Alternative: for each facet, the covering radius contribution is the circumradius of the triangle.
            # The circumradius of a spherical triangle with side lengths a, b, c (in radians) is:
            # R = arccos((cos(a)cos(b) + cos(c)) / (2 sin(a) sin(b))) ... no, that's not right.
            
            # Let me use a direct computation: the circumcenter of three points on the unit sphere.
            # The circumcenter is the point x on the unit sphere that maximizes min(|x-a|, |x-b|, |x-c|).
            # This is equivalent to finding the center of the smallest circle containing a, b, c.
            
            # For three points, the circumcenter can be found as follows:
            # Let n = (b-a) × (c-a), normalized.
            # The circumcenter lies in the plane, so x = a + s*(b-a) + t*(c-a).
            # We need |x-a| = |x-b| = |x-c|.
            
            # Let's use the formula for the circumcenter of a triangle in 3D:
            # d = 2 * (a · (b × c) + b · (c × a) + c · (a × b))
            # This is 6 times the volume, which is 0 for coplanar points. So this doesn't work.
            
            # Let me use a different approach: the circumcenter of a spherical triangle is given by:
            # x = (a * sin(2A) + b * sin(2B) + c * sin(2C)) / |...|
            # where A, B, C are the angles of the triangle.
            
            # This is getting too complex. Let's use a numerical optimization for the circumcenter.
            # But that's slow. Let's just compute the maximum distance to the three vertices for the
            # centroid of the triangle and the circumcenter if it exists.
            
            # For now, let's compute the circumradius as the maximum over the three vertices of the
            # angle between the vertex and the circumcenter.
            
            # Let's find the circumcenter by solving:
            # |x - a|^2 = |x - b|^2 = |x - c|^2, |x| = 1
            # This gives:
            # 2 x · (b - a) = |b|^2 - |a|^2 = 0 (since |a|=|b|=1)
            # 2 x · (c - a) = 0
            # So x is orthogonal to (b-a) and (c-a), meaning x is parallel to (b-a) × (c-a).
            # So the circumcenter is ± ((b-a) × (c-a)) / |...|
            
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                # Degenerate triangle
                continue
            n = n / n_norm
            # The circumcenter is either n or -n. We need the one that is on the same side as the triangle.
            # Check which one is closer to the centroid.
            centroid = (a + b + c) / 3
            centroid = centroid / np.linalg.norm(centroid)
            if np.dot(n, centroid) < 0:
                n = -n
            
            # Now compute the angular distance from n to a, b, c
            d1 = np.arccos(np.clip(np.dot(n, a), -1, 1))
            d2 = np.arccos(np.clip(np.dot(n, b), -1, 1))
            d3 = np.arccos(np.clip(np.dot(n, c), -1, 1))
            r = max(d1, d2, d3)
            max_r = max(max_r, r)
        
        return np.degrees(max_r)
    except Exception:
        return np.inf


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_score = _covering_radius_deg(pts)
    best_pts = pts.copy()
    
    # Local optimization using gradient descent on the sphere
    # Objective: minimize the covering radius, approximated by minimizing the sum of squared distances
    # to the nearest points, but that's complex. Instead, let's use a force-based approach.
    
    # Use a repulsive force model: each point repels others, with a constraint to stay on the sphere.
    # This is similar to the Thomson problem but with a different potential.
    
    # Let's try a simple approach: iterate over points, and for each point, move it in the direction
    # that reduces the maximum distance to the nearest point.
    
    # Actually, let's use a more standard approach: minimize the sum of the k-nearest neighbor distances.
    # But for now, let's just do a simple local search.
    
    # For each point, try small random perturbations and keep if it improves the score.
    random.seed(42)
    
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, N - 1)
        # Generate a random perturbation
        perturbation = np.random.randn(3) * 0.1
        new_pts = pts.copy()
        new_pts[idx] = pts[idx] + perturbation
        new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        new_score = _covering_radius_deg(new_pts)
        if new_score < best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
        # Accept with some probability if not better (simulated annealing)
        elif random.random() < 0.1:
            pts = new_pts
    
    return best_pts
# EVOLVE-BLOCK-END
