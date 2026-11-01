"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _fibonacci_sphere(n):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_covering_radius(points):
    """
    Estimate the covering radius by checking the circumradius of the facets
    of the convex hull.
    """
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        # For each facet, calculate the circumradius
        # The circumradius of a spherical triangle is related to the planar circumradius
        # of the triangle formed by the vertices, adjusted for the sphere.
        # Actually, the covering radius is the max distance from any point on the sphere
        # to the nearest point in the set.
        # For a set of points, the farthest point from the set is either:
        # 1. A vertex of the convex hull of the complement? No.
        # 2. The circumcenter of a facet of the convex hull of the points, if it lies
        #    on the sphere (which it does for spherical caps).
        
        # The covering radius is the maximum over all facets of the convex hull
        # of the angular distance from the facet's circumcenter to its vertices.
        
        for simplex in hull.simplices:
            # simplex is an array of 3 indices (for 3D)
            pts = points[simplex]
            # Compute circumcenter of the triangle formed by pts on the sphere
            # This is the normalized cross product of the edge vectors?
            # No, the circumcenter of a spherical triangle is the intersection of the sphere
            # and the plane defined by the three points, normalized?
            # Wait, the circumcenter of the spherical cap is the point on the sphere
            # equidistant from the three vertices.
            
            # Let a, b, c be the vertices.
            # The normal to the plane abc is n = (b-a) x (c-a).
            # The circumcenter C in R3 lies on the line t*n.
            # Since a, b, c are on the unit sphere, the circumcenter of the triangle
            # in R3 is not necessarily on the sphere.
            # However, the point on the sphere equidistant from a, b, c is the
            # normalization of the vector sum of the unit vectors? No.
            
            # The spherical circumcenter is given by the normalized vector:
            # v = (a x b) x (b x c) ...?
            
            # Simpler: The circumradius of the facet is the angle between the
            # circumcenter and any vertex.
            # We can compute the circumradius of the planar triangle and map it?
            # No, that's not exact.
            
            # Let's use the formula for the circumradius of a spherical triangle.
            # Or simpler: The covering radius is the max over all points y on the sphere
            # of min_i dist(y, p_i).
            # This maximum is attained at a vertex of the Voronoi diagram on the sphere.
            # The vertices of the Voronoi diagram on the sphere correspond to the
            # circumcenters of the facets of the convex hull.
            
            # So we just need to compute the circumcenter of each facet and
            # the distance to its vertices.
            
            a, b, c = pts
            # Circumcenter of spherical triangle abc
            # It is the normalized vector (a x b) x (b x c) ? No.
            # It is the solution to C.a = C.b = C.c and |C|=1.
            # C is parallel to (a x b) x (b x c) ?
            # Let's use the cross product method.
            # The normal to the great circle through a and b is n_ab = a x b.
            # The circumcenter C is orthogonal to n_ab and n_bc.
            # So C is parallel to n_ab x n_bc.
            n_ab = np.cross(a, b)
            n_bc = np.cross(b, c)
            c_center = np.cross(n_ab, n_bc)
            
            if np.linalg.norm(c_center) < 1e-10:
                # Degenerate case, points collinear or coincident
                continue
                
            c_center = c_center / np.linalg.norm(c_center)
            
            # The circumradius is the angle between c_center and a
            cos_theta = np.dot(c_center, a)
            # Clamp for numerical stability
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            if theta > max_radius:
                max_radius = theta
                
    except Exception:
        # If hull fails, fall back to a rough estimate
        # Sample points on the sphere
        n_samples = 1000
        # Generate random points
        rng = np.random.default_rng(42)
        samples = rng.normal(size=(n_samples, 3))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        
        min_dists = np.min(np.arccos(np.clip(np.dot(samples, points.T), -1, 1)), axis=1)
        max_radius = np.max(min_dists)
        
    return np.degrees(max_radius)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Initial configuration
    points = _fibonacci_sphere(N)
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    # Lloyd's Algorithm on the Sphere
    # Iterate until time runs out or convergence
    
    for iteration in range(1000):
        if time.time() > deadline:
            break
            
        try:
            hull = ConvexHull(points)
        except Exception:
            # If points are degenerate, perturb them
            points += np.random.normal(0, 0.01, size=points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
            
        # For each point, find the centroid of its Voronoi cell
        # Approximation: Centroid of the vertices of the adjacent hull faces
        new_points = np.zeros_like(points)
        
        # Map point index to list of face indices
        point_to_faces = {i: [] for i in range(N)}
        
        for i, simplex in enumerate(hull.simplices):
            for v in simplex:
                point_to_faces[v].append(i)
                
        for i in range(N):
            if not point_to_faces[i]:
                # Point is not on the hull? Should not happen for convex position
                new_points[i] = points[i]
                continue
                
            # Sum the normals of the adjacent faces
            # The normal of a face points outward.
            # The Voronoi cell centroid is in the direction of the sum of these normals?
            # Actually, the centroid of the spherical Voronoi cell is closer to the
            # normalized sum of the vertices of the adjacent faces.
            
            # Let's try the normalized sum of the face normals first.
            face_normals = hull.equations[hull.simplices[point_to_faces[i]]]
            # hull.equations gives [a, b, c, d] for ax+by+cz+d=0. Normal is [a,b,c].
            normals = face_normals[:, :3]
            
            # The normals point outward. The Voronoi cell is "inside" the hull.
            # So we should sum the inward normals? Or just sum the outward normals
            # and the resulting vector points into the cell?
            # For a point on the convex hull, the Voronoi cell is the set of points
            # closer to it than to others. It is bounded by planes bisecting the edges.
            # The centroid is roughly in the direction of the sum of the normals
            # of the adjacent faces.
            
            centroid_vec = np.sum(normals, axis=0)
            
            if np.linalg.norm(centroid_vec) < 1e-10:
                new_points[i] = points[i]
            else:
                new_points[i] = centroid_vec / np.linalg.norm(centroid_vec)
                
        # Update points
        points = new_points
        
        # Normalize
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Check radius
        current_radius = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Optional: Add a small random perturbation to escape local minima
        if iteration % 50 == 0 and iteration > 0:
            # Small jitter
            jitter = np.random.normal(0, 0.01, size=points.shape)
            points = points + jitter
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
    return best_points

# EVOLVE-BLOCK-END
