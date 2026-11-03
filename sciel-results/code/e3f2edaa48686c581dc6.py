"""
Covering the unit sphere with 130 points using Lloyd's algorithm (Voronoi relaxation).

Strategy:
1. Start with a well-distributed initial configuration (Fibonacci spiral).
2. Compute the spherical Voronoi diagram via Delaunay triangulation of the convex hull.
3. For each Voronoi cell, compute its centroid (the point that minimizes the maximum
   angular distance within the cell) and move the site to that centroid.
4. Iterate until convergence or time limit.

The key insight is that Lloyd's algorithm on the sphere finds configurations where
each site is at the centroid of its Voronoi cell, which equalizes the covering radius
across cells and often finds deeper local minima than simple facet-pushing.
"""

import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

N = 130

def _angular_distance(a, b):
    """Angular distance between two unit vectors in radians."""
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.arccos(dot)

def _compute_covering_radius(points):
    """
    Compute the covering radius (in radians) by finding the largest circumradius
    of the facets of the convex hull.
    
    For each facet (triangle) of the convex hull, the farthest point from the
    nearest site is at the circumcenter of that triangle (if it lies on the
    sphere side of the facet). The angular distance from the origin to that
    circumcenter projected onto the sphere gives the local covering radius.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # fallback: worst case
    
    max_radius = 0.0
    for simplex in hull.simplices:
        # Get the three points of this triangular facet
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Compute the circumcenter of the triangle on the unit sphere
        # The circumcenter is the intersection of the plane through p1,p2,p3 with the sphere
        # For a triangle on the unit sphere, the circumcenter direction is:
        #   n = (p1 x p2) x (p2 x p3)  (normalized)
        # which is the direction of the circumcenter of the spherical triangle
        
        # Alternative: find the point equidistant from p1, p2, p3 on the sphere
        # This is the normalized cross product approach or solve linear system
        
        # The circumcenter of a spherical triangle can be found as:
        # c = (|p1-p2|^2 * (p2 x p3) + |p2-p3|^2 * (p3 x p1) + |p3-p1|^2 * (p1 x p2)) / |...|
        # Actually, let's use a more robust method:
        
        # The circumcenter direction is parallel to:
        # (p1 - p2) x (p2 - p3) ... no, that's the normal to the plane
        # We need the point on the sphere equidistant from all three vertices
        
        # Let's solve: find x such that x·p1 = x·p2 = x·p3 and |x| = 1
        # This means x is in the null space of [p1-p2; p2-p3]
        
        v1 = p1 - p2
        v2 = p2 - p3
        # x is perpendicular to both v1 and v2, so x is parallel to v1 x v2
        x = np.cross(v1, v2)
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-12:
            continue
        x = x / x_norm
        
        # Make sure x is on the same side as the facet outward normal
        # The outward normal of the facet is in the direction of the centroid from origin
        centroid = (p1 + p2 + p3) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        
        # The angular distance from the nearest site to x is the local covering radius
        # for this facet
        d = _angular_distance(x, p1)  # distance to any of the three vertices is the same
        
        if d > max_radius:
            max_radius = d
    
    return max_radius

def _spherical_centroid(vertices):
    """
    Compute the centroid of a set of points on the sphere.
    For a spherical polygon, we use the normalized sum as an approximation,
    but for better results, we find the point that minimizes the maximum
    distance to the vertices (which is the circumcenter for small cells).
    
    For Voronoi cells, we use the normalized average as a good approximation.
    """
    c = np.sum(vertices, axis=0)
    norm = np.linalg.norm(c)
    if norm < 1e-12:
        return vertices[0].copy()
    return c / norm

def _lloyd_step(points):
    """
    Perform one Lloyd iteration:
    1. Compute Delaunay triangulation (dual of Voronoi)
    2. For each site, find its Voronoi cell vertices
    3. Move site to centroid of its cell
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points.copy()
    
    # Build Delaunay triangulation from the convex hull
    try:
        tri = Delaunay(points)
    except Exception:
        return points.copy()
    
    new_points = points.copy()
    
    # For each point, find the Delaunay triangles it belongs to
    # The Voronoi cell of point i is bounded by the circumcenters of
    # the Delaunay triangles containing point i
    
    # Build adjacency: for each point, collect all triangles it's in
    point_to_tris = [[] for _ in range(N)]
    for tri_idx, simplex in enumerate(tri.simplices):
        for vertex in simplex:
            point_to_tris[vertex].append(tri_idx)
    
    for i in range(N):
        tris = point_to_tris[i]
        if len(tris) == 0:
            continue
        
        # Collect circumcenters of all triangles containing point i
        circumcenters = []
        for tri_idx in tris:
            simplex = tri.simplices[tri_idx]
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            # Compute circumcenter of this triangle
            v1 = p1 - p2
            v2 = p2 - p3
            x = np.cross(v1, v2)
            x_norm = np.linalg.norm(x)
            if x_norm < 1e-12:
                continue
            x = x / x_norm
            
            # Ensure correct orientation
            centroid = (p1 + p2 + p3) / 3.0
            if np.dot(x, centroid) < 0:
                x = -x
            
            circumcenters.append(x)
        
        if len(circumcenters) == 0:
            continue
        
        circumcenters = np.array(circumcenters)
        
        # The new position is the centroid of the Voronoi cell
        # Use normalized average of circumcenters
        new_pos = _spherical_centroid(circumcenters)
        new_points[i] = new_pos
    
    # Renormalize
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    new_points = new_points / norms
    
    return new_points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s deadline, 10s safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Lloyd's algorithm iterations
    max_iterations = 500
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        new_points = _lloyd_step(points)
        
        # Check for coincident points
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = _angular_distance(new_points[i], new_points[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-6:
            break
        
        points = new_points
        
        # Compute covering radius
        current_radius = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Check convergence
        if it > 10:
            displacement = np.linalg.norm(points - best_points, axis=1).max()
            if displacement < 1e-8:
                break
    
    return best_points
