"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _compute_covering_radius(points):
    """
    Compute the covering radius in radians.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest configuration point. This equals the maximum circumradius of the
    spherical Voronoi cells, which can be found via the convex hull of the points.
    
    For each facet of the convex hull, we find the circumcenter of the facet (in 3D)
    and project it back to the sphere. The angular distance from that point to the
    nearest configuration point gives the covering radius for that facet.
    """
    # Build convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use pairwise distances
        from scipy.spatial.distance import cdist
        # Sample points on sphere
        n_samples = 5000
        theta = np.random.uniform(0, 2 * np.pi, n_samples)
        phi = np.arccos(2 * np.random.uniform(0, 1, n_samples) - 1)
        samples = np.column_stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        d = cdist(samples, points)
        min_d = d.min(axis=1)
        return np.arccos(np.clip(min_d, -1, 1))

    max_angle = 0.0
    for simplex in hull.simplices:
        facet_points = points[simplex]
        
        # Find the point on the sphere that is farthest from all facet points
        # This is the circumcenter of the spherical triangle
        # We can find it by finding the point that maximizes the minimum distance to facet points
        
        # For a triangular facet, the circumcenter in 3D can be found
        # But we need the point on the unit sphere that is the "farthest point" 
        # from the facet's points in angular distance
        
        # The covering radius for a facet is determined by the point on the sphere
        # that is the circumcenter of the spherical polygon formed by the facet vertices
        
        # Simple approach: use the centroid of the facet vertices, project to sphere
        centroid = facet_points.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm < 1e-10:
            continue
        centroid_unit = centroid / norm
        
        # Distance from centroid to nearest point
        d = np.min(np.sum(facet_points * centroid_unit, axis=1))
        angle = np.arccos(np.clip(d, -1, 1))
        
        # But the actual farthest point might be different. Let's use a more robust method.
        # The farthest point from a set of points on the sphere within their convex hull
        # is the circumcenter. For a triangle, this is well-defined.
        
        # Let's compute the circumcenter of the triangle (in 3D)
        a, b, c = facet_points[0], facet_points[1], facet_points[2]
        
        # Circumcenter of triangle in 3D
        # Solve: |x - a|^2 = |x - b|^2 = |x - c|^2
        # This gives a linear system
        d_ab = 2 * (b - a)
        d_ac = 2 * (c - a)
        rhs_ab = np.dot(b, b) - np.dot(a, a)
        rhs_ac = np.dot(c, c) - np.dot(a, a)
        
        A = np.array([d_ab, d_ac])
        b_vec = np.array([rhs_ab, rhs_ac])
        
        # The circumcenter lies in the plane of the triangle
        # Normal to the plane
        normal = np.cross(b - a, c - a)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-10:
            continue
        normal = normal / normal_norm
        
        # We need a third equation: x is in the plane, so normal . x = normal . a
        A_full = np.vstack([A, normal])
        b_full = np.concatenate([b_vec, [np.dot(normal, a)]])
        
        try:
            circumcenter = np.linalg.solve(A_full, b_full)
        except np.linalg.LinAlgError:
            continue
        
        # Project to unit sphere
        circum_norm = np.linalg.norm(circumcenter)
        if circum_norm < 1e-10:
            continue
        circum_unit = circumcenter / circum_norm
        
        # Check if this point is actually inside the spherical triangle
        # The angular distance from circum_unit to each vertex should be <= 90 degrees
        # for the circumcenter to be the farthest point (otherwise the farthest point
        # is on the boundary)
        dots = np.sum(facet_points * circum_unit, axis=1)
        
        # The covering radius for this facet is the max angular distance from circum_unit
        # to any point in the facet's Voronoi cell, which is the max distance to the vertices
        # if circumcenter is inside, or the max distance to an edge midpoint otherwise
        
        # For simplicity, compute the distance from circum_unit to the nearest configuration point
        # This should give the covering radius contribution from this facet
        d_to_nearest = np.min(np.sum(points * circum_unit, axis=1))
        angle = np.arccos(np.clip(d_to_nearest, -1, 1))
        
        max_angle = max(max_angle, angle)
    
    return max_angle


def _lloyd_relaxation(points, n_iterations=1000, learning_rate=1.0):
    """
    Perform Lloyd's relaxation on the sphere.
    Move each point to the centroid of its spherical Voronoi cell.
    """
    points = points.copy()
    n = len(points)
    
    for it in range(n_iterations):
        # Compute Voronoi cells using convex hull
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # For each point, find its Voronoi cell (the set of points on the sphere
        # for which this point is the nearest)
        # We approximate by finding the facet(s) adjacent to each point
        
        new_points = np.zeros_like(points)
        
        for i in range(n):
            # Find all facets that include point i
            # The Voronoi cell of point i is the set of directions where i is nearest
            # We approximate by averaging the outward normals of the facets adjacent to i
            
            # Find facets containing point i
            adjacent_facets = []
            for simplex in hull.simplices:
                if i in simplex:
                    adjacent_facets.append(simplex)
            
            if len(adjacent_facets) == 0:
                new_points[i] = points[i]
                continue
            
            # For each adjacent facet, compute the facet normal
            # The Voronoi cell direction is the average of the facet normals
            facet_normals = []
            for simplex in adjacent_facets:
                fp = points[simplex]
                normal = np.cross(fp[1] - fp[0], fp[2] - fp[0])
                normal_norm = np.linalg.norm(normal)
                if normal_norm > 1e-10:
                    # Ensure normal points outward
                    centroid = fp.mean(axis=0)
                    if np.dot(normal, centroid) < 0:
                        normal = -normal
                    facet_normals.append(normal / normal_norm)
            
            if len(facet_normals) == 0:
                new_points[i] = points[i]
                continue
            
            # Average the facet normals
            avg_normal = np.mean(facet_normals, axis=0)
            avg_norm = np.linalg.norm(avg_normal)
            if avg_norm < 1e-10:
                new_points[i] = points[i]
            else:
                new_points[i] = avg_normal / avg_norm
        
        # Update points with learning rate
        points = (1 - learning_rate) * points + learning_rate * new_points
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
    
    return points


def _generate_initial_configurations(n_configs=10):
    """Generate multiple initial configurations."""
    configs = []
    
    # Fibonacci spiral
    for config_idx in range(n_configs):
        if config_idx == 0:
            # Standard Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # Random initial configuration
            points = np.random.randn(N, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        
        configs.append(points)
    
    return configs


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate multiple initial configurations
    configs = _generate_initial_configurations(5)
    
    best_points = None
    best_radius = np.inf
    
    for config in configs:
        if time.time() > deadline - 5.0:
            break
        
        # Perform Lloyd's relaxation
        relaxed = _lloyd_relaxation(config, n_iterations=500, learning_rate=0.5)
        
        # Compute covering radius
        radius = _compute_covering_radius(relaxed)
        
        if radius < best_radius:
            best_radius = radius
            best_points = relaxed
    
    # Final polish: more iterations with smaller learning rate
    if best_points is not None and time.time() < deadline - 10.0:
        polished = _lloyd_relaxation(best_points, n_iterations=500, learning_rate=0.1)
        polished_radius = _compute_covering_radius(polished)
        if polished_radius < best_radius:
            best_points = polished
            best_radius = polished_radius
    
    if best_points is None:
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
