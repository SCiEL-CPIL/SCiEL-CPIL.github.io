import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest configuration point. This is equivalent to the maximum circumradius
    of the facets of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        for tri in facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            # Compute the normal to the plane defined by p1, p2, p3
            # This normal direction corresponds to the circumcenter of the spherical triangle
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            # Ensure the normal points outward (away from origin)
            # The circumcenter for the covering radius calculation should be the pole of the facet
            # that is "outside" the hull relative to the origin? No, the points are on the sphere.
            # The convex hull contains the origin (assuming well-distributed points).
            # The facet normal pointing away from the origin is the direction of the "hole" or cap center.
            if np.dot(c, p1) < 0:
                c = -c
            
            # The angular distance from the circumcenter c to the vertices is the circumradius.
            # Since c is the normal to the plane, it is equidistant from p1, p2, p3 in Euclidean space?
            # No, c is on the sphere. The distance is angular.
            # The dot product c.p1 gives the cosine of the angle.
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            # Due to floating point errors, the distances to p2 and p3 might differ slightly.
            # The true circumradius is the maximum distance to any vertex (the cap must cover all).
            # However, for a planar triangle on a sphere, the intersection of the normal with the sphere
            # is equidistant from the vertices in terms of chordal distance, and thus angular distance.
            # So dist should be the same for all three. We just use one.
            
            if dist > max_radius:
                max_radius = dist
                
    except Exception:
        # If hull fails, return a large value
        return np.pi / 2
        
    return np.degrees(max_radius)

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    current_radius = best_radius
    
    step_size = 0.02
    min_step_size = 1e-4
    
    while time.time() < deadline:
        # Compute hull and find worst facet
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
            
            worst_facet_idx = -1
            worst_radius = 0.0
            worst_circumcenter = None
            worst_vertices = None
            
            for idx, tri in enumerate(facets):
                p1 = points[tri[0]]
                p2 = points[tri[1]]
                p3 = points[tri[2]]
                
                v1 = p2 - p1
                v2 = p3 - p1
                cross = np.cross(v1, v2)
                norm_cross = np.linalg.norm(cross)
                
                if norm_cross < 1e-12:
                    continue
                    
                c = cross / norm_cross
                
                # Ensure normal points outward
                if np.dot(c, p1) < 0:
                    c = -c
                
                dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
                dist = np.arccos(dot_val)
                
                if dist > worst_radius:
                    worst_radius = dist
                    worst_facet_idx = idx
                    worst_circumcenter = c
                    worst_vertices = tri
                    
        except Exception:
            break
            
        if worst_facet_idx == -1:
            break
            
        # Move vertices towards the circumcenter
        # We move all 3 vertices slightly towards the circumcenter to shrink the facet
        # Store previous positions for potential revert
        prev_v1 = points[worst_vertices[0]].copy()
        prev_v2 = points[worst_vertices[1]].copy()
        prev_v3 = points[worst_vertices[2]].copy()
        
        for v_idx in worst_vertices:
            p = points[v_idx]
            # Direction to move: towards circumcenter
            # Use vector difference projected onto tangent plane? Or just linear addition and normalize?
            # Linear addition and normalization is a standard simple update.
            direction = worst_circumcenter - p
            norm_dir = np.linalg.norm(direction)
            if norm_dir < 1e-12:
                continue
            direction /= norm_dir
            
            points[v_idx] = points[v_idx] + step_size * direction
            # Normalize
            points[v_idx] /= np.linalg.norm(points[v_idx])
        
        # Check improvement
        current_radius = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            # Slightly increase or maintain step size if improving? 
            # Usually we decay. Let's decay slowly.
            step_size *= 0.98
        else:
            # Revert if worse
            points[worst_vertices[0]] = prev_v1
            points[worst_vertices[1]] = prev_v2
            points[worst_vertices[2]] = prev_v3
            
            # Reduce step size
            step_size *= 0.9
            
            # If step size is too small, try a small random perturbation to escape local minima?
            # Or just break if we can't improve?
            # Let's add a small random jitter to the worst vertices to escape local optima
            if step_size < min_step_size:
                # Random perturbation
                for v_idx in worst_vertices:
                    noise = np.random.normal(0, 0.01, 3)
                    points[v_idx] = points[v_idx] + noise
                    points[v_idx] /= np.linalg.norm(points[v_idx])
                
                # Reset step size
                step_size = 0.01
                # Re-evaluate
                current_radius = _compute_covering_radius(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
                else:
                    # Revert jitter
                    points[worst_vertices[0]] = prev_v1
                    points[worst_vertices[1]] = prev_v2
                    points[worst_vertices[2]] = prev_v3
                    # If we can't improve even with jitter, we might be stuck.
                    # But let's continue for a bit to see if other facets change.
                    pass

    return best_points
