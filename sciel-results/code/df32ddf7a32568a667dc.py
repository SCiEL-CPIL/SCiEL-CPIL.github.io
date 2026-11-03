import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _get_circumcenters(points, hull):
    """
    Compute the circumcenters and circumradii for all facets of the convex hull.
    Returns:
        centers: array of shape (M, 3) with circumcenters on the unit sphere.
        radii: array of shape (M,) with angular circumradii.
        facets: array of shape (M, 3) with indices of vertices for each facet.
    """
    facets = hull.simplices
    M = len(facets)
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    for k in range(M):
        tri = facets[k]
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            # Degenerate triangle, assign a large radius or skip
            # If degenerate, it's likely a very small facet, radius ~0
            # But if points are coincident, it's a problem.
            # For robustness, if cross is 0, set radius to 0? 
            # Actually, if points are close, the circumcenter is ill-defined.
            # But for covering, small facets are good (small holes).
            # So we can ignore them for finding the MAX radius.
            radii[k] = 0.0
            centers[k] = p1 # placeholder
            continue
            
        c = cross / norm_cross
        
        # Determine which side is the "interior" cap center.
        # The circumcenter must be on the same side of the plane as the origin?
        # No, the circumcenter of the spherical triangle is the pole of the great circle 
        # passing through the vertices, on the side where the triangle lies?
        # Actually, for a convex hull of points on the sphere, the origin is inside the hull.
        # The facet normal $c$ should point OUTWARDS (away from origin) if the origin is on the other side.
        # Let's check the dot product with p1.
        # If c . p1 > 0, c is in the "forward" hemisphere relative to p1.
        # The angular distance is arccos(c . p1).
        # If the triangle is small, the circumcenter is close to the centroid.
        # The distance from c to p1 is the circumradius.
        
        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        
        # The circumradius is the angular distance.
        # However, there are two poles. One is the center of the small cap (triangle), 
        # the other is the center of the large cap (complement).
        # We want the radius of the empty cap. 
        # For a convex hull covering, the "hole" is the small cap if the triangle is small.
        # If the triangle is large (>90 deg sides), the "hole" might be the other side?
        # No, the convex hull facets correspond to the Delaunay triangulation.
        # The circumcircle of a Delaunay triangle is empty of other points.
        # The radius of this empty circle is the circumradius.
        # If the triangle is large, the "empty" region is the large cap?
        # No, Delaunay triangles on the sphere: the circumcircle is the great circle.
        # The "empty" condition means no points inside the spherical triangle.
        # The circumcenter is the point equidistant from vertices.
        # If the triangle is small, the circumcenter is "above" it.
        # If the triangle is large, the circumcenter is "below" it?
        # Actually, arccos(c . p1) gives the angle between c and p1.
        # If c . p1 > 0, angle < 90.
        # If c . p1 < 0, angle > 90.
        # The covering radius is the max distance to the nearest point.
        # If the circumcenter is at distance R > 90, it means the points are far apart.
        # But for 130 points, R is ~11 deg. So c . p1 should be positive.
        
        radii[k] = dist
        centers[k] = c
        
    return centers, radii, facets

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        centers, radii, _ = _get_circumcenters(points, hull)
        return np.degrees(np.max(radii))
    except Exception:
        return 180.0

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.01, points.shape)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Optimization parameters
    step_size = 0.05
    min_step_size = 1e-6
    attraction_strength = 1.0
    repulsion_strength = 0.1
    stagnant_count = 0
    max_stagnant = 50
    
    while time.time() < deadline:
        
        try:
            hull = ConvexHull(points)
            centers, radii, facets = _get_circumcenters(points, hull)
        except Exception:
            # If hull fails, add noise and retry
            points += rng.normal(0, 0.01, points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue

        current_radius_rad = np.max(radii)
        current_radius_deg = np.degrees(current_radius_rad)
        
        if current_radius_deg < best_radius:
            best_radius = current_radius_deg
            best_points = points.copy()
            stagnant_count = 0
            # If improved, we can slightly increase step size or keep it
        else:
            stagnant_count += 1
            # If stagnant, reduce step size
            step_size = max(step_size * 0.95, min_step_size)
            
            if stagnant_count > max_stagnant:
                # Try a random kick to escape local minimum
                kick = rng.normal(0, 0.05, points.shape)
                points += kick
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                step_size = 0.05
                stagnant_count = 0
                continue

        # Find the worst facet (max radius)
        worst_idx = np.argmax(radii)
        worst_center = centers[worst_idx]
        worst_facet = facets[worst_idx]
        
        # Compute forces
        # 1. Repulsion between all points (to maintain uniformity and prevent collapse)
        # diff[i,j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Force ~ 1/r^3 (vector direction diff/r, magnitude 1/r^2 -> diff/r^3)
        # We want a gentle repulsion.
        inv_r3 = 1.0 / (dist_sq * np.sqrt(dist_sq))
        repulsive_forces = np.sum(diff * inv_r3[:, :, np.newaxis], axis=1)
        
        # 2. Attraction to the worst hole for the vertices of the worst facet
        attractive_forces = np.zeros_like(points)
        for idx in worst_facet:
            # Vector from point to center
            vec = worst_center - points[idx]
            # Normalize to unit vector? Or use raw vector?
            # Raw vector magnitude depends on distance.
            # We want to move towards the center.
            attractive_forces[idx] += vec / np.linalg.norm(vec)
            
        # Combine forces
        # Scale attraction to be significant but not dominant
        forces = repulsive_forces * repulsion_strength + attractive_forces * attraction_strength
        
        # Project onto tangent plane
        dot_fn = np.sum(forces * points, axis=1, keepdims=True)
        tangent_forces = forces - dot_fn * points
        
        # Update points
        new_points = points + step_size * tangent_forces
        
        # Normalize
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check for validity (non-finite, coincident)
        if not np.all(np.isfinite(new_points)):
            break
            
        points = new_points
        
        # Optional: Check improvement more frequently? 
        # The loop already computes the radius at the start of the next iteration.
        # We can check now to update best_points if it's better.
        # But we already did that at the top of the loop.
        # However, we updated `points` to `new_points`. 
        # The `best_points` update happens at the top of the NEXT loop.
        # This is fine, but we might miss a better intermediate state if we break.
        # Let's check now.
        try:
            hull_check = ConvexHull(points)
            centers_check, radii_check, _ = _get_circumcenters(points, hull_check)
            check_radius = np.degrees(np.max(radii_check))
            if check_radius < best_radius:
                best_radius = check_radius
                best_points = points.copy()
                stagnant_count = 0
        except:
            pass

    return best_points
