import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    Returns radius in degrees and the indices of the worst facets.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        worst_facets = []
        radii = []
        
        for tri in facets:
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
            
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            if dist > np.pi / 2:
                dist = np.pi - dist
                
            radii.append(dist)
            if dist > max_radius:
                max_radius = dist
                worst_facets = [tri]
            elif abs(dist - max_radius) < 1e-6:
                worst_facets.append(tri)
                
    except Exception:
        # If hull fails, return a large value and empty facets
        return 180.0, []
        
    return np.degrees(max_radius), worst_facets

def _compute_all_circumcenters(points):
    """
    Compute circumcenters and radii for all facets.
    Returns list of (circumcenter_vector, radius_radians, facet_indices)
    """
    hull = ConvexHull(points)
    facets = hull.simplices
    results = []
    
    for tri in facets:
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
        
        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        
        if dist > np.pi / 2:
            dist = np.pi - dist
            c = -c # Ensure c is the "inner" normal relative to the cap? 
            # Actually, for small triangles, c is the outward normal of the hull facet?
            # Let's check: if p1,p2,p3 are close, cross is roughly normal to sphere.
            # The circumcenter on the sphere is the point equidistant.
            # If dist < pi/2, c is the center of the small cap.
            # If dist > pi/2, -c is the center of the small cap?
            # For covering, we care about the empty cap. The center of the empty cap is the circumcenter.
            # If the triangle is small, the circumcenter is "above" it.
            # Let's just use the c that minimizes distance to vertices?
            # No, c and -c are antipodal. One is dist, other is pi-dist.
            # We want the one that is the center of the *hole*.
            # For a convex hull of points on a sphere, the facets are "faces".
            # The "hole" is the region of the sphere not covered by caps of radius R.
            # The critical points are the circumcenters of the Delaunay triangles.
            # For small triangles, the circumcenter is close to the centroid.
            
            # Let's stick to the convention: if dist > pi/2, the "center" of the spherical triangle
            # in the sense of the cap center is actually -c? 
            # Wait. If p1,p2,p3 form a small triangle, the normal c points away from the origin 
            # if the triangle is oriented CCW? 
            # Let's verify. p1=(1,0,0), p2=(0,1,0), p3=(0,0,1).
            # v1=(-1,1,0), v2=(-1,0,1). cross = (1,1,1). norm=sqrt(3). c=(1,1,1)/sqrt(3).
            # dot(c,p1) = 1/sqrt(3) ~ 0.577. arccos(0.577) ~ 54.7 deg.
            # This is < 90. So c is the circumcenter.
            
            # If the triangle was large, e.g. covering most of the sphere?
            # Then the "empty" part is small. The circumcenter of the *complement*?
            # No, the covering radius is defined by the max distance to nearest point.
            # This max occurs at the circumcenter of a Delaunay triangle.
            # For a good packing, all triangles are small.
            # So we can assume dist < pi/2.
            
            pass
            
        results.append((c, dist, tri))
        
    return results

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
    best_radius, _ = _compute_covering_radius(points)
    
    step_size = 0.05
    min_step_size = 1e-5
    
    # Structural Change:
    # Instead of global repulsion, we use a "hole-filling" strategy.
    # We identify the largest holes (facets with max circumradius).
    # We move the vertices of these facets towards the circumcenter of the hole.
    # This directly reduces the covering radius of those specific holes.
    
    while time.time() < deadline:
        
        # 1. Compute current covering radius and worst facets
        current_radius, worst_facets = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # 2. If no worst facets (should not happen), break
        if not worst_facets:
            break
            
        # 3. Compute circumcenters for the worst facets
        # We compute gradients for the worst facets only to save time? 
        # Or all facets? 
        # Let's compute for all, but weight the update by how "bad" the facet is.
        
        # Actually, let's just take the top K worst facets to reduce computation and noise.
        # We need to re-compute radii to sort them.
        # _compute_covering_radius only returns the max and the ones equal to max.
        # Let's modify the logic to get the top few.
        
        # For efficiency, let's just use the worst facets found in _compute_covering_radius.
        # But there might be many. Let's limit to top 10.
        
        # We need the circumcenter vectors.
        # Let's re-calculate for the specific worst facets.
        
        gradient = np.zeros_like(points)
        
        for tri in worst_facets[:10]: # Limit to top 10 worst
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
            
            # Check which side is the "hole"
            # The hole center is the circumcenter.
            # If the triangle is small, c is the center.
            # If the triangle is large, -c might be the center of the *complement*?
            # But for covering, we care about the point on the sphere farthest from vertices.
            # This is the circumcenter of the Delaunay triangle.
            # For small triangles, it's c.
            
            # Move vertices towards c
            # Direction from p_i to c on sphere is tangent component of (c - p_i)
            # Or simply, gradient of distance(p_i, c) wrt p_i.
            # dist = arccos(p_i . c).
            # d(dist)/d(p_i) = -c / sqrt(1 - (p_i . c)^2)
            # This is the vector pointing from p_i towards c (tangent).
            
            for idx in tri:
                p_i = points[idx]
                dot_pc = np.dot(p_i, c)
                dot_pc = np.clip(dot_pc, -1.0, 1.0)
                
                # If dot_pc is close to 1, p_i is close to c. Gradient is small.
                # If dot_pc is close to -1, p_i is far.
                
                # We want to move p_i to reduce the distance to c?
                # No, we want to reduce the *radius* of the triangle.
                # The radius is the distance from c to any vertex.
                # If we move vertices towards c, the radius decreases.
                
                # Gradient of distance(p_i, c) wrt p_i is direction from p_i to c.
                # So we should add step_size * direction_to_c to p_i.
                
                # direction_to_c is the tangent vector pointing from p_i to c.
                # It is proportional to (c - (c.p_i)p_i).
                
                proj = dot_pc * p_i
                tangent_dir = c - proj
                norm_dir = np.linalg.norm(tangent_dir)
                
                if norm_dir > 1e-9:
                    tangent_dir /= norm_dir
                    gradient[idx] += tangent_dir
                    
        # Normalize gradient
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1e-9:
            gradient /= grad_norm
            
            # Update points
            # Move towards the centers of the holes
            new_points = points + step_size * gradient
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            # Check if this step improved the radius
            new_radius, _ = _compute_covering_radius(new_points)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
                points = new_points
                # Increase step size if we are making good progress?
                # step_size = min(step_size * 1.1, 0.1)
            else:
                # If worse, revert and decrease step size
                step_size = max(step_size * 0.8, min_step_size)
                
                # If step size is very small, try a random kick
                if step_size < 1e-6:
                    kick = rng.normal(0, 0.01, points.shape)
                    points += kick
                    points /= np.linalg.norm(points, axis=1, keepdims=True)
                    step_size = 0.01

    return best_points
