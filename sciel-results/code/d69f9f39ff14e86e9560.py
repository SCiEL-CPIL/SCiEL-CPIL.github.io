"""Spherical Riemannian gradient flow for minimizing covering radius of 50 points."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _spherical_circumcenter(tri):
    """
    Compute the spherical circumcenter of a triangle on the unit sphere.
    Returns the circumcenter and the circumradius (angular distance).
    """
    a, b, c = tri[0], tri[1], tri[2]
    
    # We want a point p on the sphere such that p dot a = p dot b = p dot c = cos(R)
    # Let p be the circumcenter.
    # The vector p is in the span of a, b, c.
    # Actually, p is orthogonal to (a-b) and (a-c) in the sense that p dot (a-b) = 0? No.
    # p dot a = p dot b => p dot (a-b) = 0.
    # So p is orthogonal to u = a-b and v = a-c.
    # Thus p is parallel to w = cross(u, v).
    
    u = a - b
    v = a - c
    w = np.cross(u, v)
    norm_w = np.linalg.norm(w)
    
    if norm_w < 1e-12:
        # Degenerate case: points are collinear on a great circle
        # Find the point on the great circle that is equidistant from all three
        # This is effectively the midpoint of the longest arc
        d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
        d_ac = np.arccos(np.clip(np.dot(a, c), -1, 1))
        d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
        
        # The circumcenter is the point on the great circle passing through a,b,c
        # that is equidistant.
        # If they are on a great circle, the circumcenter is the midpoint of the longest arc?
        # No, the circumcenter of a degenerate triangle on a sphere is not well-defined in the same way.
        # We return the midpoint of the longest edge as a fallback.
        if d_ab >= d_ac and d_ab >= d_bc:
            mid = a + b
        elif d_ac >= d_ab and d_ac >= d_bc:
            mid = a + c
        else:
            mid = b + c
            
        norm_mid = np.linalg.norm(mid)
        if norm_mid < 1e-12:
            # Antipodal points?
            return a, np.pi/2
        p = mid / norm_mid
        # Calculate radius
        r = np.arccos(np.clip(np.dot(p, a), -1, 1))
        return p, r

    # w is the direction of the circumcenter
    p = w / norm_w
    
    # There are two candidates: p and -p.
    # We need to determine which one is the circumcenter of the convex hull facet.
    # The circumcenter of a facet of the convex hull of points on the sphere
    # is the point on the sphere that is "farthest" from the origin in the sense of the normal?
    # No, the circumcenter is the point equidistant from the vertices.
    # Both p and -p are equidistant.
    # The one that is the "outer" circumcenter (facing outwards from the convex hull)
    # is the one that has a positive dot product with the centroid of the triangle?
    # Actually, for a convex hull facet, the circumcenter is the point on the sphere
    # that is NOT covered by any other point.
    # Let's check both. The circumradius R is arccos(p dot a).
    # For the convex hull, the relevant circumcenter is the one that lies on the same side
    # of the great circle defined by the triangle as the "exterior".
    
    # A robust way: The circumcenter of the spherical triangle corresponding to the convex hull facet
    # is the one that maximizes the distance to the vertices? No, they are equal.
    # It is the one that is "outside" the triangle.
    # Let's compute the dot product of p with the centroid (normalized sum).
    centroid = a + b + c
    norm_centroid = np.linalg.norm(centroid)
    if norm_centroid < 1e-12:
        # Triangle contains the origin or is degenerate
        p = p # arbitrary
    else:
        centroid_norm = centroid / norm_centroid
        # The circumcenter should be on the same side as the centroid if the triangle is small (< hemisphere)
        # If dot(p, centroid_norm) < 0, flip p
        if np.dot(p, centroid_norm) < 0:
            p = -p
            
    # Calculate radius
    # It should be equal for all three, but due to numerical errors, take max or mean
    d1 = np.arccos(np.clip(np.dot(p, a), -1, 1))
    d2 = np.arccos(np.clip(np.dot(p, b), -1, 1))
    d3 = np.arccos(np.clip(np.dot(p, c), -1, 1))
    r = np.mean([d1, d2, d3])
    
    return p, r

def _compute_covering_radius(pts):
    """
    Compute the covering radius by finding the convex hull and checking each facet's circumradius.
    Returns the max radius and the indices of the facets achieving it (or close to it).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, []
    
    radii = []
    for simplex in hull.simplices:
        tri = pts[simplex]
        cc, r = _spherical_circumcenter(tri)
        radii.append(r)
        
    radii = np.array(radii)
    max_radius = np.max(radii)
    
    # Find facets with radii close to max (within 1e-4)
    active_indices = np.where(radii > max_radius - 1e-4)[0]
    
    return max_radius, active_indices

def _gradient_step(pts, active_indices, hull, step_size):
    """
    Perform a gradient step.
    Aggregates gradients from all active facets.
    """
    grad = np.zeros_like(pts)
    
    for i in active_indices:
        simplex = hull.simplices[i]
        tri = pts[simplex]
        cc, r = _spherical_circumcenter(tri)
        
        # Gradient of the circumradius with respect to the vertices
        # R = arccos(p dot v_i)
        # dR/dv_i = - (p / sqrt(1 - (p dot v_i)^2)) dot (I - v_i v_i^T)
        # Since p is the circumcenter, p dot v_i = cos(R).
        # So dR/dv_i = - p / sin(R) projected onto tangent space?
        # Actually, the gradient of the angular distance d(v, cc) with respect to v is
        # - (cc - (v dot cc) v) / sin(d)
        # This is the direction on the sphere from v towards cc?
        # Let's verify. If we move v towards cc, d decreases.
        # The vector cc - (v dot cc) v points from v towards cc in the tangent space.
        # So the gradient of d is in that direction.
        # We want to minimize d, so we move in the negative gradient direction,
        # which is towards cc.
        
        for j, idx in enumerate(simplex):
            v = pts[idx]
            dot_vc = np.dot(v, cc)
            # Tangent vector from v to cc
            t = cc - dot_vc * v
            norm_t = np.linalg.norm(t)
            
            if norm_t > 1e-12:
                # Gradient of distance is -t/norm_t ?
                # Let's check. f(v) = arccos(v dot cc).
                # df/dv = -1/sqrt(1-(v dot cc)^2) * cc.
                # Projected to tangent space: grad_tan = df/dv - (df/dv dot v) v
                # df/dv dot v = - (v dot cc) / sin(R) = -cos(R)/sin(R) = -cot(R)
                # grad_tan = -cc/sin(R) + cot(R) v
                # We know cc = cos(R) v + sin(R) u, where u is unit tangent towards cc.
                # So -cc/sin(R) = -cot(R) v - u
                # grad_tan = -cot(R) v - u + cot(R) v = -u
                # So the gradient of the distance is -u, where u is the unit tangent vector from v to cc.
                # Therefore, to minimize the distance, we move in the direction +u (towards cc).
                
                u = t / norm_t
                # Weight by 1/sin(R) ? No, the gradient magnitude is 1.
                # But we want to minimize the MAX radius.
                # If we use subgradient of max, we average gradients of active functions.
                grad[idx] += u
                
    # Normalize gradient on the sphere
    # The gradient is already in the tangent space?
    # u is tangent to v. But we are summing u's for different vertices?
    # No, for each vertex, we sum the u vectors from all facets containing it.
    # The sum is a tangent vector at that vertex.
    
    # Apply step
    new_pts = pts.copy()
    for i in range(N):
        g = grad[i]
        norm_g = np.linalg.norm(g)
        if norm_g > 1e-12:
            # Move in direction of g
            new_pts[i] = pts[i] + step_size * g
            new_pts[i] = _normalize(new_pts[i].reshape(1, -1))[0]
            
    return new_pts

def _optimize_single_start(pts, max_iters=2000, timeout=100.0):
    """
    Optimize a single starting configuration.
    """
    start_time = time.time()
    current_radius, active_indices = _compute_covering_radius(pts)
    best_pts = pts.copy()
    best_radius = current_radius
    
    step_size = 0.05
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
            
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        # Recompute active indices
        radii = []
        for simplex in hull.simplices:
            tri = pts[simplex]
            cc, r = _spherical_circumcenter(tri)
            radii.append(r)
        radii = np.array(radii)
        max_radius = np.max(radii)
        active_indices = np.where(radii > max_radius - 1e-4)[0]
        
        if max_radius < best_radius:
            best_radius = max_radius
            best_pts = pts.copy()
            
        if len(active_indices) == 0:
            break
            
        # Line search
        improved = False
        for shrink in range(20):
            new_pts = _gradient_step(pts, active_indices, hull, step_size)
            new_radius, _ = _compute_covering_radius(new_pts)
            
            if new_radius < best_radius:
                pts = new_pts
                best_radius = new_radius
                best_pts = pts.copy()
                improved = True
                break
            else:
                step_size *= 0.5
                
        if not improved:
            # Try to perturb if stuck
            if step_size < 1e-6:
                # Add small noise
                pts = pts + 0.01 * np.random.randn(N, 3)
                pts = _normalize(pts)
                step_size = 0.05
            else:
                break
                
    return best_pts, best_radius

def construct():
    # Use multiple random rotations of the Fibonacci spiral to escape local minima
    best_pts = None
    best_radius = np.pi
    
    # Total time limit for the whole function is 120s.
    # We allocate most of it to optimization.
    start_time = time.time()
    time_budget = 110.0
    
    # Try several starts
    num_starts = 5
    time_per_start = time_budget / num_starts
    
    for k in range(num_starts):
        # Generate Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        pts = _normalize(pts)
        
        # Apply a random rotation
        # Generate a random rotation matrix
        # Using QR decomposition of a random Gaussian matrix
        M = np.random.randn(3, 3)
        Q, R = np.linalg.qr(M)
        # Ensure proper rotation (det=1)
        d = np.sign(np.linalg.det(Q))
        Q[:, 0] *= d
        
        pts = pts @ Q.T
        
        # Optimize
        current_pts, current_radius = _optimize_single_start(pts, max_iters=1000, timeout=time_per_start)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = current_pts
            
        # Check time
        if time.time() - start_time > time_budget:
            break
            
    if best_pts is None:
        # Fallback to unoptimized Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts = _normalize(best_pts)
        
    return best_pts

# EVOLVE-BLOCK-END
