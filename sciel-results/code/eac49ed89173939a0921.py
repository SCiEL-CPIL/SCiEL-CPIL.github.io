import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

# EVOLVE-BLOCK-START
def _fibonacci_sphere(n):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_circumcenters_and_radii(points):
    """
    Compute the convex hull, find the circumcenters of the facets,
    and return the circumradii (angular distances) and the mapping 
    from points to facets.
    """
    # Convex hull of the points on the sphere
    # Note: Points must be in general position to avoid degeneracy.
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull fails (e.g., coplanar points)
        # Return large radii to indicate failure
        return np.full(len(points), np.pi), np.zeros((len(points), 0), dtype=int)

    facets = hull.simplices
    n_facets = len(facets)
    circumcenters = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    # Precompute inverse for speed? No, simple loop is fine for 50 points.
    for idx, facet in enumerate(facets):
        p1, p2, p3 = points[facet[0]], points[facet[1]], points[facet[2]]
        
        # Circumcenter of triangle p1, p2, p3 in 3D
        # Formula: c = (|p2|^2 (p3-p1) x (p3-p2) + |p3|^2 (p1-p2) x (p1-p3) + |p1|^2 (p2-p3) x (p2-p1)) / (2 * |...|)
        # Since points are on unit sphere, |p1|^2 = 1, etc.
        # Simplified formula for unit sphere points:
        # c = (p1 + p2 + p3) * something? No.
        
        # Standard circumcenter calculation
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross_sq = np.dot(cross, cross)
        
        if norm_cross_sq < 1e-12:
            # Degenerate triangle
            circumcenters[idx] = p1
            radii[idx] = np.pi
            continue
            
        # The circumcenter C satisfies:
        # 2 * C * v1 = |p2|^2 - |p1|^2 = 0
        # 2 * C * v2 = |p3|^2 - |p1|^2 = 0
        # Since |p1|=|p2|=|p3|=1, the circumcenter lies on the line through the origin and the vector cross(v1, v2).
        # Actually, for points on a sphere centered at origin, the circumcenter of the triangle formed by 3 points on the sphere
        # is NOT necessarily on the line from origin to centroid.
        # Wait, the circumcenter of the *spherical* triangle is the normalization of the 3D circumcenter of the *planar* triangle?
        # No. The covering radius is the max angular distance to the nearest point.
        # The "worst" point in a spherical triangle (Voronoi cell) is the circumcenter of the Delaunay triangle.
        # The Delaunay triangle vertices are p1, p2, p3.
        # The circumcenter of the spherical triangle is the point on the sphere equidistant to p1, p2, p3.
        # This point is the normalization of the 3D circumcenter of the triangle p1, p2, p3 ONLY IF the triangle is acute?
        # Actually, the point on the sphere equidistant to p1, p2, p3 is the normalization of the vector:
        # c = p1 * (p2 x p3) + p2 * (p3 x p1) + p3 * (p1 x p2) ... no.
        
        # Let's solve: c . p1 = c . p2 = c . p3 = cos(theta).
        # c = a * p1 + b * p2 + c * p3.
        # Due to symmetry, if we want the point equidistant, it's the normalization of the sum of the vectors? No.
        
        # Correct formula for circumcenter of spherical triangle (vertices on unit sphere):
        # It is the normalization of the vector:
        # v = p1 * (p2 x p3) + p2 * (p3 x p1) + p3 * (p1 x p2)
        # Let's verify.
        # (p1 x p2) x p3 = p2 (p1.p3) - p3 (p1.p2).
        # The vector orthogonal to p1 and p2 is p1 x p2.
        # The circumcenter C satisfies C . p1 = C . p2 = C . p3.
        # C is in the plane of p1, p2, p3? No, C is on the sphere.
        # The plane containing p1, p2, p3 does not pass through the origin (usually).
        # The normal to the plane is n = (p2-p1) x (p3-p1).
        # The line from the origin to the circumcenter of the *planar* triangle is parallel to n.
        # Let C_planar be the circumcenter of the planar triangle p1, p2, p3.
        # Then the spherical circumcenter is normalize(C_planar).
        
        # Calculate C_planar:
        # C_planar = p1 + alpha * v1 + beta * v2
        # alpha = (|p2|^2 (v2 . v2) - |p3|^2 (v1 . v2)) / (2 * (v1 . v1 v2 . v2 - (v1 . v2)^2)) ?
        
        # Easier:
        # C_planar is the intersection of the plane and the line passing through the origin and the normal? No.
        # The circumcenter of the planar triangle lies on the normal line to the plane passing through the origin?
        # Only if the triangle is centered at the origin? No.
        # The circumcenter of the planar triangle p1, p2, p3 is given by:
        # C = ( |p2|^2 (p3-p1) x (p3-p2) + |p3|^2 (p1-p2) x (p1-p3) + |p1|^2 (p2-p3) x (p2-p1) ) / ( 2 * ( (p2-p1) x (p3-p1) ) . (p2-p1) x (p3-p1) ) ? No.
        
        # Standard formula:
        # d = 2 * ( p1 . (p2 x p3) )
        # If d is close to 0, points are coplanar with origin.
        # C_planar = ( |p2|^2 (p3-p1) x (p3-p2) + ... ) is for 2D.
        
        # For 3D points on unit sphere:
        # The vector C = p1 * (p2 x p3) + p2 * (p3 x p1) + p3 * (p1 x p2) is orthogonal to the plane?
        # No, C is the circumcenter of the spherical triangle.
        # Let's check: C . p1 = p1.(p2xp3) + p2.(p3xp1)p1 + p3.(p1xp2)p1
        # = p1.(p2xp3) + (p2.p1)(p3.p1) - (p3.p1)(p2.p1) + ... 
        # Using vector triple product: a.(bxc) = b.(cxa) = c.(axb).
        # C . p1 = p1.(p2xp3) + p2.(p3xp1) + p3.(p1xp2)
        # This is the scalar triple product, which is the same for all permutations.
        # So C . p1 = C . p2 = C . p3.
        # Thus, C is equidistant from p1, p2, p3.
        # The spherical circumcenter is normalize(C).
        
        c_vec = p1 * np.dot(p2, np.cross(p3, p1)) + \
                p2 * np.dot(p3, np.cross(p1, p2)) + \
                p3 * np.dot(p1, np.cross(p2, p3))
        
        norm_c = np.linalg.norm(c_vec)
        if norm_c < 1e-12:
            circumcenters[idx] = p1
            radii[idx] = np.pi
            continue
            
        circumcenters[idx] = c_vec / norm_c
        # Radius is angle between circumcenter and any vertex
        cos_theta = np.dot(circumcenters[idx], p1)
        # Clip to [-1, 1] for numerical stability
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii[idx] = np.arccos(cos_theta)
        
    # Map points to facets
    # We need to know which facets contain which points for gradient calculation.
    # Create a list of facet indices for each point.
    point_facets = [[] for _ in range(len(points))]
    for f_idx, facet in enumerate(facets):
        for p_idx in facet:
            point_facets[p_idx].append(f_idx)
            
    return radii, point_facets, circumcenters, facets

def _objective_and_grad(points):
    """
    Calculate Log-Sum-Exp of circumradii and its gradient.
    """
    radii, point_facets, circumcenters, facets = _compute_circumcenters_and_radii(points)
    
    if len(radii) == 0:
        return np.pi, np.zeros_like(points)
        
    # Log-Sum-Exp approximation
    beta = 50.0
    # Handle numerical issues
    max_r = np.max(radii)
    # If radii are very small, LSE is approx max
    # exp(beta * (r - max_r)) to avoid overflow
    shifted = beta * (radii - max_r)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp < 1e-300:
        # Fallback
        obj = max_r
        weights = np.zeros_like(radii)
        weights[np.argmax(radii)] = 1.0
    else:
        obj = max_r + (1.0 / beta) * np.log(sum_exp)
        weights = exp_shifted / sum_exp
        
    # Gradient calculation
    # grad_obj = sum_k weights[k] * grad_k
    # grad_k is the gradient of radius[k] w.r.t points.
    # radius[k] = arccos( c_k . p_i ) for any i in facet k.
    # d/dp_i arccos(u) = -1/sqrt(1-u^2) * du/dp_i
    # u = c . p_i
    # du/dp_i = c - (c . p_i) p_i  (projection of c onto tangent plane at p_i)
    # Note: c is the circumcenter, which depends on p_i, p_j, p_k.
    # However, the derivative of the circumcenter w.r.t the vertices is complex.
    # Approximation: Assume c is fixed for the gradient step (first-order approximation).
    # This is a common heuristic in sphere packing optimizations.
    
    grad = np.zeros_like(points)
    
    for k in range(len(radii)):
        if weights[k] < 1e-10:
            continue
            
        c = circumcenters[k]
        facet = facets[k]
        # For each point in the facet, add the gradient contribution
        for p_idx in facet:
            p = points[p_idx]
            u = np.dot(c, p)
            u = np.clip(u, -1.0, 1.0)
            sin_theta = np.sqrt(1.0 - u * u)
            if sin_theta < 1e-10:
                continue
                
            # Gradient of angle w.r.t p
            # d(theta)/dp = - (c - u*p) / sin(theta)
            grad_vec = -(c - u * p) / sin_theta
            grad[p_idx] += weights[k] * grad_vec
            
    return obj, grad

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe
    
    # Initial configuration
    points = _fibonacci_sphere(N)
    
    best_points = points.copy()
    best_score = _objective_and_grad(best_points)[0]
    
    # Iterative Refinement
    # We will perform a sequence of gradient descent steps.
    # Step size decay.
    step_size = 0.1
    beta = 50.0 # For LSE
    
    # We can also try to increase beta over time to sharpen the max approximation
    # Start with lower beta for smoother landscape, increase later.
    
    it = 0
    while time.time() < deadline:
        it += 1
        obj, grad = _objective_and_grad(points)
        
        # Update best
        if obj < best_score:
            best_score = obj
            best_points = points.copy()
            
        # Normalize gradient?
        # The gradient is on the tangent plane.
        # Update: points_new = points - step_size * grad
        # Then normalize points_new to unit sphere.
        
        # Adaptive step size
        # If the objective increases, reduce step size.
        # Simple heuristic:
        new_points = points - step_size * grad
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_points = new_points / norms
        
        # Check if new points are valid (no coincident points)
        # Simple check: min distance
        # O(N^2) check
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(new_points[i] - new_points[j])
                if d < min_dist:
                    min_dist = d
                    if min_dist < 1e-5:
                        break
            if min_dist < 1e-5:
                break
        
        if min_dist < 1e-5:
            # Rejected
            step_size *= 0.5
            continue
        
        new_obj, _ = _objective_and_grad(new_points)
        
        if new_obj < obj:
            points = new_points
            # Increase step size slightly if improving
            step_size = min(step_size * 1.1, 0.5)
        else:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-5:
                # Try increasing beta to sharpen
                beta *= 1.5
                step_size = 0.05
                if beta > 1000:
                    break
                    
    return best_points
# EVOLVE-BLOCK-END
