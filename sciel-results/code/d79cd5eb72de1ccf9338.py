"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

def _compute_covering_radius(points):
    """
    Compute the covering radius of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest chosen point. This is equivalent to the maximum circumradius of the
    spherical triangles formed by the Delaunay triangulation of the points.
    
    Approximation: 
    1. Find the convex hull of the points.
    2. For each facet (triangle), find the circumcenter.
    3. The distance from the circumcenter to the nearest point in the facet is the local covering radius.
    4. The max over all facets is the global covering radius.
    
    Note: This assumes the points are in general position and not all in a hemisphere.
    """
    if len(points) < 3:
        return 0.0
    
    # Normalize points
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, fall back to max nearest neighbor distance / 2
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        return np.degrees(np.arccos(np.clip(1 - min_dists**2 / 2, -1, 1))) / 2.0

    max_radius_deg = 0.0
    for simplex in hull.simplices:
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Circumcenter of the triangle on the sphere
        # The circumcenter is the intersection of the sphere with the plane passing through the centroid and normal to the triangle plane?
        # Actually, the circumcenter of a spherical triangle is the point on the sphere equidistant from the 3 vertices.
        # It lies on the normal to the plane of the triangle passing through the origin? No, that's for planar.
        # For spherical triangle, the circumcenter is the normalized vector:
        # c = p1 x p2 + p2 x p3 + p3 x p1 ? No.
        
        # The circumcenter C satisfies |C - p1| = |C - p2| = |C - p3| = r.
        # C is in the span of p1, p2, p3.
        # Let C = a*p1 + b*p2 + c*p3.
        # |C|^2 = 1.
        # |C - p1|^2 = 1 - 2 a + |C|^2 - 2b(p1.p2) - 2c(p1.p3) ... this is getting complex.
        
        # Simpler: The circumcenter of a spherical triangle is the normalized vector:
        # v = (p1 x p2) * |p3 - (p3.p1_hat) p1 ... | 
        # Actually, the circumcenter is the normalized sum of the cross products of the edges?
        # Let's use the formula:
        # n = np.cross(p1, p2)
        # m = np.cross(p2, p3)
        # l = np.cross(p3, p1)
        # The circumcenter is proportional to n + m + l? No.
        
        # The circumcenter is the normalized vector:
        # c = (p1 + p2 + p3) is the centroid, not circumcenter.
        
        # Correct formula for circumcenter of spherical triangle (p1, p2, p3):
        # It is the normalized vector:
        # c = np.cross(np.cross(p1, p2), p3) ... no.
        
        # Let's solve the linear system for the coefficients a,b,c such that C = a p1 + b p2 + c p3
        # and |C - p1| = |C - p2| = |C - p3|.
        # This implies C . p1 = C . p2 = C . p3 = k.
        # So we need a vector C in the span of p1,p2,p3 that has equal dot products with p1,p2,p3.
        
        # Let G be the Gram matrix [[p1.p1, p1.p2, p1.p3], ...]
        # We want C . p_i = k for all i.
        # C = G x, where x = [a,b,c]^T.
        # (G x) . p_i = k.
        # This is equivalent to finding the vector in the span with equal projections.
        
        # A simpler geometric approach:
        # The circumcenter is the intersection of the sphere with the line passing through the origin and the "planar circumcenter" of the triangle formed by projecting p1,p2,p3 onto a plane? No.
        
        # Let's use the property that the circumcenter is the normalized vector:
        # v = np.cross(p1, p2) * np.sin(angle_3) + ...
        
        # Actually, we can just find the point on the sphere that maximizes the minimum distance to the 3 vertices? No, that's the incenter.
        # The circumcenter is the point that is *equidistant*.
        
        # Let's use the linear algebra method.
        # We want C such that C . p1 = C . p2 = C . p3.
        # Let C = a p1 + b p2 + c p3.
        # C . p1 = a + b (p1.p2) + c (p1.p3)
        # C . p2 = a (p1.p2) + b + c (p2.p3)
        # C . p3 = a (p1.p3) + b (p2.p3) + c
        
        # We want C . p1 = C . p2 = C . p3 = 1 (if C is normalized, the dot product is cos(theta)).
        # Wait, if C is the circumcenter, then |C - p_i| = r.
        # |C|^2 - 2 C.p_i + |p_i|^2 = r^2
        # 1 - 2 C.p_i + 1 = r^2
        # C.p_i = 1 - r^2/2 = constant.
        
        # So we need C in span(p1,p2,p3) such that C.p1 = C.p2 = C.p3.
        # Let M be the matrix with rows p1, p2, p3.
        # We want M^T x = k * [1,1,1]^T.
        # This is a system of 3 equations with 3 unknowns x.
        # M^T x = k * 1
        # We can set k=1 for the direction, then normalize C.
        # Let's solve M^T x = 1.
        
        M = np.vstack([p1, p2, p3])
        try:
            x = np.linalg.solve(M.T, np.ones(3))
            C = np.dot(M.T, x)
            C = C / np.linalg.norm(C)
            
            # Calculate distance from C to p1
            dot_prod = np.dot(C, p1)
            dot_prod = np.clip(dot_prod, -1.0, 1.0)
            angle = np.arccos(dot_prod)
            max_radius_deg = max(max_radius_deg, np.degrees(angle))
        except np.linalg.LinAlgError:
            # Fallback if linear system is singular
            pass
            
    return max_radius_deg

def construct():
    # Start time
    start_time = time.time()
    time_limit = 110.0  # Keep safety margin for 120s limit
    
    # Initial configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Track best
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Parameters for optimization
    s = 6.0  # Riesz exponent, higher is more repulsive
    eta = 0.05  # Initial step size
    eta_min = 0.001
    cooling = 0.999
    
    # Optimization loop
    while time.time() - start_time < time_limit:
        # Compute forces
        # dists is NxN matrix of distances
        dists = cdist(points, points)
        np.fill_diagonal(dists, 1.0)  # Avoid division by zero
        
        # Force on i is sum over j of -grad(1/d^s) = s * p_ij / d^(s+2)
        # p_ij = p_i - p_j
        # We can compute this efficiently using matrix ops?
        # Let's do it in a loop for clarity, or vectorize if possible.
        # Vectorized:
        # For each i, force_i = sum_j s * (p_i - p_j) / ||p_i - p_j||^(s+2)
        
        forces = np.zeros_like(points)
        for i in range(N):
            pi = points[i]
            pjs = points
            # diff = pi - pjs
            # dists_i = dists[i]
            # weights = s * dists_i ** (-(s+2))
            # weights[i] = 0
            # force_i = sum(pjs * weights[:, np.newaxis] * -1) ... wait
            # grad_i (1/||p_i - p_j||^s) = -s (p_i - p_j) / ||p_i - p_j||^(s+2)
            # We want to minimize energy, so we move in direction of -grad = s (p_i - p_j) / ||p_i - p_j||^(s+2)
            
            diffs = pi - pjs
            dists_i = dists[i]
            # Avoid division by zero
            dists_i_safe = np.where(dists_i < 1e-8, 1e-8, dists_i)
            weights = s * (dists_i_safe ** (-(s+2)))
            weights[i] = 0.0
            forces[i] = np.sum(weights[:, np.newaxis] * diffs, axis=0)
            
        # Update points
        points = points + eta * forces
        # Renormalize
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Check if we improved
        # Compute radius periodically to save time
        if int((time.time() - start_time) * 10) % 10 == 0: # Roughly every 0.1s
            current_radius = _compute_covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
                
        # Cool down
        eta *= cooling
        if eta < eta_min:
            # Restart with a larger step or random jitter?
            # Let's just break or reset eta
            eta = eta_min * 2
            # Add small noise to escape local minima
            noise = np.random.normal(0, 0.01, points.shape)
            points = points + noise
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis]

    return best_points
# EVOLVE-BLOCK-END
