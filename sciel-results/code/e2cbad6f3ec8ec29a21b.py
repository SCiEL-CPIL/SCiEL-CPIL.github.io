"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys

def _normalize(v):
    """Normalize vectors to unit sphere."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return v / norms

def _covering_radius(points):
    """Compute the covering radius in radians for a set of points on the sphere."""
    # Build convex hull of the points
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # fallback
    
    # For each facet (simplex), compute the circumcenter and circumradius
    # The covering radius is the max over all facets of the angular circumradius
    # where the circumcenter is on the same side as the facet's outward normal
    
    max_radius = 0.0
    
    for simplex in hull.simplices:
        pts = points[simplex]  # (d+1, 3)
        
        # Compute circumcenter of the simplex
        # For a simplex in R^3, the circumcenter is equidistant from all vertices
        # Solve: |c - p_i|^2 = |c - p_j|^2 for all i, j
        # This gives a linear system
        
        # Set up the system: 2*(p_i - p_j) · c = |p_i|^2 - |p_j|^2
        # Using the first point as reference
        ref = pts[0]
        A = []
        b = []
        for i in range(1, len(pts)):
            A.append(2.0 * (pts[i] - ref))
            b.append(np.dot(pts[i], pts[i]) - np.dot(ref, ref))
        
        A = np.array(A)
        b = np.array(b)
        
        # Solve for c in the plane of the simplex
        # The circumcenter lies in the affine hull of the simplex
        # We can parameterize: c = ref + sum_i alpha_i * (pts[i] - ref)
        # But it's easier to solve the full system with the constraint that c is in the affine hull
        
        # Actually, let's use a different approach:
        # The circumcenter c satisfies:
        #   2 * (pts[i] - pts[0]) · c = |pts[i]|^2 - |pts[0]|^2  for i = 1, ..., d
        # Plus the constraint that c is in the affine hull (which is automatically satisfied 
        # if we solve correctly, but we need d+1 equations for d+1 unknowns)
        
        # For a triangle (2-simplex) in 3D, we have 2 linear equations for c (3 unknowns),
        # so we need one more constraint. The circumcenter lies in the plane of the triangle.
        
        # Let's solve it properly:
        # c = ref + alpha*(pts[1]-ref) + beta*(pts[2]-ref) for a triangle
        # Then impose |c - pts[i]|^2 = |c - ref|^2 for i=1,2
        
        if len(pts) == 3:  # Triangle
            e1 = pts[1] - ref
            e2 = pts[2] - ref
            # c = ref + alpha*e1 + beta*e2
            # |c - pts[1]|^2 = |c - ref|^2
            # |ref + alpha*e1 + beta*e2 - pts[1]|^2 = |alpha*e1 + beta*e2|^2
            # |alpha*e1 + beta*e2 - e1|^2 = |alpha*e1 + beta*e2|^2
            # (alpha-1)^2|e1|^2 + beta^2|e2|^2 + 2*beta*(alpha-1)(e1·e2) = alpha^2|e1|^2 + beta^2|e2|^2 + 2*alpha*beta*(e1·e2)
            # -2*alpha*|e1|^2 + |e1|^2 + 2*alpha*beta*(e1·e2) - 2*beta*(e1·e2) = 2*alpha*beta*(e1·e2)
            # -2*alpha*|e1|^2 + |e1|^2 - 2*beta*(e1·e2) = 0
            # alpha*|e1|^2 + beta*(e1·e2) = |e1|^2 / 2
            
            # Similarly for pts[2]:
            # alpha*(e1·e2) + beta*|e2|^2 = |e2|^2 / 2
            
            M = np.array([[np.dot(e1, e1), np.dot(e1, e2)],
                          [np.dot(e1, e2), np.dot(e2, e2)]])
            rhs = np.array([np.dot(e1, e1) / 2.0, np.dot(e2, e2) / 2.0])
            
            det = np.linalg.det(M)
            if abs(det) < 1e-12:
                continue
            
            ab = np.linalg.solve(M, rhs)
            c = ref + ab[0] * e1 + ab[1] * e2
            
            # Check if c is on the same side as the outward normal
            # Outward normal of the facet
            n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            # The centroid of the simplex
            centroid = np.mean(pts, axis=0)
            
            # Check if c is on the outward side: (c - centroid) · n > 0
            if np.dot(c - centroid, n) <= 0:
                # Circumcenter is on the inward side, so this facet doesn't determine the covering radius
                # The covering radius for this facet is determined by the farthest point on the cap
                # which would be the antipode of the nearest vertex... actually, if the circumcenter
                # is inside, the cap is not "empty" in the sense that the sphere center is inside.
                # In this case, the angular radius is just the max distance from the vertices to the 
                # circumcenter projected, but actually the covering radius contribution is 0 if the
                # circumcenter is inside the hemisphere defined by the facet.
                # More precisely: if the circumcenter is on the inward side, the spherical cap
                # cut off by the great circle containing the facet has its center on the inward side,
                # so the farthest point on the sphere from the facet (on the outward side) is at
                # distance pi/2 from the great circle, i.e., the angular radius is pi/2 minus the
                # distance from the great circle to the circumcenter... 
                
                # Actually, let me think again. The covering radius is the max over all points y on 
                # the sphere of min_i angular_dist(y, p_i). For a facet, the worst point is the one
                # farthest from all three vertices. If the circumcenter is on the outward side, it's
                # the circumcenter itself (if it's on the sphere) or its projection. If on the inward
                # side, the worst point is the point on the great circle opposite to the circumcenter.
                
                # The angular distance from a point on the sphere to a vertex is the angle at the center.
                # The circumcenter c (not necessarily on the sphere) has angular distance to vertex i 
                # of arccos(c · p_i / |c|) if we normalize... no, the angular distance from y to p_i 
                # is arccos(y · p_i) for unit vectors.
                
                # If the circumcenter C (as a direction, normalized) is on the outward side, then
                # the point y = C/|C| on the sphere has angular distance arccos(C · p_i / |C|) to p_i.
                # This is the circumradius.
                
                # If C is on the inward side, the worst point is -C/|C| (antipode), and the angular
                # distance is arccos(-C · p_i / |C|) = pi - arccos(C · p_i / |C|).
                
                # But wait, -C/|C| might not be the worst. Let me reconsider.
                
                # For a triangle on the sphere, the Voronoi cell's farthest point from the three
                # vertices: if the circumcenter (of the planar triangle, projected to sphere) is
                # inside the spherical triangle, it's the circumcenter. Otherwise, it's on the boundary
                # (an edge midpoint or a vertex).
                
                # This is getting complicated. Let me use a simpler approach:
                # Just sample points on the sphere and find the max min-distance.
                continue
            else:
                # Circumcenter is on the outward side
                c_norm = np.linalg.norm(c)
                if c_norm < 1e-12:
                    continue
                c_unit = c / c_norm
                # Angular distance from c_unit to any vertex
                dot_vals = np.dot(pts, c_unit)
                dot_vals = np.clip(dot_vals, -1.0, 1.0)
                angle = np.arccos(dot_vals[0])  # All should be the same
                max_radius = max(max_radius, angle)
                
        elif len(pts) == 4:  # Tetrahedron
            # For a tetrahedron, the circumcenter is the unique point equidistant from all 4 vertices
            # Solve the linear system
            ref = pts[0]
            A = np.array([2.0 * (pts[i] - ref) for i in range(1, 4)])
            b = np.array([np.dot(pts[i], pts[i]) - np.dot(ref, ref) for i in range(1, 4)])
            
            det = np.linalg.det(A)
            if abs(det) < 1e-12:
                continue
            
            c = np.linalg.solve(A, b)
            
            # Check side
            centroid = np.mean(pts, axis=0)
            # Normal: use three edges from ref
            e1 = pts[1] - ref
            e2 = pts[2] - ref
            e3 = pts[3] - ref
            n = np.cross(e1, e2)
            # Make sure n points outward (away from centroid)
            if np.dot(n, centroid - ref) > 0:
                n = -n
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            if np.dot(c - centroid, n) <= 0:
                continue
            else:
                c_norm = np.linalg.norm(c)
                if c_norm < 1e-12:
                    continue
                c_unit = c / c_norm
                dot_vals = np.dot(pts, c_unit)
                dot_vals = np.clip(dot_vals, -1.0, 1.0)
                angle = np.arccos(dot_vals[0])
                max_radius = max(max_radius, angle)
    
    return max_radius

def _smooth_covering_radius(points, beta=50.0):
    """Smooth approximation of the covering radius using log-sum-exp."""
    # Create a grid of test points on the sphere
    # Use a moderately fine grid for the smooth objective
    n_theta = 60
    n_phi = 120
    
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to Cartesian
    test_points = np.zeros((THETA.size, 3))
    test_points[:, 0] = np.sin(THETA) * np.cos(PHI)
    test_points[:, 1] = np.sin(THETA) * np.sin(PHI)
    test_points[:, 2] = np.cos(THETA)
    
    # Compute angular distances from each test point to each configuration point
    # test_points: (M, 3), points: (N, 3)
    # dot product: (M, N)
    dots = np.dot(test_points, points.T)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    
    # For each test point, find the min angular distance
    min_angles = np.min(angles, axis=1)  # (M,)
    
    # Smooth max: (1/beta) * log(sum(exp(beta * x)))
    scaled = beta * min_angles
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smooth_max = log_sum_exp / beta
    
    return smooth_max

def _gradient_smooth_covering_radius(points, beta=50.0):
    """Gradient of the smooth covering radius with respect to points."""
    # This is complex; for now, use finite differences or a simpler approach
    # For efficiency, we'll use a simpler gradient estimate
    
    n_theta = 40
    n_phi = 80
    
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    test_points = np.zeros((THETA.size, 3))
    test_points[:, 0] = np.sin(THETA) * np.cos(PHI)
    test_points[:, 1] = np.sin(THETA) * np.sin(PHI)
    test_points[:, 2] = np.cos(THETA)
    
    dots = np.dot(test_points, points.T)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    
    min_idx = np.argmin(angles, axis=1)
    min_angles = np.min(angles, axis=1)
    
    # Soft assignment: use softmax over the inverse distances
    # For the smooth max, the gradient flows through the min
    # d/dx_j of min_i f_i(x) is tricky; use the soft-min
    
    # Soft-min: -1/beta * log(sum(exp(-beta * angles)))
    neg_scaled = -beta * angles
    max_neg = np.max(neg_scaled, axis=1, keepdims=True)
    exp_neg = np.exp(neg_scaled - max_neg)
    sum_exp = np.sum(exp_neg, axis=1, keepdims=True)
    weights = exp_neg / sum_exp  # (M, N) weights for soft-min
    
    # The soft-min value
    soft_min = -(max_neg + np.log(sum_exp)) / beta  # (M, 1)
    
    # Soft-max of soft-min
    sm_scaled = beta * soft_min.squeeze()
    sm_max = np.max(sm_scaled)
    exp_sm = np.exp(sm_scaled - sm_max)
    sum_exp_sm = np.sum(exp_sm)
    outer_weights = exp_sm / sum_exp_sm  # (M,)
    
    # Gradient: d/dp_j of sum_m outer_w_m * sum_i w_m_i * angle_m_i
    # angle_m_i = arccos(test_m · p_i)
    # d angle / d p_i = -test_m / sqrt(1 - (test_m · p_i)^2)
    
    grad = np.zeros_like(points)
    
    for m in range(len(outer_weights)):
        if outer_weights[m] < 1e-10:
            continue
        tw = outer_weights[m]
        tp = test_points[m]
        
        for i in range(N):
            wi = weights[m, i]
            if wi < 1e-10:
                continue
            pi = points[i]
            dot_val = np.dot(tp, pi)
            dot_val = np.clip(dot_val, -1.0 + 1e-10, 1.0 - 1e-10)
            sin_val = np.sqrt(1.0 - dot_val * dot_val)
            if sin_val < 1e-10:
                continue
            d_angle = -tp / sin_val
            grad[i] += tw * wi * d_angle
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # Phase 1: L-BFGS optimization with smooth objective
    def objective_and_grad(x):
        pts = _normalize(x.reshape(N, 3))
        val = _smooth_covering_radius(pts, beta=30.0)
        grad = _gradient_smooth_covering_radius(pts, beta=30.0)
        return val, grad.flatten()
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() > deadline - 20:
            return best_points
        
        points = _normalize(result.x.reshape(N, 3))
        radius = _covering_radius(points)
        
        if radius < best_radius:
            best_points = points.copy()
            best_radius = radius
            
    except Exception:
        pass
    
    # Phase 2: Projected gradient polish
    if time.time() < deadline - 10:
        lr = 0.01
        for step in range(200):
            if time.time() > deadline - 5:
                break
            
            pts = best_points.copy()
            grad = _gradient_smooth_covering_radius(pts, beta=50.0)
            
            # Project gradient onto tangent space
            grad_tan = grad - np.sum(pts * grad, axis=1, keepdims=True) * pts
            
            # Update
            pts = pts - lr * grad_tan
            pts = _normalize(pts)
            
            radius = _covering_radius(pts)
            if radius < best_radius:
                best_radius = radius
                best_points = pts.copy()
                lr = max(lr * 0.95, 0.001)
            else:
                lr = max(lr * 0.5, 0.001)
    
    return best_points
# EVOLVE-BLOCK-END
