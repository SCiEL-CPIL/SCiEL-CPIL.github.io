"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # 1. Heuristic Initialization: Fibonacci Sphere
    # This provides a very uniform distribution and a good starting point.
    def fibonacci_sphere(n):
        points = np.zeros((n, 3))
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        r = np.sqrt(1.0 - z * z)
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        points[:, 0] = r * np.cos(phi)
        points[:, 1] = r * np.sin(phi)
        points[:, 2] = z
        return points

    # 2. Exact Covering Radius Calculation
    # The covering radius is the maximum distance from any point on the sphere to the nearest point in the set.
    # This is determined by the circumcenters of the facets of the convex hull of the points.
    # For a facet defined by points p1, p2, ..., pk, the circumcenter c is the point on the sphere 
    # equidistant to p1..pk. The distance from c to the facet points is the circumradius.
    # We need the maximum such distance over all facets.
    
    def _covering_radius(points):
        try:
            hull = ConvexHull(points)
            max_rad = 0.0
            for simplex in hull.simplices:
                # simplex contains indices of points forming a triangle (facet)
                p1 = points[simplex[0]]
                p2 = points[simplex[1]]
                p3 = points[simplex[2]]
                
                # Find circumcenter of the spherical triangle p1, p2, p3
                # The circumcenter c satisfies c.p1 = c.p2 = c.p3 = cos(alpha)
                # and |c| = 1.
                # c is in the plane defined by p1, p2, p3? No, c is on the sphere.
                # The vector c is a linear combination of p1, p2, p3? 
                # Actually, c is the intersection of the sphere with the line passing through the origin 
                # and the circumcenter of the planar triangle formed by p1,p2,p3? 
                # No. The circumcenter of a spherical triangle is the point on the sphere 
                # whose angular distance to the vertices is equal.
                # Let c = a*p1 + b*p2 + c*p3.
                # We want c.p1 = c.p2 = c.p3.
                # Let d = c.p1.
                # c is in the span of p1, p2, p3.
                # We can solve for the coefficients using the fact that 
                # (c - d*p1) is orthogonal to p1? No.
                
                # A robust way:
                # The circumcenter c is proportional to the cross product of (p2-p1) and (p3-p1)? No.
                # Let's use the property that c is orthogonal to (p2-p1) and (p3-p1) in the tangent plane?
                # Actually, c lies on the normal to the plane of the triangle? No.
                
                # Let's use the linear system:
                # c . p1 = c . p2 => c . (p1 - p2) = 0
                # c . p2 = c . p3 => c . (p2 - p3) = 0
                # c . c = 1
                
                # Let u = p1 - p2, v = p2 - p3.
                # c is orthogonal to u and v? No, c . (p1-p2) = 0 means c is in the plane perpendicular to p1-p2.
                # This implies c is in the intersection of two planes.
                # The normal to the plane of the triangle (in R3) is n = (p2-p1) x (p3-p1).
                # The circumcenter of the planar triangle lies on the line t*n.
                # But the spherical circumcenter is not simply the normalized n.
                
                # Correct derivation:
                # c = alpha * (p1 x p2) x (p2 x p3) ... no.
                # Let's solve the linear system for c in the basis of p1, p2, p3.
                # c = x*p1 + y*p2 + z*p3
                # c.p1 = x + y(p2.p1) + z(p3.p1)
                # c.p2 = x(p1.p2) + y + z(p3.p2)
                # c.p3 = x(p1.p3) + y(p2.p3) + z
                # We want c.p1 = c.p2 = c.p3.
                # So:
                # x(1 - p1.p2) + y(p2.p1 - 1) + z(p3.p1 - p3.p2) = 0  (Eq 1: c.p1 - c.p2 = 0)
                # x(p1.p2 - p1.p3) + y(p2.p3 - p2.p2) + z(1 - p3.p2) = 0  (Eq 2: c.p2 - c.p3 = 0)
                
                # Let A = [[1 - p1.p2, p1.p2 - 1, p1.p3 - p2.p3],
                #          [p1.p2 - p1.p3, p1.p3 - 1, p2.p3 - 1]]
                # Wait, the variables are x,y,z.
                # Eq 1: x(1 - p1.p2) + y(p1.p2 - 1) + z(p1.p3 - p2.p3) = 0
                # Eq 2: x(p1.p2 - p1.p3) + y(p1.p3 - 1) + z(p2.p3 - 1) = 0
                
                # This is a 2x3 system. We have 2 equations for 3 unknowns.
                # We also need |c|=1.
                # However, we can just find the vector c that is orthogonal to (p1-p2) and (p2-p3) in the sense of dot products?
                # c . (p1 - p2) = 0
                # c . (p2 - p3) = 0
                # This means c is orthogonal to the vectors (p1-p2) and (p2-p3).
                # So c is parallel to the cross product (p1-p2) x (p2-p3).
                # Let's verify.
                # If c = k * ((p1-p2) x (p2-p3)), then c . (p1-p2) = 0 and c . (p2-p3) = 0.
                # This implies c.p1 = c.p2 and c.p2 = c.p3. So c.p1 = c.p2 = c.p3.
                # This is the circumcenter!
                
                v1 = p1 - p2
                v2 = p2 - p3
                c_vec = np.cross(v1, v2)
                
                if np.linalg.norm(c_vec) < 1e-10:
                    # Degenerate case, points are collinear or coincident
                    continue
                    
                c = c_vec / np.linalg.norm(c_vec)
                
                # Calculate angular distance from c to p1
                # dot = c . p1
                # dist = arccos(dot)
                dot_val = np.dot(c, p1)
                # Clip to [-1, 1] to avoid numerical issues in arccos
                dot_val = np.clip(dot_val, -1.0, 1.0)
                rad = np.arccos(dot_val)
                
                if rad > max_rad:
                    max_rad = rad
                    
            return max_rad
        except Exception:
            # If hull fails (e.g., points in a hemisphere or degenerate), return a large value
            return np.pi

    # 3. Objective Function for Optimization
    # We want to minimize the covering radius.
    # To make the objective smooth for L-BFGS-B, we can minimize the radius directly.
    # However, arccos has a singularity at +-1. 
    # An alternative is to minimize the maximum dot product of the circumcenters?
    # No, minimizing the radius is equivalent to maximizing the minimum "height" of the circumcenters?
    # Let's stick to minimizing the radius in degrees or radians.
    # For numerical stability in the optimizer, we can minimize the radius in radians.
    
    def objective_flat(flat_points):
        points = flat_points.reshape(-1, 3)
        # Normalize points to ensure they stay on the sphere during optimization steps
        # L-BFGS-B works on unconstrained space, so we must project back or use a penalty.
        # Better: use a representation that is naturally on the sphere?
        # Or simply normalize inside the objective. This creates a non-smooth objective at the origin,
        # but since we start on the sphere and steps are small, it's usually fine.
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        points = points / norms
        
        rad = _covering_radius(points)
        return rad

    # 4. Multi-start Optimization
    # Start with Fibonacci sphere.
    # We will run L-BFGS-B from this start.
    # Since the landscape is non-convex, we might get stuck.
    # We can try a few perturbations if time permits.
    
    best_points = fibonacci_sphere(N)
    best_rad = _covering_radius(best_points)
    
    # Convert to flat array for optimizer
    x0 = best_points.flatten()
    
    # Run L-BFGS-B
    # We don't have gradients, so L-BFGS-B will use numerical differentiation.
    # This is expensive: 300 variables -> 300+ evaluations per gradient step.
    # Each evaluation involves ConvexHull (O(N log N) or O(N^2)) and arccos.
    # For N=50, this is fast.
    # However, numerical gradients are noisy.
    
    # Alternative: Use SLSQP or COBYLA?
    # L-BFGS-B with numerical gradients is often robust enough for this scale.
    
    try:
        res = minimize(
            objective_flat, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
        )
        
        if res.success or res.fun < best_rad:
            new_points = res.x.reshape(-1, 3)
            # Normalize
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            new_points = new_points / norms
            new_rad = _covering_radius(new_points)
            
            if new_rad < best_rad:
                best_rad = new_rad
                best_points = new_points
                
    except Exception:
        pass
    
    # If we have time, try a second start with a slightly perturbed Fibonacci sphere
    if time.time() < deadline - 20:
        # Perturb the initial points slightly
        noise = np.random.normal(0, 0.01, (N, 3))
        x0_pert = (best_points + noise).flatten()
        
        try:
            res2 = minimize(
                objective_flat, 
                x0_pert, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            
            if res2.success or res2.fun < best_rad:
                new_points = res2.x.reshape(-1, 3)
                norms = np.linalg.norm(new_points, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                new_points = new_points / norms
                new_rad = _covering_radius(new_points)
                
                if new_rad < best_rad:
                    best_rad = new_rad
                    best_points = new_points
        except Exception:
            pass

    # Final check and return
    # Ensure valid output
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
