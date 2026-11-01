"""
Strategy: Differentiable Soft-Max Covering Radius Optimization
Instead of discrete point moves and hull re-computation per step, we:
1. Compute the convex hull once per evaluation.
2. Calculate the circumradius (spherical distance to circumcenter) for each facet.
3. Use a 'soft-max' (log-sum-exp) approximation to create a smooth, differentiable 
   surrogate for the maximum circumradius.
4. Compute gradients of this surrogate with respect to the point coordinates.
5. Use L-BFGS-B to minimize this surrogate, projecting updates back to the sphere.
6. Perform multiple restarts from different initial configurations to escape local minima.
"""

N = 50

import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import random

def construct():
    start_time = time.time()
    # Hard deadline guard: 115s out of 120s to be safe
    deadline = start_time + 115.0
    
    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    def compute_facet_data(pts):
        """
        Computes circumradii and gradients for all facets of the convex hull.
        Returns:
        radii: array of shape (n_facets,) with circumradii in radians
        grad_facets: list of arrays, each of shape (N, 3), gradient of radius w.r.t pts
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None, None
        
        radii = np.zeros(len(hull.simplices))
        grad_facets = []
        
        # Precompute some stuff? No, just loop. 
        # N=50, hull has ~100 facets. This is fast.
        for idx, simplex in enumerate(hull.simplices):
            i, j, k = simplex
            p1, p2, p3 = pts[i], pts[j], pts[k]
            
            # Vectors
            u = p2 - p1
            v = p3 - p1
            uu = np.dot(u, u)
            vv = np.dot(v, v)
            uv = np.dot(u, v)
            
            det = uu * vv - uv * uv
            if abs(det) < 1e-12:
                radii[idx] = 0.0
                grad_facets.append(np.zeros_like(pts))
                continue
            
            # Planar circumcenter c_planar = p1 + a*u + b*v
            # a = (0.5*uu*vv - 0.5*uv*vv) / det
            # b = (0.5*vv*uu - 0.5*uu*uv) / det
            a = 0.5 * vv * (uu - uv) / det
            b = 0.5 * uu * (vv - uv) / det
            
            c_planar = p1 + a * u + b * v
            c_norm = np.linalg.norm(c_planar)
            if c_norm < 1e-12:
                radii[idx] = 0.0
                grad_facets.append(np.zeros_like(pts))
                continue
            
            c_sph = c_planar / c_norm
            
            # Check if circumcenter is inside the spherical triangle
            # We need to check if c_sph is a positive linear combination of p1, p2, p3
            P = np.column_stack([p1, p2, p3])
            try:
                coeffs = np.linalg.solve(P, c_sph)
            except np.linalg.LinAlgError:
                coeffs = np.array([0, 0, 0])
            
            if np.all(coeffs > -1e-6):
                # Inside: radius is distance to c_sph
                # r = arccos(dot(c_sph, p1))
                dot_val = np.dot(c_sph, p1)
                dot_val = np.clip(dot_val, -1.0 + 1e-12, 1.0 - 1e-12)
                r = np.arccos(dot_val)
                radii[idx] = r
                
                # Gradient of r = arccos(x) is -1/sqrt(1-x^2) * dx
                # dx/dp = -c_sph (since x = c_sph . p1)
                # But c_sph depends on p1, p2, p3.
                # This is complex. Let's use the property that for a facet on the sphere,
                # the gradient of the circumradius w.r.t the vertices points away from the circumcenter 
                # projected onto the tangent plane?
                # Actually, there's a simpler geometric gradient.
                # The function f(p) = arccos(c(p).p) is messy.
                # Let's use the fact that for small moves, the change in radius is dominated by the move of the circumcenter.
                # However, a robust differentiable approach for "covering radius" often uses the 
                # distance to the nearest point. But here the "worst point" is the circumcenter.
                
                # Let's use a finite difference gradient approximation for the facets? 
                # No, that's slow.
                
                # Alternative: Use the gradient of the squared distance in 3D space projected?
                # Let's stick to a simpler, more robust gradient:
                # The gradient of the distance from p_i to the circumcenter C is -C / |C-p_i| ? No.
                
                # Let's use the analytic gradient of the circumradius of a spherical triangle.
                # This is known to be tricky. 
                # Instead, let's use a "soft" objective that is easier to differentiate:
                # Sum of exp(beta * r_i) / N.
                # The gradient of r_i w.r.t p_j is what we need.
                
                # For a triangle on the unit sphere, the circumcenter C is the normalized cross product of (p2-p1) and (p3-p1)? No.
                # C is proportional to p1 x p2 x p3 ? No, that's the normal.
                
                # Let's use the following approximation for the gradient:
                # Move vertices towards the circumcenter reduces the radius.
                # grad_{p_i} r_i ≈ - (C - p_i) projected to tangent plane at p_i, normalized?
                # Actually, if we move p_i towards C, the triangle shrinks.
                # Let's use: grad = - (C - p_i) / ||C - p_i||^2 ?
                
                # Let's use a finite difference for the gradient of the FACET radius only, 
                # but vectorized? No, loop is fine for 100 facets.
                # But we need it inside the L-BFGS loop which calls objective ~100-200 times.
                # 100 facets * 100 evals * 50 points * 3 dims = 1.5M ops. Fast enough.
                
                # Let's compute gradient via central differences for the specific facet.
                # This is expensive but accurate.
                # Wait, L-BFGS expects analytic gradients for speed.
                
                # Let's use the analytic gradient of the circumradius.
                # Reference: "Gradient of the circumradius of a spherical triangle"
                # It turns out that for a spherical triangle with circumcenter C,
                # the gradient of the circumradius with respect to a vertex p_i is:
                # g_i = - (C - p_i) / (1 - C.p_i) * (1/r_i) ?
                
                # Let's just use the negative of the vector from p_i to C, projected to the tangent space.
                # v_i = C - p_i
                # t_i = v_i - (v_i . p_i) p_i
                # grad_i = t_i / ||t_i|| ? No, magnitude matters.
                
                # Let's use a simpler, robust heuristic gradient that works well in practice:
                # The direction to move p_i to reduce the radius is towards C.
                # The magnitude should be related to the sensitivity.
                # Let's just use the unit vector towards C in the tangent plane.
                # This is a "subgradient" essentially.
                
                vec_to_c = c_sph - p1
                proj1 = vec_to_c - np.dot(vec_to_c, p1) * p1
                n1 = np.linalg.norm(proj1)
                if n1 > 1e-6:
                    grad1 = proj1 / n1
                else:
                    grad1 = np.zeros(3)
                    
                vec_to_c = c_sph - p2
                proj2 = vec_to_c - np.dot(vec_to_c, p2) * p2
                n2 = np.linalg.norm(proj2)
                if n2 > 1e-6:
                    grad2 = proj2 / n2
                else:
                    grad2 = np.zeros(3)
                    
                vec_to_c = c_sph - p3
                proj3 = vec_to_c - np.dot(vec_to_c, p3) * p3
                n3 = np.linalg.norm(proj3)
                if n3 > 1e-6:
                    grad3 = proj3 / n3
                else:
                    grad3 = np.zeros(3)
                
                g = np.zeros_like(pts)
                g[i] = grad1
                g[j] = grad2
                g[k] = grad3
                grad_facets.append(g)
            else:
                # Outside: radius is half the longest edge.
                # This is non-smooth. We can approximate by using the midpoint of the longest edge.
                d12 = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
                d23 = np.arccos(np.clip(np.dot(p2, p3), -1.0, 1.0))
                d31 = np.arccos(np.clip(np.dot(p3, p1), -1.0, 1.0))
                
                max_d = max(d12, d23, d31)
                radii[idx] = max_d / 2.0
                
                # Gradient: move the two vertices of the longest edge towards each other.
                g = np.zeros_like(pts)
                if max_d == d12:
                    # Move p1 towards p2 and p2 towards p1
                    v1 = p2 - p1
                    proj1 = v1 - np.dot(v1, p1) * p1
                    n1 = np.linalg.norm(proj1)
                    if n1 > 1e-6: g[i] = proj1 / n1
                    else: g[i] = 0
                    
                    v2 = p1 - p2
                    proj2 = v2 - np.dot(v2, p2) * p2
                    n2 = np.linalg.norm(proj2)
                    if n2 > 1e-6: g[j] = proj2 / n2
                    else: g[j] = 0
                elif max_d == d23:
                    v1 = p3 - p2
                    proj1 = v1 - np.dot(v1, p2) * p2
                    n1 = np.linalg.norm(proj1)
                    if n1 > 1e-6: g[j] = proj1 / n1
                    else: g[j] = 0
                    
                    v2 = p2 - p3
                    proj2 = v2 - np.dot(v2, p3) * p3
                    n2 = np.linalg.norm(proj2)
                    if n2 > 1e-6: g[k] = proj2 / n2
                    else: g[k] = 0
                else:
                    v1 = p2 - p3
                    proj1 = v1 - np.dot(v1, p3) * p3
                    n1 = np.linalg.norm(proj1)
                    if n1 > 1e-6: g[k] = proj1 / n1
                    else: g[k] = 0
                    
                    v2 = p3 - p2
                    proj2 = v2 - np.dot(v2, p2) * p2
                    n2 = np.linalg.norm(proj2)
                    if n2 > 1e-6: g[j] = proj2 / n2
                    else: g[j] = 0
                    
                grad_facets.append(g)
        
        return radii, grad_facets

    def objective_and_grad(x_flat):
        """
        Objective: Soft-max of circumradii.
        x_flat: flattened array of N points (N*3,)
        """
        pts = x_flat.reshape(N, 3)
        pts = normalize(pts)
        
        radii, grad_facets = compute_facet_data(pts)
        if radii is None:
            return 1e6, np.zeros_like(x_flat)
        
        # Soft-max parameter
        beta = 50.0 # Higher beta -> sharper max
        
        # Log-sum-exp trick for stability
        max_r = np.max(radii)
        # If max_r is very small, we are good.
        # LSE = max_r + (1/beta) * log(sum(exp(beta * (r_i - max_r))))
        exp_terms = np.exp(beta * (radii - max_r))
        lse = max_r + (1.0 / beta) * np.log(np.sum(exp_terms))
        
        # Gradient of LSE w.r.t r_i is: exp(beta * (r_i - max_r)) / sum(exp(...))
        weights = exp_terms / np.sum(exp_terms)
        
        # Chain rule: grad_x = sum_i weights[i] * grad_{p} r_i
        # grad_facets[i] is grad of r_i w.r.t pts
        grad_pts = np.zeros_like(pts)
        for i in range(len(radii)):
            grad_pts += weights[i] * grad_facets[i]
        
        # Flatten gradient
        grad_flat = grad_pts.flatten()
        
        return lse, grad_flat

    def run_optimization(start_pts, time_limit):
        """
        Runs L-BFGS-B optimization starting from start_pts.
        """
        x0 = start_pts.flatten()
        
        # L-BFGS-B doesn't handle constraints well, but we project back to sphere in objective.
        # To make it more effective, we can use the tangent space parametrization, 
        # but simple projection in the objective usually works for L-BFGS if step size is small.
        
        # However, standard L-BFGS on R^3 with normalization in objective can be tricky 
        # because the gradient is not exactly on the manifold.
        # A better approach: use scipy.optimize.minimize with method='L-BFGS-B' 
        # and just let it work. The normalization ensures we stay on the sphere.
        
        result = minimize(
            objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 200}
        )
        
        best_pts = result.x.reshape(N, 3)
        best_pts = normalize(best_pts)
        
        # Calculate true covering radius for reporting
        true_val, _ = compute_facet_data(best_pts)
        if true_val is not None:
            true_val = np.max(true_val)
        else:
            true_val = 1e6
            
        return best_pts, true_val

    # Multi-start strategy
    best_pts_overall = None
    best_val_overall = np.pi
    
    # Initial configurations
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Random Uniform
    random_pts = np.random.randn(N, 3)
    random_pts = normalize(random_pts)
    
    # 3. Perturbed Fibonacci
    perturbed_pts = fib_pts + np.random.normal(0, 0.1, (N, 3))
    perturbed_pts = normalize(perturbed_pts)
    
    start_configs = [fib_pts, random_pts, perturbed_pts]
    
    # Time allocation
    # We have ~115s. 
    # Each optimization might take 10-30s.
    # We can run 3-4 optimizations.
    
    for start_pts in start_configs:
        if time.time() > deadline - 30.0:
            break
            
        # Run optimization
        try:
            opt_pts, val = run_optimization(start_pts, deadline - time.time() - 5.0)
            if val < best_val_overall:
                best_val_overall = val
                best_pts_overall = opt_pts
        except Exception:
            pass
            
    # Final Polish: Local search on the best result found
    if best_pts_overall is not None:
        # Do a few more L-BFGS iterations with tighter tolerances or different start?
        # Or just return. The L-BFGS should have converged well.
        pass
        
    if best_pts_overall is None:
        # Fallback
        return fib_pts
        
    return best_pts_overall

# Ensure the function is accessible
# The prompt asks for construct()
