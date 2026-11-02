import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Optimization Phase: Soft-Max Minimization
    # We optimize the 3D coordinates. The objective normalizes them.
    # Test points for soft-max approximation of the covering radius.
    # Use a dense Fibonacci spiral for test points.
    M = 20000
    j = np.arange(M) + 0.5
    z_t = 1.0 - 2.0 * j / M
    phi_t = np.pi * (1.0 + np.sqrt(5.0)) * j
    r_t = np.sqrt(1.0 - z_t * z_t)
    T = np.column_stack([r_t * np.cos(phi_t), r_t * np.sin(phi_t), z_t])
    
    beta = 10.0 # Temperature for soft-max
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 1e10 # Force termination
        
        P_opt = x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(P_opt, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        P_norm = P_opt / norms
        
        # Compute distances from T to P_norm
        # T is (M, 3), P_norm is (N, 3)
        # dist_sq[i, j] = |T_i - P_norm_j|^2
        # We want min_j dist_sq[i, j]
        # Using broadcasting: T[:, np.newaxis, :] - P_norm[np.newaxis, :, :]
        # This creates an (M, N, 3) array, which is 20000 * 130 * 3 * 8 bytes ~ 62 MB. Acceptable.
        diff = T[:, np.newaxis, :] - P_norm[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Min distance for each test point
        min_dist_sq = np.min(dist_sq, axis=1)
        
        # Convert to angular distance? 
        # angle = arccos(1 - 0.5 * dist_sq) approx sqrt(dist_sq) for small angles.
        # For soft-max, we can just use dist_sq or dist.
        # Using dist is more robust for the max behavior.
        min_dist = np.sqrt(min_dist_sq)
        
        # Soft-max: (1/beta) * log(sum(exp(beta * dist)))
        # Shift for numerical stability
        max_dist = np.max(min_dist)
        exp_term = np.exp(beta * (min_dist - max_dist))
        log_sum_exp = np.log(np.sum(exp_term)) + beta * max_dist
        soft_max_val = log_sum_exp / beta
        
        return soft_max_val

    def jac(x):
        # Finite difference is slow, but L-BFGS-B can estimate.
        # Providing analytical gradient is complex due to min/softmax.
        # Let L-BFGS-B estimate the gradient.
        return None

    # Run L-BFGS-B
    try:
        res = minimize(objective, P.flatten(), method='L-BFGS-B', jac=jac, 
                       options={'maxiter': 200, 'maxls': 50, 'ftol': 1e-8, 'gtol': 1e-5})
        if time.time() - start_time < time_limit - 10:
            P = res.x.reshape(N, 3)
            # Normalize
            P = P / np.linalg.norm(P, axis=1, keepdims=True)
    except Exception:
        pass

    # 3. Polish Phase: Delaunay Triangulation Refinement
    # We will iteratively find the worst facet and adjust its vertices.
    
    def get_circumradii(P_pts):
        """
        Compute angular circumradii for all facets of the convex hull.
        Returns array of radii and facet indices.
        """
        try:
            hull = ConvexHull(P_pts)
        except Exception:
            return np.array([]), np.array([])
        
        facets = hull.simplices
        radii = np.zeros(len(facets))
        
        for k, f in enumerate(facets):
            a, b, c = P_pts[f[0]], P_pts[f[1]], P_pts[f[2]]
            # Vector v = a x b + b x c + c x a
            v = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-10:
                radii[k] = np.pi
                continue
            u = v / v_norm
            # Dot product with any vertex gives cos(R)
            cos_R = np.dot(u, a)
            # Clip for numerical safety
            cos_R = np.clip(cos_R, -1.0, 1.0)
            radii[k] = np.arccos(cos_R)
            
        return radii, facets

    def polish(P_pts, max_iters=50):
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
                
            radii, facets = get_circumradii(P_pts)
            if len(radii) == 0:
                break
                
            max_idx = np.argmax(radii)
            max_radius = radii[max_idx]
            
            # If the radius is very small, we are done
            if max_radius < 0.15: # ~8.6 degrees, well below 11.3
                break
                
            # Get the vertices of the worst facet
            f = facets[max_idx]
            v0, v1, v2 = P_pts[f[0]], P_pts[f[1]], P_pts[f[2]]
            
            # We will perform a local search on these 3 vertices.
            # Objective: Minimize the maximum circumradius of the facets involving these vertices.
            # To keep it simple, we minimize the max radius of the specific facet and its neighbors.
            # For simplicity and speed, let's just minimize the radius of this facet 
            # while keeping points on the sphere and not crossing others too much.
            # A better approach: Use Nelder-Mead on the 3 points (6 vars) 
            # minimizing the max radius of all facets in a local neighborhood.
            
            # Identify neighbors: facets sharing an edge with f
            # This is complex. Let's just minimize the radius of this facet 
            # and rely on the global structure being robust.
            # Actually, moving vertices might break other facets.
            # Let's use a simple gradient-free method on the 3 points.
            
            def local_obj(x):
                # x is 6x1
                new_v0 = x[0:3]
                new_v1 = x[3:6]
                # Normalize
                n0 = np.linalg.norm(new_v0)
                n1 = np.linalg.norm(new_v1)
                if n0 < 1e-6 or n1 < 1e-6:
                    return 10.0
                new_v0 /= n0
                new_v1 /= n1
                
                # Create a modified P
                P_mod = P_pts.copy()
                P_mod[f[0]] = new_v0
                P_mod[f[1]] = new_v1
                # Note: We are only moving 2 vertices? 
                # Moving 3 vertices in a triangle on a sphere: 2 degrees of freedom relative to the 3rd?
                # Actually, we can move all 3.
                # Let's stick to moving all 3.
                # x is 9x1
                pass
                
            # Let's implement a simpler polish:
            # Randomly perturb the worst points and keep if improvement.
            # This is slow.
            
            # Alternative: Use the fact that the optimal points are "equidistant" in some sense.
            # Let's try a simple local search using Nelder-Mead on the 3 vertices.
            
            def local_objective(x):
                # x: 9 floats
                new_v0 = x[0:3]
                new_v1 = x[3:6]
                new_v2 = x[6:9]
                
                n0 = np.linalg.norm(new_v0)
                n1 = np.linalg.norm(new_v1)
                n2 = np.linalg.norm(new_v2)
                
                if n0 < 1e-6 or n1 < 1e-6 or n2 < 1e-6:
                    return 10.0
                
                new_v0 /= n0
                new_v1 /= n1
                new_v2 /= n2
                
                P_mod = P_pts.copy()
                P_mod[f[0]] = new_v0
                P_mod[f[1]] = new_v1
                P_mod[f[2]] = new_v2
                
                # Check for collisions? 
                # If points get too close, convex hull might fail or radii become unstable.
                # We can add a penalty for min distance.
                # For now, just compute max radius of the whole hull.
                # This is expensive (ConvexHull).
                # Optimization: Only compute radii for facets involving f[0], f[1], f[2].
                # We need the list of such facets.
                
                # To avoid expensive full hull computation, we can approximate.
                # But for robustness, let's just do the full hull if we have time.
                # ConvexHull for 130 points is fast (~ms).
                
                try:
                    radii_mod, _ = get_circumradii(P_mod)
                    if len(radii_mod) == 0:
                        return 10.0
                    return np.max(radii_mod)
                except Exception:
                    return 10.0

            x0 = np.concatenate([v0, v1, v2])
            
            # Nelder-Mead is good for small dimensional problems.
            try:
                res_local = minimize(local_objective, x0, method='Nelder-Mead', 
                                     options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
                if res_local.fun < max_radius - 1e-6:
                    P_pts[f[0]] = res_local.x[0:3]
                    P_pts[f[1]] = res_local.x[3:6]
                    P_pts[f[2]] = res_local.x[6:9]
                    # Normalize
                    P_pts[f[0]] /= np.linalg.norm(P_pts[f[0]])
                    P_pts[f[1]] /= np.linalg.norm(P_pts[f[1]])
                    P_pts[f[2]] /= np.linalg.norm(P_pts[f[2]])
            except Exception:
                pass

        return P_pts

    P = polish(P, max_iters=100)
    
    # Final normalization
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    
    return P

# EVOLVE-BLOCK-END
