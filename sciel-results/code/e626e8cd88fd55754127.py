"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    def compute_exact_covering_radius(pts):
        """
        Computes the exact covering radius in degrees using convex hull facets.
        """
        try:
            hull = ConvexHull(pts)
            max_r_rad = 0.0
            
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                
                # Compute circumcenter of the spherical triangle
                v1 = p0 - p1
                v2 = p0 - p2
                c_vec = np.cross(v1, v2)
                
                norm_c = np.linalg.norm(c_vec)
                if norm_c < 1e-10:
                    continue
                    
                c_vec /= norm_c
                
                # Determine correct hemisphere for circumcenter
                centroid = (p0 + p1 + p2) / 3.0
                centroid /= np.linalg.norm(centroid)
                
                if np.dot(c_vec, centroid) < 0:
                    c_vec = -c_vec
                    
                # Check if circumcenter is inside the spherical triangle
                n1 = np.cross(p1, p2)
                n2 = np.cross(p2, p0)
                n3 = np.cross(p0, p1)
                
                # Orient normals towards centroid
                if np.dot(centroid, n1) < 0: n1 = -n1
                if np.dot(centroid, n2) < 0: n2 = -n2
                if np.dot(centroid, n3) < 0: n3 = -n3
                
                if (np.dot(c_vec, n1) >= -1e-8 and 
                    np.dot(c_vec, n2) >= -1e-8 and 
                    np.dot(c_vec, n3) >= -1e-8):
                    
                    dist = np.arccos(np.clip(np.dot(c_vec, p0), -1.0, 1.0))
                    if dist > max_r_rad:
                        max_r_rad = dist
                        
            return np.degrees(max_r_rad)
        except Exception:
            return 180.0

    def soft_covering_energy_and_grad(pts):
        """
        Computes a smooth approximation of the covering radius and its gradient.
        Uses a softmax-like function to approximate the max angular distance to nearest point.
        f(pts) = sum_i (1/N) * log(sum_j exp(-beta * dist(i, j)))
        To maximize the minimum distance (minimize covering radius), we minimize the negative of this.
        Actually, covering radius is max_y min_i dist(y, i).
        We can approximate min_i dist(y, i) by a soft-min.
        Let d_ij = arccos(dot(p_i, p_j)).
        soft_min_dist = - (1/beta) * log( sum_j exp(-beta * d_ij) )
        Then we want to maximize this soft_min_dist over the sphere? No, we want to place points such that the *minimum* distance from any y to nearest p is large.
        Wait, the objective is to MINIMIZE the covering radius.
        Covering radius R = max_y min_i dist(y, p_i).
        
        A common proxy is to minimize the sum of squared distances to nearest neighbor? No, that's for packing.
        For covering, we want to fill space.
        
        Let's use the "max-min" formulation directly on the points? No, points are variables.
        
        Alternative: Minimize the "soft covering radius".
        R_soft = max_{grid} soft_min_dist(grid, pts).
        This is hard to differentiate with respect to pts if we take max over grid.
        
        Let's try a different energy:
        E = sum_i (1/|V_i|) * integral_{V_i} dist(x, p_i)^2 dx ?
        This is the Lloyd's algorithm energy (Voronoi mean squared error). Minimizing this tends to make cells equal size, which helps covering but doesn't directly minimize the max radius.
        
        However, Inspiration 1 suggests "smooth 'soft-max' covering energy".
        
        Let's define:
        For each point p_i, let d_i(y) be the distance to p_i.
        min_j d_j(y) is the distance to the nearest point.
        We want to maximize min_j d_j(y) for all y.
        
        A smooth approximation for min_j d_j(y) is:
        m(y) = - (1/beta) * log( sum_j exp(-beta * d_j(y)) )
        
        The covering radius is approximated by max_y m(y).
        Since we can't easily optimize over y in the gradient step, we can use a fixed set of test points (grid) and take the max of m(y) over the grid, or sum of exp(beta * m(y)).
        
        Let G be a set of grid points.
        E(pts) = log( sum_{y in G} exp( beta * m(y, pts) ) )
        This approximates beta * max_y m(y, pts).
        Minimizing E minimizes the soft covering radius.
        
        Gradient calculation:
        m(y) = -1/beta * log( sum_j exp(-beta * d_j(y)) )
        d_j(y) = arccos(clip(dot(y, p_j), -1, 1))
        
        Let S = sum_j exp(-beta * d_j(y))
        m = -1/beta * log(S)
        
        dm/dp_k = -1/beta * (1/S) * dS/dp_k
        dS/dp_k = exp(-beta * d_k(y)) * (-beta) * dd_k/dp_k
        dd_k/dp_k = - y / sqrt(1 - dot(y, p_k)^2)  (gradient of arccos(x) is -1/sqrt(1-x^2))
        Actually, d/dp arccos(y.p) = - y / sqrt(1 - (y.p)^2)
        
        So dS/dp_k = exp(-beta * d_k) * (-beta) * (- y / sqrt(1 - (y.p_k)^2))
                   = beta * exp(-beta * d_k) * y / sqrt(1 - (y.p_k)^2)
        
        dm/dp_k = -1/beta * (1/S) * (beta * exp(-beta * d_k) * y / sqrt(1 - (y.p_k)^2))
                 = - exp(-beta * d_k) / S * y / sqrt(1 - (y.p_k)^2)
                 = - w_k(y) * y / sqrt(1 - (y.p_k)^2)
        
        Where w_k(y) = exp(-beta * d_k(y)) / S is the softmax weight.
        
        Now E = log( sum_{y in G} exp(beta * m(y)) )
        Let T = sum_{y in G} exp(beta * m(y))
        dE/dp_k = (1/T) * sum_{y in G} exp(beta * m(y)) * beta * dm/dp_k(y)
                 = beta * sum_{y in G} P(y) * dm/dp_k(y)
                 where P(y) = exp(beta * m(y)) / T is the softmax over grid points.
        
        So grad_k = beta * sum_{y in G} P(y) * ( - w_k(y) * y / sqrt(1 - (y.p_k)^2) )
        
        This is computable!
        """
        beta = 20.0  # Sharpness of the max approximation
        
        # Generate a fixed grid for evaluation
        # Use a smaller grid for speed, e.g., 5000 points
        n_grid = 5000
        i = np.arange(n_grid) + 0.5
        z = 1.0 - 2.0 * i / n_grid
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Compute distances from grid to all points
        # D shape: (n_grid, N)
        dots = grid_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # Compute m(y) = -1/beta * log(sum_j exp(-beta * d_j))
        # For numerical stability, subtract max
        # log_sum_exp(-beta * dists)
        max_dists = np.max(dists, axis=1, keepdims=True)
        exp_terms = np.exp(-beta * (dists - max_dists))
        log_S = np.log(np.sum(exp_terms, axis=1)) + (-beta * max_dists).squeeze(axis=1)
        m = -1.0 / beta * log_S
        
        # Compute E = log(sum exp(beta * m))
        max_m = np.max(m)
        exp_m = np.exp(beta * (m - max_m))
        T = np.sum(exp_m)
        P = exp_m / T  # Weights for grid points
        
        energy = np.log(T) + beta * max_m
        
        # Compute Gradient
        # grad_k = beta * sum_y P(y) * ( - w_k(y) * y / sqrt(1 - (y.p_k)^2) )
        # w_k(y) = exp(-beta * d_k) / S
        # Note: exp(-beta * d_k) / S = exp(-beta * (d_k - max_d)) * exp(-beta * max_d) / S
        # S = exp(-beta * max_d) * sum_j exp(-beta * (d_j - max_d))
        # So w_k(y) = exp_terms[y, k] / sum_j exp_terms[y, j]
        
        sum_exp_terms = np.sum(exp_terms, axis=1, keepdims=True)
        W = exp_terms / sum_exp_terms  # Shape (n_grid, N)
        
        # Denominator term: sqrt(1 - dots^2)
        # dots shape (n_grid, N)
        denom = np.sqrt(1.0 - dots**2)
        denom[denom < 1e-8] = 1e-8  # Avoid division by zero
        
        # Term y / denom: Shape (n_grid, 3) * (n_grid, N) -> need to broadcast
        # y is (n_grid, 3), denom is (n_grid, N)
        # We need sum_y P(y) * W[y, k] * y / denom[y, k]
        # Let's construct the gradient matrix.
        
        # grad[k] = beta * sum_y P[y] * W[y, k] * grid_pts[y] / denom[y, k]
        # Note the negative sign in dm/dp_k: - w_k * y / denom
        # So grad[k] = - beta * sum_y P[y] * W[y, k] * grid_pts[y] / denom[y, k]
        
        # We can compute this efficiently.
        # coeff[y, k] = P[y] * W[y, k] / denom[y, k]
        # grad[k] = - beta * sum_y coeff[y, k] * grid_pts[y]
        
        coeff = P[:, np.newaxis] * W / denom  # Shape (n_grid, N)
        
        # grad = -beta * (coeff.T @ grid_pts)
        grad = -beta * (coeff.T @ grid_pts)
        
        # Project gradient to tangent space
        # grad_tangent = grad - (grad . p) * p
        dot_prod = np.sum(grad * pts, axis=1, keepdims=True)
        grad = grad - dot_prod * pts
        
        return energy, grad

    # Main Strategy
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_pts = normalize(fib_pts)
    
    best_pts = fib_pts.copy()
    best_cr = compute_exact_covering_radius(best_pts)
    
    # 2. Optimize using L-BFGS-B
    # We optimize in Cartesian coordinates, but project back to sphere after each step?
    # L-BFGS-B doesn't handle constraints well like trust-constr, but we can use a penalty or just project.
    # Better: Use a parameterization that stays on the sphere? Or just minimize in R^3 and normalize at the end?
    # If we minimize in R^3, the gradient points off-sphere.
    # The standard trick is to minimize on the sphere using Riemannian optimization, but scipy doesn't have that built-in easily.
    # Alternative: Use the fact that scaling doesn't change the direction. We can fix one component or use angles.
    # Simplest: Minimize in R^3, but after each function evaluation, normalize the points before computing energy/grad.
    # However, the gradient computed is for the normalized points. We need to ensure the optimizer moves in the right direction.
    # If we pass normalized pts to the function, the gradient is tangent.
    # We can use a callback to normalize pts after each step? No, minimize doesn't work that way.
    
    # Let's use a wrapper that normalizes inputs.
    def objective_and_grad(x):
        pts = x.reshape(N, 3)
        pts = normalize(pts)
        
        # Check time
        if time.time() > deadline - 5:
            raise ValueError("Timeout")
            
        e, g = soft_covering_energy_and_grad(pts)
        return e, g
        
    # Initial guess
    x0 = fib_pts.flatten()
    
    try:
        result = minimize(
            objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': 200, 'maxfun': 1000, 'disp': False}
        )
        
        if result.success or result.status in [1, 2]:
            opt_pts = normalize(result.x.reshape(N, 3))
            opt_cr = compute_exact_covering_radius(opt_pts)
            if opt_cr < best_cr:
                best_cr = opt_cr
                best_pts = opt_pts
    except Exception as e:
        pass
        
    # 3. Local Refinement with Hull-based search (using the best points as start)
    # This helps fix any non-convexities or local minima in the soft energy that the exact hull radius might penalize differently.
    
    def lloyd_relax(pts, steps=20):
        n_grid = 10000
        i = np.arange(n_grid) + 0.5
        z = 1.0 - 2.0 * i / n_grid
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        grid_pts = normalize(grid_pts)
        
        current_pts = pts.copy()
        for _ in range(steps):
            dots = grid_pts @ current_pts.T
            nearest_idx = np.argmax(dots, axis=1)
            new_pts = np.zeros_like(current_pts)
            for k in range(N):
                mask = (nearest_idx == k)
                if np.any(mask):
                    new_pts[k] = np.sum(grid_pts[mask], axis=0)
                else:
                    new_pts[k] = current_pts[k]
            new_pts = normalize(new_pts)
            current_pts = new_pts
        return current_pts

    # Apply Lloyd's to the optimized points to ensure valid Voronoi structure
    lloyd_pts = lloyd_relax(best_pts, steps=10)
    lloyd_cr = compute_exact_covering_radius(lloyd_pts)
    if lloyd_cr < best_cr:
        best_cr = lloyd_cr
        best_pts = lloyd_pts

    # Final Hull-based refinement
    def refine_with_hull(pts, iterations=10):
        current_pts = pts.copy()
        best_pts_so_far = current_pts.copy()
        best_cr, _, _ = compute_exact_covering_radius(current_pts), None, None
        # Re-implement simple version for speed
        best_cr = compute_exact_covering_radius(current_pts)
        
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break
            # Find worst facet
            try:
                hull = ConvexHull(current_pts)
                max_r_rad = 0.0
                worst_face = None
                worst_center = None
                for face in hull.simplices:
                    p0, p1, p2 = current_pts[face[0]], current_pts[face[1]], current_pts[face[2]]
                    v1 = p0 - p1
                    v2 = p0 - p2
                    c_vec = np.cross(v1, v2)
                    norm_c = np.linalg.norm(c_vec)
                    if norm_c < 1e-10: continue
                    c_vec /= norm_c
                    centroid = (p0 + p1 + p2) / 3.0
                    centroid /= np.linalg.norm(centroid)
                    if np.dot(c_vec, centroid) < 0: c_vec = -c_vec
                    n1 = np.cross(p1, p2)
                    n2 = np.cross(p2, p0)
                    n3 = np.cross(p0, p1)
                    if np.dot(centroid, n1) < 0: n1 = -n1
                    if np.dot(centroid, n2) < 0: n2 = -n2
                    if np.dot(centroid, n3) < 0: n3 = -n3
                    if (np.dot(c_vec, n1) >= -1e-8 and np.dot(c_vec, n2) >= -1e-8 and np.dot(c_vec, n3) >= -1e-8):
                        dist = np.arccos(np.clip(np.dot(c_vec, p0), -1.0, 1.0))
                        if dist > max_r_rad:
                            max_r_rad = dist
                            worst_face = face
                            worst_center = c_vec
            except:
                break
                
            if worst_face is None: break
            
            cr = np.degrees(max_r_rad)
            if cr < best_cr:
                best_cr = cr
                best_pts_so_far = current_pts.copy()
            
            # Move vertices of worst face towards center
            step_size = 0.1
            for idx in worst_face:
                new_pos = current_pts[idx] + step_size * (worst_center - current_pts[idx])
                current_pts[idx] = new_pos / np.linalg.norm(new_pos)
                
            # Small Lloyd relaxation
            current_pts = lloyd_relax(current_pts, steps=2)
            
        return best_pts_so_far

    final_pts = refine_with_hull(best_pts, iterations=15)
    final_cr = compute_exact_covering_radius(final_pts)
    if final_cr < best_cr:
        best_cr = final_cr
        best_pts = final_pts

    return best_pts
# EVOLVE-BLOCK-END
