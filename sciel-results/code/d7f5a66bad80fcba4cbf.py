"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def project_to_sphere(pts):
        """Project points onto the unit sphere."""
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return pts / norms

    def compute_facet_data(points):
        """
        Compute convex hull and facet circumcenters/radii (vectorized).
        Returns: hull, centers (array), radii (array), worst_idx
        """
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        if np.any(norms < 1e-15):
            return None, None, None, None
        points = points / norms
        
        try:
            hull = ConvexHull(points)
        except Exception:
            return None, None, None, None
        
        facets = hull.simplices
        n_facets = len(facets)
        if n_facets == 0:
            return hull, np.zeros((0,3)), np.zeros(0), None
            
        a = points[facets[:, 0]]  # (F, 3)
        b = points[facets[:, 1]]
        c = points[facets[:, 2]]
        
        # Vectorized circumcenter computation
        v1 = a - b
        v2 = b - c
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1)
        
        # Handle degenerate facets (collinear points)
        valid = norm_n > 1e-12
        
        centers = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)
        
        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            
            # Ensure circumcenter faces outward (toward 'a')
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            
            centers[valid] = cc
            
            # Calculate radii (angular distance from center to vertex a)
            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)
        
        if n_facets > 0 and np.any(radii > 0):
            worst_idx = np.argmax(radii)
        else:
            worst_idx = None
            
        return hull, centers, radii, worst_idx

    def compute_covering_radius(points):
        """
        Compute the actual covering radius (max facet circumradius).
        Returns radius and worst center.
        """
        hull, centers, radii, worst_idx = compute_facet_data(points)
        if hull is None or worst_idx is None:
            return np.pi, np.array([0.0, 0.0, 1.0])
        return radii[worst_idx], centers[worst_idx]

    def soft_max_radius_and_grad(points, beta=50.0):
        """
        Compute soft-max radius and its gradient w.r.t. points.
        """
        hull, centers, radii, worst_idx = compute_facet_data(points)
        if hull is None or worst_idx is None:
            return np.pi, np.zeros_like(points)
        
        max_r = np.max(radii)
        # Use log-sum-exp trick for numerical stability
        shifted = beta * (radii - max_r)
        max_shifted = np.max(shifted)
        exp_terms = np.exp(shifted - max_shifted)
        log_sum = max_shifted + np.log(np.sum(exp_terms))
        soft_radius = max_r + log_sum / beta
        weights = exp_terms / np.sum(exp_terms)
        
        grads = np.zeros_like(points)
        n_facets = len(hull.simplices)
        
        for i in range(n_facets):
            w = weights[i]
            if w < 1e-15:
                continue
            r = radii[i]
            if r < 1e-12:
                continue
            
            C = centers[i]
            sin_r = np.sin(r)
            if sin_r < 1e-10:
                continue
            
            simplex = hull.simplices[i]
            p_indices = simplex
            p_pts = points[p_indices]
            
            # Gradient of arccos(dot(C, p)) w.r.t. p is -C/sin(r)
            grad_contrib = -C / sin_r
            
            # Project to tangent space at p
            dots = np.sum(p_pts * grad_contrib, axis=1)
            radial_comps = dots[:, np.newaxis] * p_pts
            tangential_grads = grad_contrib - radial_comps
            
            for k in range(3):
                grads[p_indices[k]] += w * tangential_grads[k]
        
        return soft_radius, grads

    def optimize_lbfgs_schedule(pts, time_budget):
        """
        Run L-BFGS-B with a schedule of increasing beta values.
        """
        current = pts.copy()
        best = pts.copy()
        best_radius, _ = compute_covering_radius(best)
        start_t = time.time()
        
        # Schedule of beta values: start smooth, end sharp
        beta_schedule = [10.0, 50.0, 200.0, 1000.0]
        
        for beta in beta_schedule:
            if time.time() - start_t > time_budget:
                break
                
            def obj_grad(x, b=beta):
                if time.time() - start_t > time_budget:
                    raise Exception("timeout")
                p = project_to_sphere(x.reshape(N, 3))
                soft_r, grads = soft_max_radius_and_grad(p, beta=b)
                return soft_r, grads.reshape(-1)

            try:
                x0 = current.flatten()
                # Adjust maxiter based on remaining time
                remaining = time_budget - (time.time() - start_t)
                max_iter = int(min(200, max(50, 10 * remaining)))
                
                result = minimize(
                    obj_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'maxfun': max_iter*2, 'ftol': 1e-12, 'gtol': 1e-9}
                )
                
                if result.success or np.isfinite(result.fun):
                    new_pts = project_to_sphere(result.x.reshape(N, 3))
                    new_radius, _ = compute_covering_radius(new_pts)
                    
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best = new_pts.copy()
                        current = new_pts.copy()
                        
            except Exception:
                pass
                
        return best, best_radius

    def polish_gradient_descent(pts, time_budget, beta=500.0):
        """
        Final local polish using projected gradient descent on tangent space.
        """
        current = pts.copy()
        best = pts.copy()
        best_radius, _ = compute_covering_radius(best)
        start_t = time.time()
        
        step_size = 0.01
        min_step = 1e-12
        decay = 0.8
        
        while time.time() - start_t < time_budget:
            soft_radius, grads = soft_max_radius_and_grad(current, beta=beta)
            
            # Normalize gradient to control step size magnitude
            grad_norm = np.linalg.norm(grads)
            if grad_norm < 1e-12:
                break
                
            # Take step in tangent space
            new_pts = current - step_size * grads
            new_pts = project_to_sphere(new_pts)
            
            new_radius, _ = compute_covering_radius(new_pts)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best = new_pts.copy()
                current = new_pts.copy()
                # Slightly increase step if successful? Or keep constant?
                # step_size = min(step_size * 1.05, 0.05)
            else:
                step_size *= decay
                if step_size < min_step:
                    break
                    
        return best, best_radius

    def optimize_worst_facet(pts, step_size, iterations=50):
        """
        Iteratively move the vertices of the worst facet towards the worst center.
        """
        pts = pts.copy()
        prev_radius = np.pi
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(pts)

            if radius >= prev_radius:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
            else:
                prev_radius = radius

            try:
                hull = ConvexHull(pts)
            except:
                break

            facet_data = []
            for simplex in hull.simplices:
                v1 = pts[simplex[0]]
                v2 = pts[simplex[1]]
                v3 = pts[simplex[2]]
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-12:
                    continue
                c = n / norm_n
                if np.dot(c, v1) < 0:
                    c = -c
                facet_data.append((c, simplex))

            if not facet_data:
                break

            target_indices_set = set()
            for c, simplex in facet_data:
                if np.dot(c, worst_center) > 0.999:
                    for idx in simplex:
                        target_indices_set.add(idx)

            if not target_indices_set:
                break

            for idx in target_indices_set:
                v = pts[idx]
                dir_vec = worst_center - v
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    pts[idx] = v + step_size * dir_vec
                    pts[idx] /= np.linalg.norm(pts[idx])

        return pts

    # Phase 1: Initial Fibonacci spiral
    current = fibonacci_sphere(N)
    best = current.copy()
    best_radius, _ = compute_covering_radius(best)

    # Phase 2: L-BFGS with Beta Schedule (Main Optimization)
    if time.time() < deadline - 40:
        try:
            lbfgs_pts, lbfgs_radius = optimize_lbfgs_schedule(current, time_budget=35.0)
            if lbfgs_radius < best_radius:
                best_radius = lbfgs_radius
                best = lbfgs_pts.copy()
                current = lbfgs_pts.copy()
        except Exception:
            pass

    # Phase 3: Projected Gradient Polish
    if time.time() < deadline - 20:
        try:
            polish_pts, polish_radius = polish_gradient_descent(current, time_budget=15.0, beta=500.0)
            if polish_radius < best_radius:
                best_radius = polish_radius
                best = polish_pts.copy()
                current = polish_pts.copy()
        except Exception:
            pass

    # Phase 4: Heuristic Worst-Facet Optimization
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    for step in step_sizes:
        if time.time() > deadline - 10:
            break
        current = optimize_worst_facet(current, step, iterations=50)
        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        else:
            current = best.copy()

    # Phase 5: Random Restarts with Full Pipeline
    if time.time() < deadline - 15:
        np.random.seed(42)
        restarts = 20
        for restart in range(restarts):
            if time.time() > deadline - 10:
                break
            
            # Perturb best solution
            perturbed = best.copy()
            noise_scale = 0.01 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])
            
            # Quick L-BFGS refinement
            if time.time() < deadline - 12:
                try:
                    lbfgs_pts, lbfgs_radius = optimize_lbfgs_schedule(perturbed, time_budget=2.0)
                    if lbfgs_radius < compute_covering_radius(perturbed)[0]:
                        perturbed = lbfgs_pts
                except Exception:
                    pass
            
            # Quick Gradient Polish
            if time.time() < deadline - 11:
                try:
                    polish_pts, polish_radius = polish_gradient_descent(perturbed, time_budget=1.0, beta=200.0)
                    if polish_radius < compute_covering_radius(perturbed)[0]:
                        perturbed = polish_pts
                except Exception:
                    pass
            
            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 6: Final High-Precision Polish
    if time.time() < deadline - 2:
        try:
            final_pts, final_radius = polish_gradient_descent(best, time_budget=3.0, beta=1000.0)
            if final_radius < best_radius:
                best = final_pts
                best_radius = final_radius
        except Exception:
            pass

    # Final normalization and validation
    best = project_to_sphere(best)
    norms = np.linalg.norm(best, axis=1)
    valid = (norms > 1e-10) & np.all(np.isfinite(best), axis=1)
    if not np.all(valid):
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
