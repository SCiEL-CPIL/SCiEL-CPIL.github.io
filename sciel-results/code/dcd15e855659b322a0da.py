"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

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

    def compute_facet_circumcenters(pts):
        """
        Compute all facet circumcenters and their angular radii (vectorized).
        Returns: (radii_array, centers_array, simplices_array)
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None, None, None

        simplices = hull.simplices  # shape (F, 3)
        F = len(simplices)
        
        # Vectorized: get all three vertices for each facet
        v1 = pts[simplices[:, 0]]  # (F, 3)
        v2 = pts[simplices[:, 1]]  # (F, 3)
        v3 = pts[simplices[:, 2]]  # (F, 3)

        # Compute normals
        edge1 = v2 - v1  # (F, 3)
        edge2 = v3 - v1  # (F, 3)
        normals = np.cross(edge1, edge2)  # (F, 3)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)  # (F, 1)
        
        # Filter out degenerate facets
        valid = norms.squeeze() > 1e-12
        if not np.any(valid):
            return None, None, None
        
        normals = normals[valid]
        v1 = v1[valid]
        simplices_valid = simplices[valid]
        
        # Normalize normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        centers = normals / norms  # (F, 3)
        
        # Ensure centers point outward (dot with v1 > 0)
        dots = np.sum(centers * v1, axis=1)  # (F,)
        centers[dots < 0] *= -1
        
        # Compute radii: radius = arccos(dot(center, v1))
        cos_r = np.clip(np.sum(centers * v1, axis=1), -1.0, 1.0)
        radii = np.arccos(cos_r)  # (F,)
        
        return radii, centers, simplices_valid

    def compute_covering_radius(pts):
        """
        Compute the covering radius in radians and the worst-case circumcenter.
        """
        radii, centers, simplices = compute_facet_circumcenters(pts)
        if radii is None:
            return np.pi, np.array([0.0, 0.0, 1.0])

        max_idx = np.argmax(radii)
        max_radius = radii[max_idx]
        worst_center = centers[max_idx]

        return max_radius, worst_center

    def soft_max_radius_and_grad(pts, beta=50.0):
        """
        Compute soft-max radius and its gradient w.r.t. points.
        """
        radii, centers, simplices = compute_facet_circumcenters(pts)
        if radii is None:
            return np.pi, np.zeros_like(pts)
        
        max_r = np.max(radii)
        # Use log-sum-exp trick for numerical stability
        shifted = beta * (radii - max_r)
        max_shifted = np.max(shifted)
        exp_terms = np.exp(shifted - max_shifted)
        log_sum = max_shifted + np.log(np.sum(exp_terms))
        soft_radius = max_r + log_sum / beta
        weights = exp_terms / np.sum(exp_terms)
        
        grads = np.zeros_like(pts)
        n_facets = len(simplices)
        
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
            
            simplex = simplices[i]
            p_indices = simplex
            p_pts = pts[p_indices]
            
            # Gradient of arccos(dot(C, p)) w.r.t. p is -C/sin(r)
            grad_contrib = -C / sin_r
            
            # Project to tangent space at each point in the facet
            dots = np.sum(p_pts * grad_contrib, axis=1)
            radial_comps = dots[:, np.newaxis] * p_pts
            tangential_grads = grad_contrib - radial_comps
            
            for k in range(3):
                grads[p_indices[k]] += w * tangential_grads[k]
        
        return soft_radius, grads

    def soft_max_radius_objective(flat_pts, beta=50.0):
        """
        Smooth surrogate for the max facet circumradius using log-sum-exp.
        This is differentiable and avoids the non-smoothness of the max operator.
        """
        if time.time() > deadline - 10:
            return 1e10

        pts = flat_pts.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms

        radii, centers, simplices = compute_facet_circumcenters(pts)
        if radii is None:
            return 1e10

        # Log-sum-exp for numerical stability
        max_r = np.max(radii)
        if max_r > 0:
            log_sum = max_r + np.log(np.sum(np.exp((radii - max_r) * beta)) / len(radii)) / beta
        else:
            log_sum = 0.0
        
        # Add a small variance penalty for uniformity
        var_penalty = 0.005 * np.var(radii ** 2)
        
        return log_sum + var_penalty

    def optimize_worst_facet(pts, step_size, iterations=50):
        """
        Iteratively move the vertices of the worst facet towards the worst center.
        Uses tangent space projection and adaptive step size.
        """
        pts = pts.copy()
        prev_radius = np.pi
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(pts)

            # If radius hasn't improved, reduce step size
            if radius >= prev_radius - 1e-14:
                step_size *= 0.7
                if step_size < 1e-10:
                    break
            else:
                prev_radius = radius

            # Get facet data (vectorized)
            radii, centers, simplices = compute_facet_circumcenters(pts)
            if radii is None:
                break

            # Find facets that are close to the worst center (within threshold)
            dots = np.sum(centers * worst_center, axis=1)  # (F,)
            mask = dots > 0.998  # Slightly looser threshold to include nearby facets
            target_indices = set()
            for simplex in simplices[mask]:
                for idx in simplex:
                    target_indices.add(int(idx))

            if not target_indices:
                break

            # Move vertices towards worst_center in tangent space
            for idx in target_indices:
                v = pts[idx]
                dir_vec = worst_center - v
                # Project onto tangent space: remove radial component
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    pts[idx] = v + step_size * dir_vec
                    pts[idx] /= np.linalg.norm(pts[idx])

        return pts

    # Phase 1: Initial Fibonacci spiral with multiple offsets
    best = None
    best_radius = np.pi

    for offset in [0.5, 0.3, 0.7]:
        if time.time() > deadline - 15:
            break

        current = fibonacci_sphere(N, offset)

        # Phase 2: L-BFGS-B optimization on the smooth soft-max surrogate
        if time.time() < deadline - 20:
            flat_initial = current.flatten()
            try:
                result = minimize(
                    soft_max_radius_objective,
                    flat_initial,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 800, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.success or result.fun < soft_max_radius_objective(flat_initial):
                    current = project_to_sphere(result.x.reshape(N, 3))
            except Exception:
                pass

        # Phase 3: Worst-facet optimization to refine
        if time.time() < deadline - 15:
            for step in [0.1, 0.05, 0.02, 0.01, 0.005]:
                if time.time() > deadline - 15:
                    break
                current = optimize_worst_facet(current, step, iterations=80)

        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()

    # Phase 4: Gradient-based refinement using the soft-max gradient
    if time.time() < deadline - 30:
        current = best.copy()
        step = 0.01
        prev_radius = compute_covering_radius(current)[0]
        stall_count = 0
        
        for it in range(500):
            if time.time() > deadline - 30:
                break
            
            radius, _ = compute_covering_radius(current)
            if radius < prev_radius - 1e-14:
                prev_radius = radius
                stall_count = 0
            else:
                step *= 0.8
                stall_count += 1
                if step < 1e-12 or stall_count > 20:
                    break
            
            soft_r, grads = soft_max_radius_and_grad(current, beta=50.0)
            grad_norm = np.linalg.norm(grads)
            if grad_norm < 1e-15:
                break
            
            direction = grads / grad_norm
            new_pts = current - step * direction
            new_pts = project_to_sphere(new_pts)
            
            new_radius, _ = compute_covering_radius(new_pts)
            if new_radius < radius - 1e-15:
                current = new_pts
                step = min(step * 1.05, 0.05)
        
        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()

    # Phase 5: Random restarts with small perturbations
    if time.time() < deadline - 25:
        np.random.seed(42)  # For reproducibility
        for restart in range(40):
            if time.time() > deadline - 25:
                break

            # Create a perturbed version of the best solution
            # Use a wider range of noise scales
            noise_scale = 0.03 * np.exp(-restart * 0.06)
            perturbed = best.copy()
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])

            # Optimize the perturbed solution
            if time.time() < deadline - 25:
                flat_pert = perturbed.flatten()
                try:
                    result = minimize(
                        soft_max_radius_objective,
                        flat_pert,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'maxfun': 400, 'ftol': 1e-9, 'gtol': 1e-7}
                    )
                    if result.success or result.fun < soft_max_radius_objective(flat_pert):
                        perturbed = project_to_sphere(result.x.reshape(N, 3))
                except Exception:
                    pass

            for step in [0.015, 0.008, 0.004, 0.002]:
                if time.time() > deadline - 25:
                    break
                perturbed = optimize_worst_facet(perturbed, step, iterations=60)

            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 6: Final high-precision polish via adaptive step size in tangent space
    if time.time() < deadline - 5:
        final = best.copy()
        # Use adaptive step size that decreases as we converge
        step = 0.003
        prev_radius = compute_covering_radius(final)[0]
        stall_count = 0
        for it in range(1000):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(final)

            # If no improvement, reduce step size
            if radius >= prev_radius - 1e-14:
                step *= 0.6
                stall_count += 1
                if step < 1e-13 or stall_count > 30:
                    break
            else:
                prev_radius = radius
                stall_count = 0

            # Get facet data (vectorized)
            radii, centers, simplices = compute_facet_circumcenters(final)
            if radii is None:
                break

            # Find all facets close to worst_center (use tighter threshold)
            dots = np.sum(centers * worst_center, axis=1)
            mask = dots > 0.9995
            target_indices_set = set()
            for simplex in simplices[mask]:
                for idx in simplex:
                    target_indices_set.add(int(idx))

            if not target_indices_set:
                break

            for idx in target_indices_set:
                v = final[idx]
                dir_vec = worst_center - v
                # Project onto tangent space: remove radial component
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                dir_vec_norm = np.linalg.norm(dir_vec)
                if dir_vec_norm > 1e-12:
                    dir_vec /= dir_vec_norm
                    final[idx] = v + step * dir_vec
                    final[idx] /= np.linalg.norm(final[idx])

        radius, _ = compute_covering_radius(final)
        if radius < best_radius:
            best_radius = radius
            best = final

    # Final normalization and validation
    best = project_to_sphere(best)

    # Ensure all points are non-zero and finite
    norms = np.linalg.norm(best, axis=1)
    valid = (norms > 1e-10) & np.all(np.isfinite(best), axis=1)
    if not np.all(valid):
        # Fallback to Fibonacci if something went wrong
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
