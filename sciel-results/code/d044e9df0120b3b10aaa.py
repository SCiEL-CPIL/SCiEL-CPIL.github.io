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

    def compute_facet_circumcenters(pts):
        """
        Compute all facet circumcenters and their angular radii.
        Returns: list of (radius_radians, circumcenter, simplex_indices)
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return []

        results = []
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

            d = np.linalg.norm(c - v1)
            radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            results.append((radius, c, simplex))

        return results

    def compute_covering_radius(pts):
        """
        Compute the covering radius in radians and the worst-case circumcenter.
        """
        facets = compute_facet_circumcenters(pts)
        if not facets:
            return np.pi, np.array([0.0, 0.0, 1.0])

        max_radius = 0.0
        worst_center = None
        for radius, center, _ in facets:
            if radius > max_radius:
                max_radius = radius
                worst_center = center

        return max_radius, worst_center

    def spring_mass_relax(pts, iterations=500):
        """
        Spring-mass relaxation: treat convex hull edges as springs with rest length
        derived from optimal hexagonal packing. Forces vertices to minimize edge
        length variance while maintaining spherical constraint.
        """
        pts = pts.copy()
        # Optimal hexagonal packing edge length on sphere: for N points, 
        # each point has ~6 neighbors, area per point = 4*pi/N
        # Hexagon side s: area = (3*sqrt(3)/2)*s^2 = 4*pi/N => s = sqrt(8*pi/(3*sqrt(3)*N))
        s_opt = np.sqrt(8.0 * np.pi / (3.0 * np.sqrt(3.0) * N))
        
        damping = 0.1
        prev_energy = np.inf
        
        for it in range(iterations):
            if time.time() > deadline - 20:
                break
            
            try:
                hull = ConvexHull(pts)
            except:
                break
            
            # Build edge list from hull facets
            edge_set = set()
            for simplex in hull.simplices:
                edge_set.add(tuple(sorted((simplex[0], simplex[1]))))
                edge_set.add(tuple(sorted((simplex[1], simplex[2]))))
                edge_set.add(tuple(sorted((simplex[2], simplex[0]))))
            
            edges = list(edge_set)
            if not edges:
                break
            
            # Compute forces: spring force = k * (current_length - rest_length)
            forces = np.zeros_like(pts)
            for i, j in edges:
                v_i = pts[i]
                v_j = pts[j]
                # Angular distance
                cos_theta = np.clip(np.dot(v_i, v_j), -1.0, 1.0)
                theta = np.arccos(cos_theta)
                
                # Spring force magnitude
                k = 1.0
                force_mag = k * (theta - s_opt)
                
                # Direction: along the geodesic from i to j
                if theta < 1e-10:
                    continue
                # Tangent direction from i toward j
                dir_vec = v_j - np.dot(v_j, v_i) * v_i
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm < 1e-10:
                    continue
                dir_vec /= dir_norm
                
                # Apply force to both endpoints
                forces[i] += force_mag * dir_vec
                forces[j] -= force_mag * dir_vec
            
            # Update positions with damping
            for i in range(N):
                v = pts[i]
                # Project force to tangent space
                f_tangent = forces[i] - np.dot(forces[i], v) * v
                pts[i] = v + damping * f_tangent
                pts[i] /= np.linalg.norm(pts[i])
            
            # Check convergence
            current_energy = np.sum((np.linalg.norm(forces, axis=1))**2)
            if abs(prev_energy - current_energy) < 1e-12:
                break
            prev_energy = current_energy
        
        return pts

    def sum_sq_min_dist_objective(params):
        """
        Objective for L-BFGS-B: smooth surrogate based on soft-max of facet circumradii.
        This directly targets minimizing the covering radius.
        """
        if time.time() > deadline - 10:
            return 1e10

        pts = params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms

        facets = compute_facet_circumcenters(pts)
        if not facets:
            return 1e10

        radii = np.array([r for r, _, _ in facets])
        if len(radii) == 0:
            return 1e10

        tau = 0.015
        scaled = radii / tau
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        return tau * log_sum_exp

    def optimize_worst_facet(pts, step_size, iterations=50):
        """
        Iteratively move the vertices of the worst facet towards the worst center.
        Uses adaptive step size and tangent space projection.
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

    def project_to_sphere(pts):
        """Project points onto the unit sphere."""
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return pts / norms

    # Phase 1: Initial Fibonacci spiral
    current = fibonacci_sphere(N)
    best = current.copy()
    best_radius, _ = compute_covering_radius(best)

    # Phase 2: Spring-mass relaxation to get a good global structure
    if time.time() < deadline - 40:
        try:
            spring_pts = spring_mass_relax(current, iterations=500)
            spring_radius, _ = compute_covering_radius(spring_pts)
            if spring_radius < best_radius:
                best_radius = spring_radius
                best = spring_pts.copy()
                current = spring_pts.copy()
        except Exception:
            pass

    # Phase 3: L-BFGS-B on soft-max facet radii for refinement
    if time.time() < deadline - 25:
        try:
            x0 = current.flatten()
            for schedule_idx in range(2):
                if time.time() > deadline - 25:
                    break
                ftol = 1e-9 / (schedule_idx + 1)
                gtol = 1e-7 / (schedule_idx + 1)
                result = minimize(
                    sum_sq_min_dist_objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 250, 'maxfun': 1800, 'ftol': ftol, 'gtol': gtol}
                )
                if result.success or result.fun < sum_sq_min_dist_objective(x0):
                    lbfgs_pts = project_to_sphere(result.x.reshape(N, 3))
                    lbfgs_radius, _ = compute_covering_radius(lbfgs_pts)
                    if lbfgs_radius < best_radius:
                        best_radius = lbfgs_radius
                        best = lbfgs_pts.copy()
                        current = lbfgs_pts.copy()
                        x0 = best.flatten()
        except Exception:
            pass

    # Phase 4: Worst-facet optimization with adaptive step size
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    for step in step_sizes:
        if time.time() > deadline - 15:
            break
        current = optimize_worst_facet(current, step, iterations=100)
        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        else:
            current = best.copy()

    # Phase 5: Random restarts with small perturbations
    if time.time() < deadline - 25:
        np.random.seed(42)
        for restart in range(50):
            if time.time() > deadline - 25:
                break
            perturbed = best.copy()
            noise_scale = 0.02 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])
            for step in [0.01, 0.005, 0.002, 0.001]:
                if time.time() > deadline - 25:
                    break
                perturbed = optimize_worst_facet(perturbed, step, iterations=60)
            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 6: Final high-precision polish
    if time.time() < deadline - 3:
        final = best.copy()
        step = 0.001
        prev_radius = compute_covering_radius(final)[0]
        stall_count = 0
        for it in range(2000):
            if time.time() > deadline - 3:
                break
            radius, worst_center = compute_covering_radius(final)
            if radius >= prev_radius - 1e-15:
                step *= 0.5
                stall_count += 1
                if step < 1e-15 or stall_count > 50:
                    break
            else:
                prev_radius = radius
                stall_count = 0

            try:
                hull = ConvexHull(final)
            except:
                break

            target_indices_set = set()
            for simplex in hull.simplices:
                v1 = final[simplex[0]]
                v2 = final[simplex[1]]
                v3 = final[simplex[2]]
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-12:
                    continue
                c = n / norm_n
                if np.dot(c, v1) < 0:
                    c = -c
                if np.dot(c, worst_center) > 0.99995:
                    for idx in simplex:
                        target_indices_set.add(idx)

            if not target_indices_set:
                break

            for idx in target_indices_set:
                v = final[idx]
                dir_vec = worst_center - v
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
    norms = np.linalg.norm(best, axis=1)
    valid = (norms > 1e-10) & np.all(np.isfinite(best), axis=1)
    if not np.all(valid):
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
