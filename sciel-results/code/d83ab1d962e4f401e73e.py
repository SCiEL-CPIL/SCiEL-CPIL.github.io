"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay

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

    def compute_facets(pts):
        """
        Compute all facet circumcenters and their angular radii.
        Returns list of (radius_radians, circumcenter, simplex_indices)
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
        facets = compute_facets(pts)
        if not facets:
            return np.pi, np.array([0.0, 0.0, 1.0])

        max_radius = 0.0
        worst_center = None
        for radius, center, _ in facets:
            if radius > max_radius:
                max_radius = radius
                worst_center = center

        return max_radius, worst_center

    def spherical_voronoi_centroids(pts):
        """
        Lloyd's algorithm step: compute spherical Voronoi centroids.
        Uses Delaunay triangulation of the points on the sphere.
        """
        try:
            # Use Delaunay triangulation (works for points in general position)
            tri = Delaunay(pts)
        except Exception:
            return None

        # For each triangle, compute the centroid of the corresponding spherical Voronoi cell
        # The centroid of a spherical Voronoi cell is the circumcenter of the dual triangle
        # weighted by the cell area
        centroids = np.zeros_like(pts)
        weights = np.zeros(len(pts))

        for simplex in tri.simplices:
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

            # Weight by the solid angle (approximated by the area of the spherical triangle)
            # Using the formula: area = 2*arcsin(|det([v1,v2,v3])| / (2*|v1||v2||v3|))
            det_val = np.abs(np.dot(v1, np.cross(v2, v3)))
            area_weight = 2.0 * np.arcsin(np.clip(det_val / 2.0, 0.0, 1.0))

            for idx in simplex:
                centroids[idx] += c * area_weight
                weights[idx] += area_weight

        # Normalize
        valid = weights > 1e-10
        result = pts.copy()
        for i in range(len(pts)):
            if valid[i]:
                result[i] = centroids[i] / weights[i]
                result[i] /= np.linalg.norm(result[i])
        return result

    def lloyd_iterations(pts, max_iters=30):
        """
        Run Lloyd's algorithm for spherical Voronoi tessellation.
        """
        pts = pts.copy()
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break

            centroids = spherical_voronoi_centroids(pts)
            if centroids is None:
                break

            # Check convergence
            diff = np.linalg.norm(centroids - pts)
            pts = centroids
            if diff < 1e-8:
                break

        return pts

    def smooth_energy_gradient(pts):
        """
        Compute the gradient of a smooth energy that approximates the covering radius.
        Energy = repulsive forces between points + pressure away from facet circumcenters.
        """
        grad = np.zeros_like(pts)

        # Repulsive force: inverse square of angular distance
        for i in range(N):
            for j in range(i + 1, N):
                d = np.dot(pts[i], pts[j])
                d = np.clip(d, -0.999, 0.999)
                ang = np.arccos(d)
                if ang < 1e-6:
                    continue
                # Force magnitude ~ 1/ang^2, direction away from each other
                fmag = 1.0 / (ang * ang + 1e-6)
                # Tangent direction from j to i
                tangent_i = pts[i] - d * pts[j]
                tangent_i_norm = np.linalg.norm(tangent_i)
                if tangent_i_norm < 1e-10:
                    continue
                tangent_i /= tangent_i_norm
                grad[i] += fmag * tangent_i
                grad[j] -= fmag * tangent_i

        # Pressure away from facet circumcenters
        facets = compute_facets(pts)
        for radius, center, simplex in facets:
            # Pressure strength proportional to how "worst" this facet is
            pressure = radius * 0.5
            for idx in simplex:
                v = pts[idx]
                dir_vec = v - center
                dir_vec_norm = np.linalg.norm(dir_vec)
                if dir_vec_norm < 1e-10:
                    continue
                dir_vec /= dir_vec_norm
                # Project onto tangent space
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                dir_vec_norm = np.linalg.norm(dir_vec)
                if dir_vec_norm < 1e-10:
                    continue
                dir_vec /= dir_vec_norm
                grad[idx] += pressure * dir_vec

        return grad

    def optimize_worst_facet(pts, step_size, iterations=50):
        """
        Iteratively move the vertices of the worst facet towards the worst center.
        Uses adaptive step size and tangent space projection.
        """
        pts = pts.copy()
        prev_radius = np.pi
        for _ in range(iterations):
            if time.time() > deadline - 10:
                break

            radius, worst_center = compute_covering_radius(pts)

            # If radius hasn't improved, reduce step size
            if radius >= prev_radius:
                step_size *= 0.5
                if step_size < 1e-10:
                    break
            else:
                prev_radius = radius

            # Find the facet corresponding to worst_center
            try:
                hull = ConvexHull(pts)
            except:
                break

            target_indices = None
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
                if np.dot(c, worst_center) > 0.9999:
                    target_indices = simplex
                    break

            if target_indices is None:
                break

            # Move vertices towards worst_center in tangent space
            for idx in target_indices:
                v = pts[idx]
                dir_vec = worst_center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    # Project onto tangent space
                    radial_comp = np.dot(dir_vec, v)
                    dir_vec = dir_vec - radial_comp * v
                    dir_vec_norm = np.linalg.norm(dir_vec)
                    if dir_vec_norm > 1e-10:
                        dir_vec /= dir_vec_norm
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

    # Phase 2: Lloyd's algorithm for global structure
    if time.time() < deadline - 40:
        lloyd_pts = lloyd_iterations(current, max_iters=25)
        lloyd_radius, _ = compute_covering_radius(lloyd_pts)
        if lloyd_radius < best_radius:
            best_radius = lloyd_radius
            best = lloyd_pts.copy()
            current = lloyd_pts.copy()

    # Phase 3: Smooth energy gradient descent
    if time.time() < deadline - 30:
        smooth_pts = current.copy()
        step = 0.05
        for it in range(200):
            if time.time() > deadline - 30:
                break
            grad = smooth_energy_gradient(smooth_pts)
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            grad /= grad_norm
            smooth_pts = smooth_pts + step * grad
            smooth_pts = project_to_sphere(smooth_pts)
            # Decay step size
            step *= 0.98

        smooth_radius, _ = compute_covering_radius(smooth_pts)
        if smooth_radius < best_radius:
            best_radius = smooth_radius
            best = smooth_pts.copy()
            current = smooth_pts.copy()

    # Phase 4: Worst-facet optimization with adaptive step size
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    for step in step_sizes:
        if time.time() > deadline - 15:
            break
        current = optimize_worst_facet(current, step, iterations=60)
        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        else:
            current = best.copy()

    # Phase 5: Random restarts with small perturbations + Lloyd + polish
    if time.time() < deadline - 25:
        for restart in range(10):
            if time.time() > deadline - 25:
                break

            perturbed = best.copy()
            noise_scale = 0.04 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])

            # Quick Lloyd
            if time.time() < deadline - 25:
                perturbed = lloyd_iterations(perturbed, max_iters=8)

            # Polish
            for step in [0.02, 0.01]:
                if time.time() > deadline - 25:
                    break
                perturbed = optimize_worst_facet(perturbed, step, iterations=30)

            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 6: Final high-precision polish
    if time.time() < deadline - 5:
        final = best.copy()
        for _ in range(150):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(final)

            try:
                hull = ConvexHull(final)
            except:
                break

            target_indices = None
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
                if np.dot(c, worst_center) > 0.9999:
                    target_indices = simplex
                    break

            if target_indices is None:
                break

            # Adaptive step size
            step = 0.002 * max(0.3, radius / (np.pi / 10))

            for idx in target_indices:
                v = final[idx]
                dir_vec = worst_center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    radial_comp = np.dot(dir_vec, v)
                    dir_vec = dir_vec - radial_comp * v
                    dir_vec_norm = np.linalg.norm(dir_vec)
                    if dir_vec_norm > 1e-10:
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
