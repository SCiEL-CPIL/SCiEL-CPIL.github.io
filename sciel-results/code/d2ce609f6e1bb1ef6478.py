"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Cover the unit sphere with 50 equal caps.
    Strategy:
    1. Start with a good heuristic (Fibonacci spiral).
    2. Refine using a variant of Lloyd's algorithm / Voronoi relaxation.
    3. Use gradient-based optimization with a precise covering radius estimator.
    4. Use time budget to iterate until deadline.
    """
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s limit

    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_precise(points):
        """
        Compute the covering radius using the convex hull approach.
        The covering radius is the maximum circumradius of the spherical Voronoi cells,
        which corresponds to the maximum circumradius of the facets of the convex hull
        of the points (each facet defines a cap that must be empty).
        """
        try:
            hull = ConvexHull(points)
        except Exception:
            # Fallback: sampling-based estimate
            return compute_covering_radius_sampled(points)
        
        max_circumradius = 0.0
        # For each facet (triangle) of the convex hull, compute its circumradius
        # The circumradius of a triangle with side lengths a, b, c is abc / (4*Area)
        # On the unit sphere, the angular radius of the cap defined by a facet is
        # related to the circumradius of the triangle.
        
        # Actually, for points on the unit sphere, the covering radius is the maximum
        # over all facets of the angular distance from the facet's circumcenter to a vertex.
        # For a triangle with vertices v1, v2, v3 on the unit sphere, the circumcenter
        # (in 3D) is the point equidistant from all three. The angular radius is the
        # angle between the circumcenter (normalized) and any vertex.
        
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            # Compute the circumcenter of the triangle
            # The circumcenter lies in the plane of the triangle
            # We can find it by solving for a point equidistant from v1, v2, v3
            # and in the plane of v1, v2, v3.
            
            # Alternative: use the formula for circumradius of a triangle
            a = np.linalg.norm(v2 - v3)
            b = np.linalg.norm(v1 - v3)
            c = np.linalg.norm(v1 - v2)
            
            # Area of triangle using cross product
            cross = np.cross(v2 - v1, v3 - v1)
            area = 0.5 * np.linalg.norm(cross)
            
            if area < 1e-10:
                continue
            
            # Circumradius in 3D
            R_3d = (a * b * c) / (4.0 * area)
            
            # For points on the unit sphere, the angular radius of the cap
            # is arccos of the dot product between the normalized circumcenter
            # and a vertex. But we need to be careful.
            
            # Actually, the angular covering radius for a facet is the maximum
            # angular distance from any point in the spherical Voronoi cell to
            # the nearest site. This is equal to the circumradius of the triangle
            # formed by the three sites, measured as an angle on the sphere.
            
            # For a triangle with side lengths a, b, c (chord lengths on unit sphere),
            # the angular side lengths are 2*arcsin(a/2), etc.
            # The circumradius (angular) can be computed from the angular sides.
            
            # Simpler approach: The circumcenter in 3D space, when normalized,
            # gives the center of the cap. The angular radius is the angle
            # between this center and any vertex.
            
            # Compute circumcenter:
            # Let w = v1 + alpha*(v2-v1) + beta*(v3-v1)
            # We need |w - v1|^2 = |w - v2|^2 = |w - v3|^2
            # This gives a linear system for alpha, beta.
            
            e1 = v2 - v1
            e2 = v3 - v1
            
            # |w - v1|^2 = |w - v2|^2
            # |v1 + alpha*e1 + beta*e2 - v1|^2 = |v1 + alpha*e1 + beta*e2 - v2|^2
            # |alpha*e1 + beta*e2|^2 = |alpha*e1 + beta*e2 - e1|^2
            # alpha^2*|e1|^2 + 2*alpha*beta*(e1.e2) + beta^2*|e2|^2 = 
            #   alpha^2*|e1|^2 + 2*alpha*beta*(e1.e2) + beta^2*|e2|^2 - 2*alpha*|e1|^2 - 2*beta*(e1.e2) + |e1|^2
            # 0 = -2*alpha*|e1|^2 - 2*beta*(e1.e2) + |e1|^2
            # 2*alpha*|e1|^2 + 2*beta*(e1.e2) = |e1|^2
            
            # |w - v1|^2 = |w - v3|^2
            # 2*alpha*(e1.e2) + 2*beta*|e2|^2 = |e2|^2
            
            A_mat = np.array([
                [2.0 * np.dot(e1, e1), 2.0 * np.dot(e1, e2)],
                [2.0 * np.dot(e1, e2), 2.0 * np.dot(e2, e2)]
            ])
            b_vec = np.array([np.dot(e1, e1), np.dot(e2, e2)])
            
            try:
                params = np.linalg.solve(A_mat, b_vec)
            except np.linalg.LinAlgError:
                continue
            
            circumcenter = v1 + params[0] * e1 + params[1] * e2
            norm_cc = np.linalg.norm(circumcenter)
            if norm_cc < 1e-10:
                continue
            circumcenter_norm = circumcenter / norm_cc
            
            # Angular radius: angle between circumcenter_norm and v1
            dot_val = np.clip(np.dot(circumcenter_norm, v1), -1.0, 1.0)
            angular_radius = np.arccos(dot_val)
            
            if angular_radius > max_circumradius:
                max_circumradius = angular_radius
        
        return np.degrees(max_circumradius)

    def compute_covering_radius_sampled(points, num_test_points=100000):
        """
        Estimate covering radius by sampling many points on the sphere.
        """
        test_points = fibonacci_sphere(num_test_points)
        dot_products = test_points @ points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        min_dot = np.min(dot_products, axis=1)
        angles = np.arccos(min_dot)
        max_angle = np.max(angles)
        return np.degrees(max_angle)

    # Start with Fibonacci
    best_points = fibonacci_sphere(N)
    best_score = compute_covering_radius_sampled(best_points)

    rng = np.random.RandomState(42)

    # Pre-generate test points for Lloyd iterations
    num_lloyd_test = 100000
    lloyd_test_points = fibonacci_sphere(num_lloyd_test)

    step_size = 0.1
    iterations = 0

    # Phase 1: Lloyd's algorithm
    while time.time() - start_time < timeout * 0.4:
        dot_products = lloyd_test_points @ best_points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        nearest = np.argmax(dot_products, axis=1)

        new_points = np.zeros_like(best_points)
        counts = np.zeros(N)
        for i in range(N):
            mask = (nearest == i)
            if np.any(mask):
                centroid = lloyd_test_points[mask].mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-10:
                    new_points[i] = centroid / norm
                    counts[i] = np.sum(mask)
                else:
                    new_points[i] = best_points[i]
                    counts[i] = 1
            else:
                new_points[i] = best_points[i]
                counts[i] = 1

        for i in range(N):
            if counts[i] > 0:
                current = best_points[i]
                target = new_points[i]
                moved = current + step_size * (target - current)
                norm = np.linalg.norm(moved)
                if norm > 1e-10:
                    best_points[i] = moved / norm

        iterations += 1

        if iterations % 10 == 0:
            current_score = compute_covering_radius_sampled(best_points, 50000)
            if current_score < best_score:
                best_score = current_score
                step_size *= 0.95
            else:
                if iterations % 50 == 0:
                    perturbation = rng.standard_normal((N, 3)) * 0.01
                    best_points = best_points + perturbation
                    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
                    best_points = best_points / norms
                    step_size = max(0.01, step_size * 0.5)

    # Phase 2: Gradient-based refinement using precise covering radius
    # We'll use a simple gradient descent on the covering radius
    # The covering radius is determined by the "worst" facet(s)
    # We can move points to reduce the circumradius of the worst facet
    
    def get_worst_facet(points):
        """Find the facet with the largest circumradius."""
        try:
            hull = ConvexHull(points)
        except Exception:
            return None, None, None
        
        max_circumradius = 0.0
        worst_simplex = None
        circumcenter = None
        
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            e1 = v2 - v1
            e2 = v3 - v1
            
            A_mat = np.array([
                [2.0 * np.dot(e1, e1), 2.0 * np.dot(e1, e2)],
                [2.0 * np.dot(e1, e2), 2.0 * np.dot(e2, e2)]
            ])
            b_vec = np.array([np.dot(e1, e1), np.dot(e2, e2)])
            
            try:
                params = np.linalg.solve(A_mat, b_vec)
            except np.linalg.LinAlgError:
                continue
            
            cc = v1 + params[0] * e1 + params[1] * e2
            norm_cc = np.linalg.norm(cc)
            if norm_cc < 1e-10:
                continue
            cc_norm = cc / norm_cc
            
            dot_val = np.clip(np.dot(cc_norm, v1), -1.0, 1.0)
            angular_radius = np.arccos(dot_val)
            
            if angular_radius > max_circumradius:
                max_circumradius = angular_radius
                worst_simplex = simplex
                circumcenter = cc_norm
        
        return worst_simplex, circumcenter, max_circumradius

    # Gradient descent phase
    grad_step = 0.005
    for _ in range(200):
        if time.time() - start_time >= timeout:
            break
        
        worst_simplex, circumcenter, max_circum = get_worst_facet(best_points)
        
        if worst_simplex is None:
            break
        
        # Move the points in the worst facet away from the circumcenter
        # to increase the angular radius... wait, we want to DECREASE the covering radius.
        # The covering radius is the maximum angular distance from the circumcenter to a vertex.
        # To reduce it, we should move the vertices closer to the circumcenter,
        # or move the circumcenter closer to the vertices.
        # Actually, we want to reduce the circumradius, which means making the triangle
        # "smaller" on the sphere, or moving points to reduce the worst case.
        
        # A better approach: for each point in the worst simplex, move it slightly
        # toward the circumcenter to reduce the angular spread.
        
        for idx in worst_simplex:
            current = best_points[idx]
            # Move toward circumcenter
            moved = current + grad_step * (circumcenter - current)
            norm = np.linalg.norm(moved)
            if norm > 1e-10:
                best_points[idx] = moved / norm

        # Also, check if any point is too close to another and separate them
        for i in range(N):
            for j in range(i+1, N):
                dot_val = np.dot(best_points[i], best_points[j])
                if dot_val > 0.99:  # Very close
                    # Push them apart
                    diff = best_points[i] - best_points[j]
                    norm_diff = np.linalg.norm(diff)
                    if norm_diff > 1e-10:
                        diff_norm = diff / norm_diff
                        best_points[i] = best_points[i] + 0.001 * diff_norm
                        best_points[j] = best_points[j] - 0.001 * diff_norm
                        best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
                        best_points[j] = best_points[j] / np.linalg.norm(best_points[j])

        # Evaluate score periodically
        if _ % 10 == 0:
            current_score = compute_covering_radius_precise(best_points)
            if current_score < best_score:
                best_score = current_score
            else:
                # Try a small perturbation
                if _ % 50 == 0:
                    perturbation = rng.standard_normal((N, 3)) * 0.005
                    best_points = best_points + perturbation
                    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
                    best_points = best_points / norms

    # Final refinement with more Lloyd iterations
    step_size = 0.01
    while time.time() - start_time < timeout:
        dot_products = lloyd_test_points @ best_points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        nearest = np.argmax(dot_products, axis=1)

        new_points = np.zeros_like(best_points)
        counts = np.zeros(N)
        for i in range(N):
            mask = (nearest == i)
            if np.any(mask):
                centroid = lloyd_test_points[mask].mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-10:
                    new_points[i] = centroid / norm
                    counts[i] = np.sum(mask)
                else:
                    new_points[i] = best_points[i]
                    counts[i] = 1
            else:
                new_points[i] = best_points[i]
                counts[i] = 1

        for i in range(N):
            if counts[i] > 0:
                current = best_points[i]
                target = new_points[i]
                moved = current + step_size * (target - current)
                norm = np.linalg.norm(moved)
                if norm > 1e-10:
                    best_points[i] = moved / norm

        if iterations % 10 == 0:
            current_score = compute_covering_radius_precise(best_points)
            if current_score < best_score:
                best_score = current_score

        iterations += 1
        if iterations % 20 == 0:
            step_size *= 0.98

    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms

    return best_points
# EVOLVE-BLOCK-END
