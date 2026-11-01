"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius(pts):
        """Compute covering radius as max circumradius over convex hull faces (vectorized)."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices  # shape (F, 3)
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]

            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)

            radii = np.zeros(len(faces))
            valid = norms > 1e-10

            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r

            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)

            return float(np.max(radii))
        except Exception:
            # Fallback: grid-based estimate
            lat = np.linspace(-np.pi/2, np.pi/2, 50)
            lon = np.linspace(0, 2*np.pi, 100, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            dot = np.dot(grid_pts, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def compute_face_radii(pts):
        """Return array of face circumradii (in radians) for gradient-aware optimization."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]

            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)

            radii = np.zeros(len(faces))
            valid = norms > 1e-10

            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r

            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)

            return radii, faces
        except Exception:
            return None, None

    def soft_max(radii, beta=50.0):
        """Soft-max surrogate for the max of face radii, differentiable."""
        m = np.max(radii)
        if m < 1e-15:
            return 0.0, np.zeros(len(radii))
        shifted = radii - m
        exp_vals = np.exp(beta * shifted)
        log_sum = m + np.log(np.sum(exp_vals)) / beta
        # Gradient weights
        weights = exp_vals / np.sum(exp_vals)
        return log_sum, weights

    def compute_soft_objective(pts, beta=50.0):
        """Compute soft-max of face radii and the weights."""
        face_radii, faces = compute_face_radii(pts)
        if face_radii is None:
            return None, None, None
        obj, weights = soft_max(face_radii, beta)
        return obj, face_radii, weights

    def gradient_descent_face(pts, n_iter=200, lr=0.01, beta=50.0):
        """
        Gradient descent on the soft-max of face circumradii.
        For each face, we move vertices AWAY from the circumcenter
        to reduce the circumradius (shrink the spherical triangle).
        """
        best = pts.copy()
        best_rad = compute_covering_radius(best)

        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break

            face_radii, faces = compute_face_radii(pts)
            if face_radii is None:
                break

            # Use soft-max weights to focus on large faces
            _, weights = soft_max(face_radii, beta)

            # Compute gradient: for each vertex, accumulate the direction
            # that reduces the largest face radii
            grad = np.zeros_like(pts)

            for fi in range(len(faces)):
                w = weights[fi]
                if w < 1e-8:
                    continue

                f = faces[fi]
                a = pts[f[0]]
                b = pts[f[1]]
                c = pts[f[2]]

                ab = np.cross(a, b)
                bc = np.cross(b, c)
                ca = np.cross(c, a)
                c_dir = ab + bc + ca
                norm_c = np.linalg.norm(c_dir)

                if norm_c < 1e-10:
                    continue
                c_dir = c_dir / norm_c

                # Move vertices AWAY from the circumcenter to shrink the triangle
                # and reduce the circumradius
                for vi in f:
                    p = pts[vi]
                    # Tangential component of c_dir at p (direction toward circumcenter)
                    tangent = c_dir - np.dot(p, c_dir) * c_dir
                    tangent_norm = np.linalg.norm(tangent)
                    if tangent_norm > 1e-10:
                        # Moving p AWAY from circumcenter (negative tangent)
                        # decreases the circumradius
                        grad[vi] += -w * tangent / tangent_norm

            # Normalize gradient
            grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
            grad_norms[grad_norms < 1e-15] = 1.0
            grad = grad / grad_norms

            # Apply step
            step = lr * (1.0 - 0.5 * it / n_iter)
            pts = pts + step * grad
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms

            # Check improvement
            rad = compute_covering_radius(pts)
            if rad < best_rad - 1e-12:
                best_rad = rad
                best = pts.copy()

        return best

    def polish_step(points, alpha, n_iter=50):
        """High-precision polish: repulsion to maintain uniformity + targeted
        shrinking of the largest facets."""
        for i in range(n_iter):
            # Repulsion to maintain even spacing
            dot = np.dot(points, points.T)
            dot = np.clip(dot, -1.0, 1.0)
            np.fill_diagonal(dot, 1.0)
            dist_sq = 2.0 - 2.0 * dot
            dist_sq[dist_sq < 1e-10] = 1e-10
            w = 1.0 / (dist_sq ** 1.5)
            np.fill_diagonal(w, 0.0)
            sum_w = np.sum(w, axis=1)
            sum_p_w = w @ points
            forces = points * sum_w[:, np.newaxis] - sum_p_w

            # Add targeted shrinking of large facets
            face_radii, faces = compute_face_radii(points)
            if face_radii is not None and faces is not None:
                max_rad = np.max(face_radii)
                if max_rad > 1e-10:
                    shrink_force = np.zeros_like(points)
                    # Only shrink facets that are near the maximum
                    threshold = max_rad * 0.9
                    for fi in range(len(faces)):
                        if face_radii[fi] < threshold:
                            continue
                        f = faces[fi]
                        a, b, c = points[f[0]], points[f[1]], points[f[2]]
                        ab = np.cross(a, b)
                        bc = np.cross(b, c)
                        ca = np.cross(c, a)
                        c_dir = ab + bc + ca
                        norm_c = np.linalg.norm(c_dir)
                        if norm_c < 1e-10:
                            continue
                        c_dir = c_dir / norm_c
                        weight = (face_radii[fi] - threshold) / (max_rad - threshold + 1e-15)
                        for vi in f:
                            p = points[vi]
                            toward = c_dir - np.dot(p, c_dir) * c_dir
                            tn = np.linalg.norm(toward)
                            if tn > 1e-10:
                                shrink_force[vi] += weight * toward / tn
                    forces += 0.5 * shrink_force

            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        return points

    def targeted_face_reduction(points, n_iter=100):
        """Directly reduce the largest facet radii by pulling vertices toward each other
        (away from circumcenter) to shrink the spherical triangles."""
        best = points.copy()
        best_rad = compute_covering_radius(best)

        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break

            face_radii, faces = compute_face_radii(points)
            if face_radii is None:
                break

            # Find the largest facets and pull their vertices together
            top_k = min(15, len(faces))
            top_indices = np.argsort(face_radii)[-top_k:]

            # For each large facet, pull vertices toward each other (toward centroid)
            displacement = np.zeros_like(points)
            for fi in top_indices:
                if face_radii[fi] < best_rad * 0.85:
                    continue
                f = faces[fi]
                a, b, c = points[f[0]], points[f[1]], points[f[2]]
                # Compute circumcenter of the triangle on the sphere
                ab = np.cross(a, b)
                bc = np.cross(b, c)
                ca = np.cross(c, a)
                c_dir = ab + bc + ca
                norm_c = np.linalg.norm(c_dir)
                if norm_c < 1e-10:
                    continue
                c_dir = c_dir / norm_c
                # Pull each vertex toward the circumcenter to shrink the triangle
                for vi in f:
                    p = points[vi]
                    # Tangential direction toward circumcenter
                    toward = c_dir - np.dot(p, c_dir) * c_dir
                    toward_norm = np.linalg.norm(toward)
                    if toward_norm > 1e-10:
                        displacement[vi] += toward / toward_norm

            # Normalize and apply
            norm_disp = np.linalg.norm(displacement, axis=1, keepdims=True)
            norm_disp[norm_disp < 1e-15] = 1.0
            step_size = 0.015 * (1.0 - 0.5 * it / n_iter)
            points = points + step_size * displacement / norm_disp
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms

            # Check improvement
            rad = compute_covering_radius(points)
            if rad < best_rad - 1e-10:
                best_rad = rad
                best = points.copy()

        return best

    best_points = None
    best_rad = np.inf

    n_restarts = 5

    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break

        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            # Random start
            points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            points = best_points + 0.12 * np.random.randn(N, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms

        alpha = 0.04
        decay = 0.997
        max_iter = 400
        stall_count = 0

        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break

            # Get face radii for density-aware weighting
            face_radii, faces = compute_face_radii(points)

            # Standard repulsion step with density-aware weighting
            dot = np.dot(points, points.T)
            dot = np.clip(dot, -1.0, 1.0)
            np.fill_diagonal(dot, 1.0)
            dist_sq = 2.0 - 2.0 * dot
            dist_sq[dist_sq < 1e-10] = 1e-10
            w = 1.0 / (dist_sq ** 1.5)
            np.fill_diagonal(w, 0.0)

            # Boost repulsion for points that are vertices of large faces
            if face_radii is not None and faces is not None:
                vertex_max_rad = np.zeros(N)
                for fi in range(len(faces)):
                    for vi in faces[fi]:
                        if face_radii[fi] > vertex_max_rad[vi]:
                            vertex_max_rad[vi] = face_radii[fi]
                # Normalize and use as weight multiplier
                if np.max(vertex_max_rad) > 1e-10:
                    weight_boost = 1.0 + 0.5 * vertex_max_rad / np.max(vertex_max_rad)
                    w = w * weight_boost[:, np.newaxis]

            sum_w = np.sum(w, axis=1)
            sum_p_w = w @ points
            forces = points * sum_w[:, np.newaxis] - sum_p_w
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay

            if it % 3 == 0:
                rad = compute_covering_radius(points)
                if rad < best_rad:
                    best_rad = rad
                    best_points = points.copy()
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > 15:
                        # Reduce alpha more aggressively on stall
                        alpha *= 0.93
                        stall_count = 0

            if alpha < 1e-7:
                break

        # Polish phase for this restart
        if time.time() < deadline - 15.0 and best_points is not None:
            polished = polish_step(best_points.copy(), 0.005, 30)
            rad_polished = compute_covering_radius(polished)
            if rad_polished < best_rad:
                best_rad = rad_polished
                best_points = polished

            # Targeted face reduction
            if time.time() < deadline - 10.0:
                targeted = targeted_face_reduction(best_points.copy(), n_iter=80)
                rad_targeted = compute_covering_radius(targeted)
                if rad_targeted < best_rad:
                    best_rad = rad_targeted
                    best_points = targeted

            # Gradient descent on soft-max objective
            if time.time() < deadline - 8.0:
                grad_pts = gradient_descent_face(best_points.copy(), n_iter=120, lr=0.012, beta=80.0)
                rad_grad = compute_covering_radius(grad_pts)
                if rad_grad < best_rad:
                    best_rad = rad_grad
                    best_points = grad_pts

    # Final global polish
    if time.time() < deadline - 5.0 and best_points is not None:
        final = polish_step(best_points.copy(), 0.003, 60)
        rad_final = compute_covering_radius(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final

        # Final targeted reduction
        if time.time() < deadline - 3.0:
            final2 = targeted_face_reduction(best_points.copy(), n_iter=80)
            rad_final2 = compute_covering_radius(final2)
            if rad_final2 < best_rad:
                best_rad = rad_final2
                best_points = final2

        # Final gradient descent polish
        if time.time() < deadline - 2.0:
            final3 = gradient_descent_face(best_points.copy(), n_iter=100, lr=0.005, beta=100.0)
            rad_final3 = compute_covering_radius(final3)
            if rad_final3 < best_rad:
                best_rad = rad_final3
                best_points = final3

        # One more polish after gradient descent
        if time.time() < deadline - 1.0:
            final4 = polish_step(best_points.copy(), 0.001, 40)
            rad_final4 = compute_covering_radius(final4)
            if rad_final4 < best_rad:
                best_rad = rad_final4
                best_points = final4

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
