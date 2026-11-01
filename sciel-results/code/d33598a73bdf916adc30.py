"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_and_worst_face(pts):
        """Compute covering radius and return (radius, worst_face_indices, hull)."""
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
            
            max_idx = np.argmax(radii)
            return float(radii[max_idx]), faces[max_idx], hull, radii
        except Exception:
            return np.inf, None, None, None

    def repulsion_force(points, exponent=1.5):
        """Compute repulsion forces between all point pairs."""
        dot = np.dot(points, points.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        w = 1.0 / (dist_sq ** exponent)
        np.fill_diagonal(w, 0.0)
        sum_w = np.sum(w, axis=1)
        sum_p_w = w @ points
        forces = points * sum_w[:, np.newaxis] - sum_p_w
        return forces

    def gradient_descent(points, alpha_init, decay, max_iter, exponent=1.5):
        """Optimize points via gradient descent on the repulsion energy."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            forces = repulsion_force(points, exponent)
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay
        return points

    def targeted_refinement(points, max_iters=200):
        """Refine by targeting the worst facet: move its 3 vertices to reduce circumradius."""
        pts = points.copy()
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            rad, face, hull, all_radii = compute_covering_radius_and_worst_face(pts)
            if face is None:
                break
            
            i, j, k = face
            a = pts[i].copy()
            b = pts[j].copy()
            c = pts[k].copy()
            
            # Compute the circumcenter direction for this triangle
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            c_norm = np.linalg.norm(c_dir)
            
            if c_norm < 1e-10:
                # Degenerate - try moving points apart
                mid = (a + b + c) / 3.0
                mid /= np.linalg.norm(mid)
                # Move each point away from the centroid direction
                for idx, p in [(i, a), (j, b), (k, c)]:
                    direction = p - 0.1 * mid
                    direction /= np.linalg.norm(direction)
                    pts[idx] = direction
                continue
            
            c_dir /= c_norm
            # The circumcenter is in direction c_dir from origin
            # To reduce the circumradius, we want to move points toward the circumcenter
            # but projected onto the sphere tangent plane
            
            # Move each vertex slightly toward the circumcenter
            step = 0.005
            for idx, p in [(i, a), (j, b), (k, c)]:
                # Project circumcenter direction to tangent plane at p
                tangent = c_dir - np.dot(c_dir, p) * p
                t_norm = np.linalg.norm(tangent)
                if t_norm > 1e-10:
                    tangent /= t_norm
                    new_p = p + step * tangent
                    new_p /= np.linalg.norm(new_p)
                    pts[idx] = new_p
            
            # Check if improvement
            new_rad, _, _, _ = compute_covering_radius_and_worst_face(pts)
            if new_rad >= rad:
                # Revert
                pts[i] = a
                pts[j] = b
                pts[k] = c
                # Try smaller step
                step *= 0.5
                if step < 1e-6:
                    break
                for idx, p in [(i, a), (j, b), (k, c)]:
                    tangent = c_dir - np.dot(c_dir, p) * p
                    t_norm = np.linalg.norm(tangent)
                    if t_norm > 1e-10:
                        tangent /= t_norm
                        new_p = p + step * tangent
                        new_p /= np.linalg.norm(new_p)
                        pts[idx] = new_p
                new_rad, _, _, _ = compute_covering_radius_and_worst_face(pts)
                if new_rad >= rad:
                    pts[i] = a
                    pts[j] = b
                    pts[k] = c
                    break
        
        return pts

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with gradient descent
    n_restarts = 5
    exponents = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    for restart in range(n_restarts):
        if time.time() > deadline - 40.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            rng = np.random.RandomState(42)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100)
            points = points + 0.3 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(200)
            points = points + 0.15 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(300)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        exp = exponents[restart % len(exponents)]
        points = gradient_descent(points, 0.05, 0.995, 200, exp)
        rad, _, _, _ = compute_covering_radius_and_worst_face(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Targeted refinement on the worst facet
    if best_points is not None and time.time() < deadline - 10.0:
        refined = targeted_refinement(best_points.copy(), max_iters=300)
        rad, _, _, _ = compute_covering_radius_and_worst_face(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    # Phase 3: More gradient descent with smaller steps
    if best_points is not None and time.time() < deadline - 8.0:
        for trial_alpha in [0.005, 0.002, 0.001]:
            if time.time() > deadline - 5.0:
                break
            candidate = gradient_descent(best_points.copy(), trial_alpha, 0.99, 150, 1.5)
            rad, _, _, _ = compute_covering_radius_and_worst_face(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 4: Another round of targeted refinement
    if best_points is not None and time.time() < deadline - 5.0:
        refined = targeted_refinement(best_points.copy(), max_iters=200)
        rad, _, _, _ = compute_covering_radius_and_worst_face(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    # Final check
    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
