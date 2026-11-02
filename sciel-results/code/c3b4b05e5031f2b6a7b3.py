"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

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

    def compute_covering_radius_hull(pts):
        """Compute exact covering radius via convex hull facets."""
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
            
            return float(np.max(radii))
        except Exception:
            return np.inf

    def get_worst_face(pts):
        """Return the index of the worst face and its radius."""
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
            
            max_idx = np.argmax(radii)
            return max_idx, radii[max_idx], faces[max_idx]
        except Exception:
            return -1, np.inf, None

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
            if time.time() > deadline - 10.0:
                break
            forces = repulsion_force(points, exponent)
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay
        return points

    def local_facet_optimization(pts, max_iters=500):
        """
        Local optimization: repeatedly find the worst facet and move its vertices
        towards the facet normal to reduce the circumradius.
        """
        pts = pts.copy()
        current_rad = compute_covering_radius_hull(pts)
        step = 0.02
        no_improve_count = 0
        max_no_improve = 50

        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            
            face_idx, rad, face = get_worst_face(pts)
            if face is None or rad >= current_rad - 1e-12:
                if face is not None:
                    current_rad = rad
                break
            
            i, j, k = face
            a = pts[i].copy()
            b = pts[j].copy()
            c = pts[k].copy()
            
            # Compute the circumcenter direction (normal to the face)
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            c_norm = np.linalg.norm(c_dir)
            
            if c_norm < 1e-10:
                # Degenerate face, use centroid direction
                mid = (a + b + c) / 3.0
                mid /= np.linalg.norm(mid)
                c_dir = mid
                c_norm = 1.0
            else:
                c_dir /= c_norm
            
            # Move each vertex towards the circumcenter direction
            old_pts = pts.copy()
            improved = False
            
            for idx, p in [(i, a), (j, b), (k, c)]:
                # Tangent direction towards circumcenter
                tangent = c_dir - np.dot(c_dir, p) * p
                t_norm = np.linalg.norm(tangent)
                if t_norm > 1e-10:
                    tangent /= t_norm
                    new_p = p + step * tangent
                    new_p /= np.linalg.norm(new_p)
                    pts[idx] = new_p
                    improved = True
            
            if not improved:
                break
            
            new_rad = compute_covering_radius_hull(pts)
            if new_rad < current_rad - 1e-12:
                current_rad = new_rad
                no_improve_count = 0
                # Try to increase step slightly if improving well
                if new_rad < current_rad - 0.001:
                    step = min(step * 1.05, 0.05)
            else:
                # Revert
                pts = old_pts
                no_improve_count += 1
                step *= 0.7
                if no_improve_count > max_no_improve:
                    break
        
        return pts, current_rad

    def anneal_optimization(pts, max_iters=1000):
        """
        Simulated annealing style: randomly perturb points and accept if better.
        Focus on moving points near the worst facet.
        """
        pts = pts.copy()
        current_rad = compute_covering_radius_hull(pts)
        best_rad = current_rad
        best_pts = pts.copy()
        temperature = 0.01
        cooling = 0.995
        rng = np.random.RandomState(42)

        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            
            # Find worst face
            face_idx, rad, face = get_worst_face(pts)
            if face is None:
                break
            
            i, j, k = face
            
            # Choose which point to move: one of the face vertices or a random point
            if rng.random() < 0.7:
                # Move one of the face vertices
                candidates = [i, j, k]
                idx = candidates[rng.randint(3)]
            else:
                # Move a random point
                idx = rng.randint(N)
            
            p = pts[idx].copy()
            
            # Generate a random perturbation
            scale = temperature * 2.0
            noise = rng.randn(3) * scale
            new_p = p + noise
            new_p /= np.linalg.norm(new_p)
            
            old_p = pts[idx].copy()
            pts[idx] = new_p
            
            new_rad = compute_covering_radius_hull(pts)
            
            # Accept if better or with probability based on temperature
            delta = new_rad - current_rad
            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                current_rad = new_rad
                if new_rad < best_rad:
                    best_rad = new_rad
                    best_pts = pts.copy()
            else:
                pts[idx] = old_p
            
            temperature *= cooling
        
        return best_pts, best_rad

    def targeted_move(pts, max_iters=300):
        """
        For the worst facet, try to find a better position for one of its vertices
        by doing a local search in the tangent space.
        """
        pts = pts.copy()
        current_rad = compute_covering_radius_hull(pts)
        best_rad = current_rad
        best_pts = pts.copy()
        rng = np.random.RandomState(123)

        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            
            face_idx, rad, face = get_worst_face(pts)
            if face is None:
                break
            
            i, j, k = face
            
            # Compute circumcenter
            a = pts[i]
            b = pts[j]
            c = pts[k]
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            c_norm = np.linalg.norm(c_dir)
            
            if c_norm < 1e-10:
                mid = (a + b + c) / 3.0
                mid /= np.linalg.norm(mid)
                c_dir = mid
            else:
                c_dir /= c_norm
            
            # Try moving each vertex towards the circumcenter with random perturbation
            improved = False
            for idx, p in [(i, a), (j, b), (k, c)]:
                # Base direction towards circumcenter
                tangent = c_dir - np.dot(c_dir, p) * p
                t_norm = np.linalg.norm(tangent)
                if t_norm < 1e-10:
                    continue
                tangent /= t_norm
                
                # Add random perturbation
                noise = rng.randn(3) * 0.01
                move_dir = tangent + noise
                move_dir /= np.linalg.norm(move_dir)
                
                step = 0.01 + rng.random() * 0.02
                new_p = p + step * move_dir
                new_p /= np.linalg.norm(new_p)
                
                old_p = pts[idx].copy()
                pts[idx] = new_p
                
                new_rad = compute_covering_radius_hull(pts)
                if new_rad < best_rad - 1e-12:
                    best_rad = new_rad
                    best_pts = pts.copy()
                    improved = True
                else:
                    pts[idx] = old_p
            
            if not improved and it > 100:
                break
        
        return best_pts, best_rad

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion-based gradient descent
    n_restarts = 6
    exponents = [1.0, 1.5, 2.0, 1.2, 1.8, 2.5]
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60.0:
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
            if best_points is not None:
                rng = np.random.RandomState(300)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 4:
            # Golden ratio based different spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = 2.399963229728653 * i  # Different golden ratio multiple
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            if best_points is not None:
                rng = np.random.RandomState(500)
                points = best_points.copy()
                half = N // 2
                points[:half] += 0.05 * rng.randn(half, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        exp = exponents[restart % len(exponents)]
        points = gradient_descent(points, 0.05, 0.995, 200, exp)
        rad = compute_covering_radius_hull(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Local facet optimization on best point
    if best_points is not None and time.time() < deadline - 30.0:
        refined, rad = local_facet_optimization(best_points.copy(), max_iters=500)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    # Phase 3: Targeted move optimization
    if best_points is not None and time.time() < deadline - 20.0:
        refined, rad = targeted_move(best_points.copy(), max_iters=300)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    # Phase 4: Another round of local facet optimization
    if best_points is not None and time.time() < deadline - 15.0:
        refined, rad = local_facet_optimization(best_points.copy(), max_iters=300)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    # Phase 5: Annealing
    if best_points is not None and time.time() < deadline - 10.0:
        refined, rad = anneal_optimization(best_points.copy(), max_iters=500)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    # Phase 6: Final targeted refinement
    if best_points is not None and time.time() < deadline - 5.0:
        refined, rad = local_facet_optimization(best_points.copy(), max_iters=200)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    # Final check
    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
