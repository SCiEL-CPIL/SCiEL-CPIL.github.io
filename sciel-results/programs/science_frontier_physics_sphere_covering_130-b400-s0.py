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
    
    def repulsion_step_density_aware(points, alpha, face_radii=None, faces=None):
        """Repulsion with density-aware weighting: points in large-cap regions get stronger push."""
        dot = np.dot(points, points.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        w = 1.0 / (dist_sq ** 1.5)
        np.fill_diagonal(w, 0.0)
        
        # Density-aware weighting: weight vertices in large faces more
        if face_radii is not None and faces is not None:
            vertex_max_rad = np.zeros(len(points))
            for fi in range(len(faces)):
                for vi in faces[fi]:
                    if face_radii[fi] > vertex_max_rad[vi]:
                        vertex_max_rad[vi] = face_radii[fi]
            if np.max(vertex_max_rad) > 1e-10:
                weight_boost = 1.0 + 0.5 * vertex_max_rad / np.max(vertex_max_rad)
                w = w * weight_boost[:, np.newaxis]
        
        sum_w = np.sum(w, axis=1)
        sum_p_w = w @ points
        forces = points * sum_w[:, np.newaxis] - sum_p_w
        points = points + alpha * forces
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        return points
    
    def polish_step(points, alpha, n_iter=50):
        """High-precision polish with very small steps."""
        for _ in range(n_iter):
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
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        return points
    
    def targeted_face_reduction(points, n_iter=100):
        """Directly reduce the largest facet radii by pushing vertices of large facets apart."""
        best = points.copy()
        best_rad = compute_covering_radius(best)
        
        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break
            
            face_radii, faces = compute_face_radii(points)
            if face_radii is None:
                break
            
            # Find the largest facets and push their vertices apart
            top_k = min(10, len(faces))
            top_indices = np.argsort(face_radii)[-top_k:]
            
            # For each large facet, compute the circumcenter direction and push vertices away
            displacement = np.zeros_like(points)
            for fi in top_indices:
                if face_radii[fi] < best_rad * 0.8:
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
                # Push each vertex away from the circumcenter
                for vi in f:
                    # Gradient: move vertex away from circumcenter
                    away = points[vi] - np.dot(points[vi], c_dir) * c_dir
                    away = away / (np.linalg.norm(away) + 1e-15)
                    # Also add a small component along the vertex direction (to maintain normalization)
                    displacement[vi] += away * 0.1 + points[vi] * 0.01
            
            # Normalize and apply
            norm_disp = np.linalg.norm(displacement, axis=1, keepdims=True)
            norm_disp[norm_disp < 1e-15] = 1.0
            step_size = 0.02 * (1.0 - it / n_iter)
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
        last_improvement = np.inf
        
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
    
    # Final global polish
    if time.time() < deadline - 5.0 and best_points is not None:
        final = polish_step(best_points.copy(), 0.002, 50)
        rad_final = compute_covering_radius(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final
        
        # Final targeted reduction
        if time.time() < deadline - 3.0:
            final2 = targeted_face_reduction(best_points.copy(), n_iter=60)
            rad_final2 = compute_covering_radius(final2)
            if rad_final2 < best_rad:
                best_rad = rad_final2
                best_points = final2
    
    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)
    
    return best_points

# EVOLVE-BLOCK-END
