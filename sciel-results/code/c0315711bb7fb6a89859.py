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
        """Compute covering radius as max circumradius over convex hull faces."""
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

    def get_face_radii_and_vertices(pts):
        """Return face radii, faces, and per-vertex max face radius."""
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
            
            # Per-vertex max face radius
            vertex_max_rad = np.zeros(len(pts))
            for fi in range(len(faces)):
                for vi in faces[fi]:
                    if radii[fi] > vertex_max_rad[vi]:
                        vertex_max_rad[vi] = radii[fi]
            
            return radii, faces, vertex_max_rad
        except Exception:
            return None, None, None

    def repulsion_optimize(points, max_iter=600, deadline=None):
        """Repulsion-based optimization with face-radius-aware weighting."""
        if deadline is None:
            deadline = time.time() + 100
        
        alpha = 0.03
        decay = 0.996
        best_local = None
        best_local_rad = np.inf
        stall_count = 0
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute face radii for weighting
            face_radii, faces, vertex_max_rad = get_face_radii_and_vertices(points)
            
            # Standard repulsion
            dot = np.dot(points, points.T)
            dot = np.clip(dot, -1.0, 1.0)
            np.fill_diagonal(dot, 1.0)
            dist_sq = 2.0 - 2.0 * dot
            dist_sq[dist_sq < 1e-10] = 1e-10
            w = 1.0 / (dist_sq ** 1.5)
            np.fill_diagonal(w, 0.0)
            
            # Boost repulsion for points that are vertices of large faces
            if face_radii is not None and vertex_max_rad is not None:
                max_vmr = np.max(vertex_max_rad)
                if max_vmr > 1e-10:
                    weight_boost = 1.0 + 0.5 * vertex_max_rad / max_vmr
                    w = w * weight_boost[:, np.newaxis]
            
            sum_w = np.sum(w, axis=1)
            sum_p_w = w @ points
            forces = points * sum_w[:, np.newaxis] - sum_p_w
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay
            
            # Check improvement every few iterations
            if it % 5 == 0:
                rad = compute_covering_radius(points)
                if rad < best_local_rad:
                    best_local_rad = rad
                    best_local = points.copy()
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > 15:
                        alpha *= 0.9
                        stall_count = 0
            
            if alpha < 1e-7:
                break
        
        return best_local if best_local is not None else points, best_local_rad

    def targeted_face_reduction(points, max_iter=100, deadline=None):
        """Directly reduce the largest faces by moving their vertices."""
        if deadline is None:
            deadline = time.time() + 30
        
        best_local = points.copy()
        best_local_rad = compute_covering_radius(points)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            face_radii, faces, vertex_max_rad = get_face_radii_and_vertices(points)
            if face_radii is None:
                break
            
            # Find the largest face
            max_idx = np.argmax(face_radii)
            max_rad = face_radii[max_idx]
            
            if max_rad < 0.197:  # ~11.3 degrees in radians
                break
            
            # Get the three vertices of the largest face
            face = faces[max_idx]
            v0, v1, v2 = face[0], face[1], face[2]
            
            # Compute the circumcenter direction
            a = points[v0]
            b = points[v1]
            c = points[v2]
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            c_norm = np.linalg.norm(c_dir)
            
            if c_norm > 1e-10:
                c_dir = c_dir / c_norm
                
                # Move each vertex slightly toward the circumcenter
                # but only if it helps reduce the face radius
                move_alpha = 0.01 * (1.0 - it / max_iter)
                
                # Compute perpendicular components to move vertices
                # toward the center of the face
                for vi in [v0, v1, v2]:
                    p = points[vi]
                    # Project p onto the plane of the face and move toward center
                    # Simple approach: move slightly toward circumcenter
                    new_p = p + move_alpha * (c_dir - np.dot(p, c_dir) * p)
                    new_p = new_p / np.linalg.norm(new_p)
                    points[vi] = new_p
            
            # Also do a small global repulsion step to maintain balance
            dot = np.dot(points, points.T)
            dot = np.clip(dot, -1.0, 1.0)
            np.fill_diagonal(dot, 1.0)
            dist_sq = 2.0 - 2.0 * dot
            dist_sq[dist_sq < 1e-10] = 1e-10
            w = 1.0 / (dist_sq ** 2.0)
            np.fill_diagonal(w, 0.0)
            sum_w = np.sum(w, axis=1)
            sum_p_w = w @ points
            forces = points * sum_w[:, np.newaxis] - sum_p_w
            points = points + 0.005 * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
            rad = compute_covering_radius(points)
            if rad < best_local_rad:
                best_local_rad = rad
                best_local = points.copy()
        
        return best_local, best_local_rad

    def polish(points, n_iter=50, alpha=0.003):
        """Fine polishing with very small steps."""
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

    best_points = None
    best_rad = np.inf

    # Start with Fibonacci spiral
    points = fibonacci_spiral(N).astype(np.float64)
    
    # Phase 1: Repulsion optimization from Fibonacci start
    if time.time() < deadline - 60:
        points, rad = repulsion_optimize(points, max_iter=400, deadline=deadline - 40)
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()
    
    # Phase 2: Targeted face reduction
    if time.time() < deadline - 30 and best_points is not None:
        points, rad = targeted_face_reduction(best_points.copy(), max_iter=80, deadline=deadline - 20)
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()
    
    # Phase 3: Another round of repulsion from improved start
    if time.time() < deadline - 20 and best_points is not None:
        points, rad = repulsion_optimize(best_points.copy(), max_iter=300, deadline=deadline - 10)
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()
    
    # Phase 4: Final targeted reduction
    if time.time() < deadline - 10 and best_points is not None:
        points, rad = targeted_face_reduction(best_points.copy(), max_iter=50, deadline=deadline - 5)
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()
    
    # Phase 5: Final polish
    if time.time() < deadline - 5 and best_points is not None:
        points = polish(best_points.copy(), n_iter=30, alpha=0.002)
        rad = compute_covering_radius(points)
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()
    
    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)
    
    return best_points

# EVOLVE-BLOCK-END
