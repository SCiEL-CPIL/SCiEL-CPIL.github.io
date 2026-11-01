"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    
    def compute_face_radii_and_circumcenters(pts):
        """Return arrays of face circumradii and circumcenters for gradient-aware optimization."""
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
            circumcenters = np.zeros((len(faces), 3))
            valid = norms > 1e-10
            
            if np.any(valid):
                cc = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
                circumcenters[valid] = cc
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                circumcenters[~valid] = (a_deg + b_deg + c_deg) / 3.0
                norms_cc = np.linalg.norm(circumcenters[~valid], axis=1, keepdims=True)
                circumcenters[~valid] /= norms_cc
            
            return radii, circumcenters, faces
        except Exception:
            return None, None, None
    
    def facet_gradient_step(points, alpha, radii, circumcenters, faces, beta=20.0):
        """
        Move each vertex toward the circumcenters of facets it belongs to,
        weighted by a soft-max over facet radii to smooth the max.
        Vectorized implementation for speed.
        """
        n_points = len(points)
        n_faces = len(faces)
        forces = np.zeros((n_points, 3))
        
        if n_faces == 0:
            return points
        
        # Soft-max weights: w_i = exp(beta * r_i) / sum(exp(beta * r_j))
        max_r = np.max(radii)
        weights = np.exp(beta * (radii - max_r))
        weights /= np.sum(weights)
        
        # Vectorized force computation
        # For each face fi, for each vertex vi in faces[fi]:
        # force = circumcenters[fi] - points[vi], projected tangentially
        # forces[vi] += weights[fi] * force
        
        # Get all vertex indices from faces
        all_vis = faces.ravel()  # shape (3*n_faces,)
        # Reshape weights to broadcast: each face's weight applied to its 3 vertices
        w_expanded = np.repeat(weights, 3)  # shape (3*n_faces,)
        
        # Get corresponding points and circumcenters
        pts_expanded = points[all_vis]  # shape (3*n_faces, 3)
        cc_expanded = np.repeat(circumcenters, 3, axis=0)  # shape (3*n_faces, 3)
        
        # Compute forces: cc - p, then project to tangent plane
        forces_raw = cc_expanded - pts_expanded  # shape (3*n_faces, 3)
        # Project out radial component: force - (force . p) * p
        radial_dots = np.sum(forces_raw * pts_expanded, axis=1, keepdims=True)  # shape (3*n_faces, 1)
        forces_tangent = forces_raw - radial_dots * pts_expanded  # shape (3*n_faces, 3)
        
        # Weighted forces
        weighted_forces = w_expanded[:, np.newaxis] * forces_tangent  # shape (3*n_faces, 3)
        
        # Accumulate forces for each vertex
        for i, vi in enumerate(all_vis):
            forces[vi] += weighted_forces[i]
        
        points = points + alpha * forces
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        return points
    
    def repulsion_step(points, alpha):
        """Standard repulsion step to prevent clustering."""
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
    
    def polish_step(points, alpha, n_iter=50):
        """High-precision polish with very small steps."""
        for it in range(n_iter):
            if time.time() > deadline - 3.0:
                break
            radii, ccs, faces = compute_face_radii_and_circumcenters(points)
            if radii is not None:
                points = facet_gradient_step(points, alpha, radii, ccs, faces, beta=35.0)
            points = repulsion_step(points, alpha * 0.05)
        return points
    
    def scipy_optimize(points, max_iter=200):
        """Use scipy minimize with BFGS on the soft-max covering radius."""
        def objective_flat(x):
            pts = x.reshape(-1, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            radii, _, _ = compute_face_radii_and_circumcenters(pts)
            if radii is None:
                return 1.0
            # Soft-max with high beta for smooth approximation of max
            beta = 50.0
            max_r = np.max(radii)
            soft_max = max_r + np.log(np.sum(np.exp(beta * (radii - max_r)))) / beta
            return soft_max
        
        def gradient_flat(x):
            pts = x.reshape(-1, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            radii, ccs, faces = compute_face_radii_and_circumcenters(pts)
            if radii is None:
                return np.zeros_like(x)
            
            beta = 50.0
            max_r = np.max(radii)
            weights = np.exp(beta * (radii - max_r))
            weights /= np.sum(weights)
            
            grad_pts = np.zeros_like(pts)
            for fi in range(len(faces)):
                w = weights[fi]
                if w < 1e-8:
                    continue
                cc = ccs[fi]
                for vi in faces[fi]:
                    p = pts[vi]
                    force = cc - p
                    radial = np.dot(force, p) * p
                    force = force - radial
                    grad_pts[vi] += w * force
            
            return grad_pts.ravel()
        
        x0 = points.ravel()
        result = minimize(objective_flat, x0, jac=gradient_flat, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-10})
        
        if result.success or result.fun < 1.0:
            pts = result.x.reshape(-1, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            return pts
        return points
    
    best_points = None
    best_rad = np.inf
    
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            # Golden ratio spiral with slight perturbation
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 0.1 * np.random.randn(N)
            r = np.sqrt(np.maximum(0, 1.0 - z * z))
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 3:
            # Random start with larger perturbation
            points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            points = points + 0.15 * np.random.randn(N, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        else:
            points = best_points + 0.06 * np.random.randn(N, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        
        alpha = 0.02
        decay = 0.997
        max_iter = 600
        stall_count = 0
        last_rad = np.inf
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            radii, ccs, faces = compute_face_radii_and_circumcenters(points)
            
            if radii is not None:
                points = facet_gradient_step(points, alpha, radii, ccs, faces, beta=25.0)
                points = repulsion_step(points, alpha * 0.03)
            
            alpha *= decay
            
            if it % 3 == 0:
                rad = compute_covering_radius(points)
                if rad < best_rad:
                    best_rad = rad
                    best_points = points.copy()
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > 20:
                        alpha *= 0.9
                        stall_count = 0
            
            if alpha < 1e-9:
                break
        
        # Polish phase
        if time.time() < deadline - 15.0 and best_points is not None:
            polished = polish_step(best_points.copy(), 0.003, 60)
            rad_polished = compute_covering_radius(polished)
            if rad_polished < best_rad:
                best_rad = rad_polished
                best_points = polished
    
    # Final high-precision polish
    if time.time() < deadline - 10.0 and best_points is not None:
        final = polish_step(best_points.copy(), 0.0015, 80)
        rad_final = compute_covering_radius(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final
    
    if time.time() < deadline - 5.0 and best_points is not None:
        final2 = polish_step(best_points.copy(), 0.0008, 100)
        rad_final2 = compute_covering_radius(final2)
        if rad_final2 < best_rad:
            best_rad = rad_final2
            best_points = final2
    
    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
