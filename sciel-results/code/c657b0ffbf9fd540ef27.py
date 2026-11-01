"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

    def compute_covering_radius(pts):
        """Compute covering radius as max circumradius over convex hull faces."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices  # shape (F, 3)
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            # Compute circumcenter and circumradius for each face
            ab = b - a
            ac = c - a
            bc = c - b
            
            # Normal vector
            nrm = np.cross(ab, ac)
            nrm_norm = np.linalg.norm(nrm, axis=1, keepdims=True)
            nrm_norm[nrm_norm < 1e-12] = 1e-12
            nrm = nrm / nrm_norm
            
            # Barycentric coordinates or solve for circumcenter
            # The circumcenter lies on the line: center = a + t * nrm
            # |center - a|^2 = |center - b|^2
            # |a + t*nrm - a|^2 = |a + t*nrm - b|^2
            # t^2 = |ab + t*nrm|^2 = |ab|^2 + 2*t*(ab.nrm) + t^2
            # 0 = |ab|^2 + 2*t*(ab.nrm)
            # t = -|ab|^2 / (2*(ab.nrm))
            
            ab_dot_nrm = np.einsum('ij,ij->i', ab, nrm)
            bc_dot_nrm = np.einsum('ij,ij->i', bc, nrm)
            
            # Use a more stable formulation
            # Actually, let's use the standard circumcenter formula
            # d = 2 * (a x (b-c)) ... no let's just solve the system
            
            # Simpler: project to 2D or use the formula:
            # R = |a-b| |b-c| |c-a| / (4 * |a x b x c|) ... no that's for area
            # R = |ab| * |bc| * |ca| / (8 * Area)
            
            area = 0.5 * np.linalg.norm(nrm)  # This is area of triangle in 3D
            area[area < 1e-12] = 1e-12
            
            ab_len = np.linalg.norm(ab, axis=1)
            bc_len = np.linalg.norm(bc, axis=1)
            ca_len = np.linalg.norm(a - c, axis=1)
            
            circumradius = (ab_len * bc_len * ca_len) / (4.0 * area)
            
            # Check if the circumcenter's cap is empty (all other points are on the same side)
            # Center of sphere for this face:
            # a + t * nrm where t = -|ab|^2 / (2 * ab.nrm)
            t = -0.5 * (ab_len ** 2) / ab_dot_nrm
            t[ab_dot_nrm < 1e-12] = 0.0
            t = np.clip(t, -10, 10)  # Clamp for numerical stability
            
            centers = a + t[:, np.newaxis] * nrm
            centers_norm = np.linalg.norm(centers, axis=1, keepdims=True)
            centers_norm[centers_norm < 1e-12] = 1e-12
            centers = centers / centers_norm
            
            # Check if any other point is "outside" the cap (i.e., dot with center > cos(circumradius))
            # Actually, for a valid covering, the cap centered at 'centers[i]' with radius circumradius[i]
            # should contain no other points. So for all j != i's vertices, dot(pts[j], centers[i]) < cos(circumradius[i])
            
            cos_radius = np.cos(circumradius)
            dot_all = pts @ centers.T  # shape (N, F)
            
            # For each face f, check if any point (not the 3 vertices of f) has dot > cos_radius[f]
            # We'll mark invalid faces where this happens
            valid = np.ones(len(faces), dtype=bool)
            for f in range(len(faces)):
                # Exclude the 3 vertices of this face
                mask = np.ones(N, dtype=bool)
                mask[faces[f, 0]] = False
                mask[faces[f, 1]] = False
                mask[faces[f, 2]] = False
                if np.any(dot_all[mask, f] > cos_radius[f] + 1e-9):
                    valid[f] = False
            
            # If some faces are invalid, the covering radius is determined by the valid ones
            # Actually, if a face is invalid, it means the circumcenter cap is not empty,
            # so the true covering radius is determined by the largest facet that IS empty.
            # If all are invalid, something is wrong.
            if not np.any(valid):
                # Fallback to grid
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
                return np.max(min_dists)
            
            return np.max(circumradius[valid])
        except Exception:
            # Fallback to grid
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
            return np.max(min_dists)

    def repulsion_step(points, alpha):
        """Coulomb repulsion step on the sphere."""
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
        
        points += alpha * forces
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1e-12
        points /= norms
        return points

    def riemannian_gradient_step(points, alpha):
        """Gradient step using log-sum-exp surrogate for covering radius."""
        # Use a grid of sample points on the sphere
        lat = np.linspace(-np.pi/2, np.pi/2, 40)
        lon = np.linspace(0, 2*np.pi, 80, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Compute angular distances
        dot = np.dot(grid_pts, points.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)  # shape (M, N)
        
        # Log-sum-exp surrogate: log(sum(exp(beta * d))) / beta
        beta = 20.0
        max_d = np.max(ang_dists, axis=1, keepdims=True)
        lse = np.log(np.sum(np.exp(beta * (ang_dists - max_d)), axis=1, keepdims=True)) / beta + max_d.squeeze()
        
        # Gradient of lse wrt ang_dists: exp(beta * (d - max_d)) / sum(exp(beta * (d - max_d)))
        weights = np.exp(beta * (ang_dists - max_d))
        weights = weights / np.sum(weights, axis=1, keepdims=True)  # shape (M, N)
        
        # Gradient wrt points: for each grid point m, grad = sum_n weights[m,n] * grad(ang_dist[m,n])
        # grad of arccos(p·q) wrt p = -q / sqrt(1 - (p·q)^2)
        # So grad wrt points[i] = -sum_m weights[m,i] * grid_pts[m] / sqrt(1 - dot[m,i]^2)
        
        sqrt_term = np.sqrt(1.0 - dot * dot + 1e-10)  # shape (M, N)
        
        # grad[i] = -sum_m weights[m,i] * grid_pts[m] / sqrt_term[m,i]
        # = -grid_pts.T @ (weights / sqrt_term)
        
        grad_weights = weights / sqrt_term  # shape (M, N)
        grad_points = -grid_pts.T @ grad_weights  # shape (N, 3)
        
        # Riemannian gradient: project to tangent space
        # grad_tangent = grad - (grad · p) * p
        grad_dot_p = np.einsum('ij,ij->i', grad_points, points)
        grad_tangent = grad_points - grad_dot_p[:, np.newaxis] * points
        
        # Normalize for stability
        grad_norm = np.linalg.norm(grad_tangent, axis=1, keepdims=True)
        grad_norm[grad_norm < 1e-12] = 1e-12
        grad_tangent = grad_tangent / grad_norm
        
        points -= alpha * grad_tangent
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1e-12
        points /= norms
        return points

    # Initialize with Fibonacci spiral
    points = fibonacci_spiral(N).astype(np.float64)
    
    best_points = points.copy()
    best_rad = compute_covering_radius(best_points)
    
    # Phase 1: Repulsion-based relaxation
    alpha = 0.05
    decay = 0.995
    max_iter_rep = 500
    
    for it in range(max_iter_rep):
        if time.time() > deadline - 10.0:
            break
        
        points = repulsion_step(points, alpha)
        alpha *= decay
        
        if it % 20 == 0:
            rad = compute_covering_radius(points)
            if rad < best_rad:
                best_rad = rad
                best_points = points.copy()
        
        if alpha < 1e-5:
            break
    
    # Phase 2: Riemannian gradient descent on log-sum-exp surrogate
    alpha_gd = 0.02
    decay_gd = 0.99
    max_iter_gd = 300
    
    for it in range(max_iter_gd):
        if time.time() > deadline - 5.0:
            break
        
        best_points = riemannian_gradient_step(best_points, alpha_gd)
        alpha_gd *= decay_gd
        
        if it % 10 == 0:
            rad = compute_covering_radius(best_points)
            if rad < best_rad:
                best_rad = rad
                # Don't update best_rad here, just track
        
    # Final refinement
    if time.time() < deadline - 2.0:
        alpha_fin = 0.005
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            best_points = riemannian_gradient_step(best_points, alpha_fin)
            alpha_fin *= 0.95
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    best_points /= norms
    
    return best_points

# EVOLVE-BLOCK-END
