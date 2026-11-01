"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety

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

    def gradient_descent_repulsion(points, alpha_init, decay, max_iter, exponent=1.5):
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

    def soft_max_covering_radius_and_grad(pts, grid_pts, beta=20.0):
        """Compute soft-max covering radius and its gradient w.r.t. pts."""
        dot = np.dot(grid_pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)
        min_dists = np.min(ang_dists, axis=1)
        
        neg_beta_dists = -beta * min_dists
        max_neg = np.max(neg_beta_dists)
        log_sum = max_neg + np.log(np.sum(np.exp(neg_beta_dists - max_neg)))
        radius = -log_sum / beta
        
        weights = np.exp(neg_beta_dists - max_neg)
        weights /= np.sum(weights)
        
        min_indices = np.argmin(ang_dists, axis=1)
        
        dot_gi = dot[np.arange(len(grid_pts)), min_indices]
        denom = np.sqrt(np.maximum(1.0 - dot_gi * dot_gi, 1e-12))
        coeff = weights / denom
        
        grads = np.zeros_like(pts)
        np.add.at(grads, min_indices, -coeff[:, np.newaxis] * grid_pts)
        
        grads /= beta
        return radius, grads

    def gradient_descent_smooth(points, alpha_init, decay, max_iter, beta=20.0, grid_pts=None):
        """Optimize using smooth surrogate with gradient descent."""
        if grid_pts is None:
            lat = np.linspace(-np.pi/2, np.pi/2, 40)
            lon = np.linspace(0, 2*np.pi, 80, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z_grid = np.sin(LA)
            x_grid = np.cos(LA) * np.cos(LO)
            y_grid = np.cos(LA) * np.sin(LO)
            grid_pts = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])
        
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            radius, grads = soft_max_covering_radius_and_grad(points, grid_pts, beta)
            points = points + alpha * grads
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay
        return points

    best_points = None
    best_rad = np.inf

    # Precompute grids
    lat_c = np.linspace(-np.pi/2, np.pi/2, 40)
    lon_c = np.linspace(0, 2*np.pi, 80, endpoint=False)
    LA_c, LO_c = np.meshgrid(lat_c, lon_c, indexing='ij')
    z_gc = np.sin(LA_c)
    x_gc = np.cos(LA_c) * np.cos(LO_c)
    y_gc = np.cos(LA_c) * np.sin(LO_c)
    grid_coarse = np.column_stack([x_gc.ravel(), y_gc.ravel(), z_gc.ravel()])
    
    lat_f = np.linspace(-np.pi/2, np.pi/2, 80)
    lon_f = np.linspace(0, 2*np.pi, 160, endpoint=False)
    LA_f, LO_f = np.meshgrid(lat_f, lon_f, indexing='ij')
    z_gf = np.sin(LA_f)
    x_gf = np.cos(LA_f) * np.cos(LO_f)
    y_gf = np.cos(LA_f) * np.sin(LO_f)
    grid_fine = np.column_stack([x_gf.ravel(), y_gf.ravel(), z_gf.ravel()])

    # Phase 1: Multiple random starts with repulsion-based optimization
    n_restarts = 6
    exponents = [1.0, 1.5, 2.0, 3.0, 1.2, 1.8]
    
    for restart in range(n_restarts):
        if time.time() > deadline - 50.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            rng = np.random.RandomState(42 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100 + restart)
            points = points + 0.3 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            rng = np.random.RandomState(200 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 4:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(300 + restart)
            points = points + 0.5 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(400 + restart)
                points = best_points + 0.15 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        exp = exponents[restart % len(exponents)]
        points = gradient_descent_repulsion(points, 0.05, 0.995, 200, exp)
        rad = compute_covering_radius(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: L-BFGS-B optimization with smooth objective
    if best_points is not None and time.time() < deadline - 40.0:
        def objective_and_grad(flat_pts):
            pts = flat_pts.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            radius, grads = soft_max_covering_radius_and_grad(pts, grid_coarse, beta=30.0)
            # Project gradient to tangent space
            flat_grads = (grads - np.sum(grads * pts, axis=1, keepdims=True) * pts).flatten()
            return radius, flat_grads
        
        x0 = best_points.flatten()
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < best_rad:
                candidate = res.x.reshape(N, 3)
                candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
                rad = compute_covering_radius(candidate)
                if rad < best_rad:
                    best_rad = rad
                    best_points = candidate.copy()
        except Exception:
            pass

    # Phase 3: Smooth surrogate optimization with multi-scale annealing
    if best_points is not None and time.time() < deadline - 30.0:
        beta_schedule = [8.0, 15.0, 25.0, 40.0, 60.0]
        alpha_schedule = [0.03, 0.02, 0.015, 0.01, 0.008]
        decay_schedule = [0.997, 0.996, 0.995, 0.994, 0.993]
        iter_schedule = [80, 80, 60, 60, 50]
        
        for idx, (b, a0, dcy, n_it) in enumerate(zip(beta_schedule, alpha_schedule, decay_schedule, iter_schedule)):
            if time.time() > deadline - 15.0:
                break
            candidate = best_points.copy()
            candidate = gradient_descent_smooth(candidate, a0, dcy, n_it, beta=b, grid_pts=grid_coarse)
            rad = compute_covering_radius(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 4: Fine-grid polishing
    if best_points is not None and time.time() < deadline - 20.0:
        beta_fine = [30.0, 50.0, 70.0]
        alpha_fine = [0.01, 0.005, 0.003]
        decay_fine = [0.995, 0.993, 0.991]
        iter_fine = [60, 50, 40]
        
        for idx, (b, a0, dcy, n_it) in enumerate(zip(beta_fine, alpha_fine, decay_fine, iter_fine)):
            if time.time() > deadline - 10.0:
                break
            candidate = best_points.copy()
            candidate = gradient_descent_smooth(candidate, a0, dcy, n_it, beta=b, grid_pts=grid_fine)
            rad = compute_covering_radius(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 5: Targeted worst-facet refinement
    if best_points is not None and time.time() < deadline - 10.0:
        points = best_points.copy()
        
        try:
            hull = ConvexHull(points)
            faces = hull.simplices
        except Exception:
            faces = None
        
        max_iter = 30
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            if faces is None or it % 10 == 0:
                try:
                    hull = ConvexHull(points)
                    faces = hull.simplices
                except Exception:
                    break
            
            a = points[faces[:, 0]]
            b = points[faces[:, 1]]
            c = points[faces[:, 2]]
            
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
            
            worst_face_local = np.argmax(radii)
            worst_face = faces[worst_face_local]
            worst_radius = radii[worst_face_local]
            
            improved = False
            for vi in range(3):
                worst_idx = worst_face[vi]
                
                fa, fb, fc = points[worst_face[0]], points[worst_face[1]], points[worst_face[2]]
                ab2 = np.cross(fa, fb)
                bc2 = np.cross(fb, fc)
                ca2 = np.cross(fc, fa)
                c_dir2 = ab2 + bc2 + ca2
                norm_c2 = np.linalg.norm(c_dir2)
                if norm_c2 < 1e-10:
                    continue
                c_dir2 /= norm_c2
                
                for step in [0.02, 0.01, 0.005, 0.002]:
                    new_pt = points[worst_idx] + step * (c_dir2 - points[worst_idx])
                    new_pt /= np.linalg.norm(new_pt)
                    
                    old_pt = points[worst_idx].copy()
                    points[worst_idx] = new_pt
                    
                    new_rad = compute_covering_radius(points)
                    
                    if new_rad < best_rad - 1e-10:
                        best_rad = new_rad
                        best_points = points.copy()
                        improved = True
                        break
                    else:
                        points[worst_idx] = old_pt
                
                if improved:
                    break
            
            if not improved:
                if it % 15 == 8:
                    rng = np.random.RandomState(it * 7 + 13)
                    trial = best_points.copy()
                    trial += 0.03 * rng.randn(N, 3)
                    trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                    trial_rad = compute_covering_radius(trial)
                    if trial_rad < best_rad:
                        best_rad = trial_rad
                        best_points = trial.copy()
                        points = trial.copy()
                        faces = None

    # Final ultra-fine polish
    if best_points is not None and time.time() < deadline - 5.0:
        lat_uf = np.linspace(-np.pi/2, np.pi/2, 100)
        lon_uf = np.linspace(0, 2*np.pi, 200, endpoint=False)
        LA_uf, LO_uf = np.meshgrid(lat_uf, lon_uf, indexing='ij')
        z_guf = np.sin(LA_uf)
        x_guf = np.cos(LA_uf) * np.cos(LO_uf)
        y_guf = np.cos(LA_uf) * np.sin(LO_uf)
        grid_ultra_fine = np.column_stack([x_guf.ravel(), y_guf.ravel(), z_guf.ravel()])
        
        final = gradient_descent_smooth(best_points.copy(), 0.003, 0.995, 30, beta=60.0, grid_pts=grid_ultra_fine)
        rad_final = compute_covering_radius(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final.copy()

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
