"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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

    def make_grid(n_lat, n_lon):
        lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
        lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    grid_coarse = make_grid(20, 40)
    grid_medium = make_grid(40, 80)
    grid_pts = make_grid(80, 160)

    def lloyd_iteration(points, n_iters=50, grid=None):
        """Lloyd-like refinement: move each point to the centroid of its Voronoi cell on the sphere."""
        if grid is None:
            grid = grid_pts
        for _ in range(n_iters):
            if time.time() > deadline - 10.0:
                break
            
            dot = np.dot(grid, points.T)
            dot = np.clip(dot, -1.0, 1.0)
            nearest_idx = np.argmax(dot, axis=1)
            
            new_points = np.zeros_like(points)
            for i in range(N):
                mask = nearest_idx == i
                if np.any(mask):
                    cell_pts = grid[mask]
                    d = np.clip(1.0 - dot[mask, i], 1e-10, None)
                    w = 1.0 / d
                    centroid = np.sum(cell_pts * w[:, np.newaxis], axis=0)
                    norm = np.linalg.norm(centroid)
                    if norm > 1e-10:
                        new_points[i] = centroid / norm
                    else:
                        new_points[i] = points[i]
                else:
                    new_points[i] = points[i]
            
            alpha = 0.5
            points = (1 - alpha) * points + alpha * new_points
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        return points

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

    def facet_soft_max_objective(pts_flat, beta=100.0):
        """
        Smooth surrogate: soft-max of facet circumradii.
        This directly targets the true objective (convex hull facet circumradii).
        """
        pts = pts_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms

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
            norms_c = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms_c > 1e-10
            
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms_c[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0 + 1e-12, 1.0 - 1e-12)
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
            
            m = np.max(radii)
            if m < 1e-15:
                return 0.0, np.zeros(3 * N)
            
            shifted = radii - m
            exp_vals = np.exp(beta * shifted)
            sum_exp = np.sum(exp_vals)
            obj = m + np.log(sum_exp) / beta
            
            weights = exp_vals / sum_exp
            significant = weights > 1e-8
            if not np.any(significant):
                return obj, np.zeros(3 * N)
            
            # Compute gradient with respect to points
            # For each facet, the circumcenter direction is c_dir_valid
            # Moving a vertex toward the circumcenter reduces the circumradius
            grad = np.zeros_like(pts)
            
            for k in np.where(significant)[0]:
                w_k = weights[k]
                face = faces[k]
                i, j, l = face[0], face[1], face[2]
                
                if valid[k]:
                    c_dir_k = c_dir_valid[k] if np.any(valid) else c_dir[k] / norms_c[k]
                    # Gradient of circumradius w.r.t. each vertex
                    # The circumcenter is c_dir_k (unit vector)
                    # Moving vertex p_i toward c_dir_k reduces the distance
                    # The gradient contribution is proportional to -w_k * (c_dir_k - p_i) projected tangentially
                    for idx in [i, j, l]:
                        p_idx = pts[idx]
                        u = np.dot(p_idx, c_dir_k)
                        g_perp = c_dir_k - u * p_idx
                        denom = np.sqrt(max(1.0 - u * u, 1e-12))
                        grad[idx] += w_k * g_perp / denom
                
                # Note: For degenerate facets, we skip gradient contribution
                # as the direction is less clear
            
            # Project gradient to tangent space
            grad_dot_p = np.sum(grad * pts, axis=1, keepdims=True)
            grad_tan = grad - grad_dot_p * pts
            
            grad_flat = (grad_tan / norms).ravel()
            
            return obj, grad_flat
        except Exception:
            # Fallback to grid-based soft-max
            return grid_soft_max_objective(pts_flat, beta, grid_pts)

    def grid_soft_max_objective(pts_flat, beta=100.0, grid=None):
        """
        Smooth surrogate: soft-max of angular distances over the grid.
        """
        if grid is None:
            grid = grid_pts
        pts = pts_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms

        dot = np.dot(grid, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)

        min_dists = np.min(ang_dists, axis=1)
        nearest_idx = np.argmin(ang_dists, axis=1)

        m = np.max(min_dists)
        if m < 1e-15:
            return 0.0, np.zeros(3 * N)
        
        shifted = min_dists - m
        exp_vals = np.exp(beta * shifted)
        sum_exp = np.sum(exp_vals)
        obj = m + np.log(sum_exp) / beta

        weights = exp_vals / sum_exp
        significant = weights > 1e-8
        if not np.any(significant):
            return obj, np.zeros(3 * N)

        sig_weights = weights[significant]
        sig_idx = nearest_idx[significant]
        sig_grid = grid[significant]

        u = np.sum(sig_grid * pts[sig_idx], axis=1)
        u = np.clip(u, -1.0 + 1e-12, 1.0 - 1e-12)
        g_perp = sig_grid - u[:, np.newaxis] * pts[sig_idx]
        denom = np.sqrt(1.0 - u * u)
        denom[denom < 1e-12] = 1e-12
        
        contrib = -sig_weights[:, np.newaxis] * g_perp / denom[:, np.newaxis]
        
        grad = np.zeros_like(pts)
        np.add.at(grad, sig_idx, contrib)
        
        grad_dot_p = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tan = grad - grad_dot_p * pts

        grad_flat = (grad_tan / norms).ravel()

        return obj, grad_flat

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple random starts with Lloyd refinement
    n_restarts = 6
    for restart in range(n_restarts):
        if time.time() > deadline - 40.0:
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
            points = points + 0.2 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            rng = np.random.RandomState(200)
            centers = rng.randn(10, 3)
            centers /= np.linalg.norm(centers, axis=1, keepdims=True)
            points = np.zeros((N, 3))
            for i in range(N):
                c = centers[i % 10]
                points[i] = c + 0.3 * rng.randn(3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 4:
            if best_points is not None:
                rng = np.random.RandomState(300 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)
        else:
            rng = np.random.RandomState(400 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Coarse Lloyd
        points = lloyd_iteration(points, n_iters=20, grid=grid_coarse)
        # Medium Lloyd
        points = lloyd_iteration(points, n_iters=30, grid=grid_medium)
        # Fine Lloyd
        points = lloyd_iteration(points, n_iters=20, grid=grid_pts)
        
        rad = compute_covering_radius(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Facet-based soft-max L-BFGS-B optimization
    if best_points is not None and time.time() < deadline - 25.0:
        x0 = best_points.ravel()
        
        # Multi-scale beta schedule
        beta_schedule = [50.0, 100.0, 200.0, 300.0]
        iters_per_beta = [150, 100, 80, 60]
        
        for beta, maxiter in zip(beta_schedule, iters_per_beta):
            if time.time() > deadline - 12.0:
                break
            
            def obj_and_grad_facet(x, b=beta):
                if time.time() > deadline - 5.0:
                    return 1e6, np.zeros_like(x)
                val, grad = facet_soft_max_objective(x, beta=b)
                return val, grad

            try:
                result = minimize(
                    obj_and_grad_facet,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8,
                             'maxfun': maxiter, 'disp': False}
                )
                opt_pts = result.x.reshape(N, 3)
                opt_norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
                opt_norms[opt_norms < 1e-15] = 1.0
                opt_pts = opt_pts / opt_norms

                rad_opt = compute_covering_radius(opt_pts)
                if rad_opt < best_rad:
                    best_rad = rad_opt
                    best_points = opt_pts.copy()
                    x0 = opt_pts.ravel()
            except Exception:
                pass

    # Phase 3: Grid-based soft-max L-BFGS-B for robustness
    if best_points is not None and time.time() < deadline - 15.0:
        x0 = best_points.ravel()
        beta = 150.0

        def obj_and_grad_grid(x):
            if time.time() > deadline - 5.0:
                return 1e6, np.zeros_like(x)
            val, grad = grid_soft_max_objective(x, beta=beta, grid=grid_pts)
            return val, grad

        try:
            result = minimize(
                obj_and_grad_grid,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8,
                         'maxfun': 100, 'disp': False}
            )
            opt_pts = result.x.reshape(N, 3)
            opt_norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            opt_norms[opt_norms < 1e-15] = 1.0
            opt_pts = opt_pts / opt_norms

            rad_opt = compute_covering_radius(opt_pts)
            if rad_opt < best_rad:
                best_rad = rad_opt
                best_points = opt_pts.copy()
        except Exception:
            pass

    # Phase 4: Final Lloyd + repulsion polish
    if best_points is not None and time.time() < deadline - 10.0:
        points = best_points.copy()
        points = lloyd_iteration(points, n_iters=25, grid=grid_pts)
        
        alpha = 0.01
        decay = 0.998
        for _ in range(80):
            if time.time() > deadline - 5.0:
                break
            forces = repulsion_force(points, 1.5)
            points = points + alpha * forces
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            alpha *= decay
        
        points = lloyd_iteration(points, n_iters=20, grid=grid_pts)
        
        rad = compute_covering_radius(points)
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 5: Very high-beta facet optimization for final polish
    if best_points is not None and time.time() < deadline - 5.0:
        x0 = best_points.ravel()
        beta = 400.0

        def obj_and_grad_final(x):
            if time.time() > deadline - 2.0:
                return 1e6, np.zeros_like(x)
            val, grad = facet_soft_max_objective(x, beta=beta)
            return val, grad

        try:
            result = minimize(
                obj_and_grad_final,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10,
                         'maxfun': 50, 'disp': False}
            )
            opt_pts = result.x.reshape(N, 3)
            opt_norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            opt_norms[opt_norms < 1e-15] = 1.0
            opt_pts = opt_pts / opt_norms

            rad_opt = compute_covering_radius(opt_pts)
            if rad_opt < best_rad:
                best_rad = rad_opt
                best_points = opt_pts.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
