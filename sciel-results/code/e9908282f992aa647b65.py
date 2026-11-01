"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety

    # Precompute a high-resolution grid on the sphere
    n_lat = 80
    n_lon = 160
    lat = np.linspace(-np.pi/2 + 0.5*np.pi/n_lat, np.pi/2 - 0.5*np.pi/n_lat, n_lat)
    lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    LA, LO = np.meshgrid(lat, lon, indexing='ij')
    gz = np.sin(LA)
    gx = np.cos(LA) * np.cos(LO)
    gy = np.cos(LA) * np.sin(LO)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    grid_pts = grid_pts / np.linalg.norm(grid_pts, axis=1, keepdims=True)
    n_grid = len(grid_pts)

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius(pts):
        """Compute covering radius using convex hull face circumradii."""
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
            dot = np.dot(grid_pts, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def soft_min_covering(pts, beta=30.0):
        """
        Compute soft-min of distances from each grid point to nearest config point,
        then return the soft-max over grid points. This gives a smooth objective.
        """
        # Compute angular distances: (n_grid, N)
        dot = np.dot(grid_pts, pts.T)  # (n_grid, N)
        dot = np.clip(dot, -1.0, 1.0)
        # Angular distance in radians
        ang_dists = np.arccos(dot)  # (n_grid, N)

        # Soft-min over points (axis=1): for each grid point, find "nearest" point smoothly
        # soft_min = -soft_max(-x)
        neg_dists = -ang_dists  # (n_grid, N)
        max_neg = np.max(neg_dists, axis=1, keepdims=True)  # (n_grid, 1)
        exp_neg = np.exp(beta * (neg_dists - max_neg))  # (n_grid, N)
        sum_exp_neg = np.sum(exp_neg, axis=1, keepdims=True)  # (n_grid, 1)
        soft_min_vals = -(max_neg[:, 0] + np.log(sum_exp_neg[:, 0]) / beta)  # (n_grid,)

        # Soft-max over grid points
        max_sm = np.max(soft_min_vals)
        exp_sm = np.exp(beta * (soft_min_vals - max_sm))
        sum_exp_sm = np.sum(exp_sm)
        soft_max_val = max_sm + np.log(sum_exp_sm) / beta

        return soft_max_val, exp_sm / sum_exp_sm

    def soft_min_covering_grad(pts, beta=30.0):
        """
        Compute soft-min covering radius and its gradient w.r.t. pts.
        Returns (objective, gradient) where gradient shape is (N, 3).
        """
        dot = np.dot(grid_pts, pts.T)  # (n_grid, N)
        dot_clipped = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot_clipped)  # (n_grid, N)

        # Soft-min over points
        neg_dists = -ang_dists
        max_neg = np.max(neg_dists, axis=1, keepdims=True)
        exp_neg = np.exp(beta * (neg_dists - max_neg))
        sum_exp_neg = np.sum(exp_neg, axis=1, keepdims=True)
        # Weights for soft-min: w_j = exp(beta * (-d_j - max_neg)) / sum
        w_min = exp_neg / sum_exp_neg  # (n_grid, N)
        soft_min_vals = -(max_neg[:, 0] + np.log(sum_exp_neg[:, 0]) / beta)

        # Soft-max over grid points
        max_sm = np.max(soft_min_vals)
        exp_sm = np.exp(beta * (soft_min_vals - max_sm))
        sum_exp_sm = np.sum(exp_sm)
        w_grid = exp_sm / sum_exp_sm  # (n_grid,)
        soft_max_val = max_sm + np.log(sum_exp_sm) / beta

        # Gradient computation:
        # d(soft_min)/d(pts_i) = sum_j w_min[j,i] * d(-ang_dist_ji)/d(pts_i)
        # d(soft_max)/d(soft_min_j) = w_grid[j]
        # Chain rule: d(obj)/d(pts_i) = sum_j w_grid[j] * sum_k w_min[j,k] * d(-ang_dist_jk)/d(pts_i)
        # Only k=i contributes to d(pts_i)
        # d(ang_dist_ji)/d(pts_i): ang_dist = arccos(dot_ji), dot_ji = grid_j . pts_i
        # d(arccos(x))/dx = -1/sqrt(1-x^2)
        # d(dot_ji)/d(pts_i) = grid_j
        # So d(ang_dist_ji)/d(pts_i) = -grid_j / sqrt(1 - dot_ji^2)
        # d(-ang_dist_ji)/d(pts_i) = grid_j / sqrt(1 - dot_ji^2)

        # For each grid point j and point i, compute contribution
        # grad_i = sum_j w_grid[j] * w_min[j,i] * grid_j / sqrt(1 - dot_ji^2)

        # Vectorized:
        # sqrt(1 - dot^2)
        sqrt_term = np.sqrt(np.maximum(1.0 - dot_clipped ** 2, 1e-15))  # (n_grid, N)
        # d(-ang)/d(pts_i) = grid_j / sqrt_term[j,i]
        # Contribution to grad_i from grid j: w_grid[j] * w_min[j,i] * grid_j / sqrt_term[j,i]
        # grad_i = sum_j (w_grid[j] * w_min[j,i] / sqrt_term[j,i]) * grid_j

        # weights[j,i] = w_grid[j] * w_min[j,i] / sqrt_term[j,i]
        weights = w_grid[:, np.newaxis] * w_min / sqrt_term  # (n_grid, N)

        # grad_i = sum_j weights[j,i] * grid_j
        # grad = weights.T @ grid_pts  -> (N, n_grid) @ (n_grid, 3) = (N, 3)
        grad = weights.T @ grid_pts  # (N, 3)

        return soft_max_val, grad

    def repulsion_polish(points, alpha, n_iter=30):
        """Simple repulsion-based polish."""
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

    n_restarts = 6

    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break

        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            # Spherical Fibonacci with perturbation
            points = fibonacci_spiral(N).astype(np.float64)
            points = points + 0.1 * np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                points = best_points + 0.15 * np.random.randn(N, 3)
            else:
                points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)

        # Phase 1: Repulsion-based initial optimization
        alpha = 0.05
        decay = 0.998
        max_iter = 300
        stall_count = 0

        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break

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
            alpha *= decay

            if it % 5 == 0:
                rad = compute_covering_radius(points)
                if rad < best_rad:
                    best_rad = rad
                    best_points = points.copy()
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > 20:
                        alpha *= 0.95
                        stall_count = 0

            if alpha < 1e-7:
                break

        # Phase 2: L-BFGS-B optimization with soft-min covering objective
        if time.time() < deadline - 15.0 and best_points is not None:
            # Flatten points for L-BFGS-B
            x0 = best_points.flatten().copy()

            def objective_and_grad(x):
                pts = x.reshape(N, 3)
                # Normalize to unit sphere
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms[norms < 1e-15] = 1.0
                pts = pts / norms
                obj, grad = soft_min_covering_grad(pts, beta=30.0)
                # Project gradient to tangent space: grad_i -= (grad_i . pts_i) pts_i
                dot_g_p = np.sum(grad * pts, axis=1, keepdims=True)
                grad_tangent = grad - dot_g_p * pts
                return obj, grad_tangent.flatten()

            try:
                result = minimize(
                    objective_and_grad,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'maxls': 20, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                opt_pts = result.x.reshape(N, 3)
                opt_pts = opt_pts / np.linalg.norm(opt_pts, axis=1, keepdims=True)
                rad_opt = compute_covering_radius(opt_pts)
                if rad_opt < best_rad:
                    best_rad = rad_opt
                    best_points = opt_pts.copy()
            except Exception:
                pass

            # Phase 3: Second L-BFGS-B pass with higher beta for sharper optimization
            if time.time() < deadline - 10.0 and best_points is not None:
                x0 = best_points.flatten().copy()

                def objective_and_grad2(x):
                    pts = x.reshape(N, 3)
                    norms = np.linalg.norm(pts, axis=1, keepdims=True)
                    norms[norms < 1e-15] = 1.0
                    pts = pts / norms
                    obj, grad = soft_min_covering_grad(pts, beta=50.0)
                    dot_g_p = np.sum(grad * pts, axis=1, keepdims=True)
                    grad_tangent = grad - dot_g_p * pts
                    return obj, grad_tangent.flatten()

                try:
                    result2 = minimize(
                        objective_and_grad2,
                        x0,
                        jac=True,
                        method='L-BFGS-B',
                        options={'maxiter': 40, 'maxls': 20, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    opt_pts2 = result2.x.reshape(N, 3)
                    opt_pts2 = opt_pts2 / np.linalg.norm(opt_pts2, axis=1, keepdims=True)
                    rad_opt2 = compute_covering_radius(opt_pts2)
                    if rad_opt2 < best_rad:
                        best_rad = rad_opt2
                        best_points = opt_pts2.copy()
                except Exception:
                    pass

            # Phase 4: Final repulsion polish
            if time.time() < deadline - 5.0 and best_points is not None:
                polished = repulsion_polish(best_points.copy(), 0.003, 40)
                rad_polished = compute_covering_radius(polished)
                if rad_polished < best_rad:
                    best_rad = rad_polished
                    best_points = polished.copy()

    # Final refinement with L-BFGS-B
    if time.time() < deadline - 5.0 and best_points is not None:
        x0 = best_points.flatten().copy()

        def final_obj_grad(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            pts = pts / norms
            obj, grad = soft_min_covering_grad(pts, beta=80.0)
            dot_g_p = np.sum(grad * pts, axis=1, keepdims=True)
            grad_tangent = grad - dot_g_p * pts
            return obj, grad_tangent.flatten()

        try:
            result_final = minimize(
                final_obj_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 30, 'maxls': 20, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            final_pts = result_final.x.reshape(N, 3)
            final_pts = final_pts / np.linalg.norm(final_pts, axis=1, keepdims=True)
            rad_final = compute_covering_radius(final_pts)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final_pts.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
