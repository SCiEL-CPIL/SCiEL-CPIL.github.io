"""Program for covering the sphere with 130 equal caps using repulsion-based optimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, least_squares

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

    def compute_covering_radius_exact(pts):
        """Compute covering radius exactly as max circumradius over convex hull faces."""
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
            # Fallback: dense grid-based estimate
            lat = np.linspace(-np.pi/2, np.pi/2, 100)
            lon = np.linspace(0, 2*np.pi, 200, endpoint=False)
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

    def repulsion_energy_grad(pts, exponent=2.0):
        """Compute repulsion energy and gradient for uniform distribution."""
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        
        w = 1.0 / (dist_sq ** exponent)
        np.fill_diagonal(w, 0.0)
        
        w_grad = exponent * (dist_sq ** (-exponent/2 - 1))
        np.fill_diagonal(w_grad, 0.0)
        
        grad = w_grad @ pts
        
        return float(np.sum(w)), grad

    def gradient_descent_repulsion(pts, exponent, alpha_init, decay, max_iter):
        """Optimize points by gradient descent on repulsion energy."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            energy, grad = repulsion_energy_grad(pts, exponent)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            pts = pts + alpha * grad
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            alpha *= decay
        
        return pts

    def facet_residuals(pts, target_radius):
        """
        Compute residuals for least_squares: for each convex hull facet,
        residual = circumradius - target_radius.
        """
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
            
            return radii - target_radius
        except Exception:
            return np.zeros(N)

    def soft_voronoi_objective_and_grad(x_flat, grid_pts, temperature=30.0):
        """
        Differentiable 'soft-Voronoi' objective using log-sum-exp approximation.
        Approximates the covering radius as a smooth function of point positions.
        
        The covering radius R = max_y min_i d(y, p_i).
        We approximate min_i d(y, p_i) with a soft-min over i.
        Then we approximate max_y with a soft-max over a fixed grid y.
        
        This yields a smooth objective that can be optimized with L-BFGS-B.
        """
        pts = x_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        pts_norm = pts / norms

        M = len(grid_pts)

        # Compute angular distances: M x N
        dot = np.dot(grid_pts, pts_norm.T)  # M x N
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)  # M x N

        # Level 1: Soft-min over points k for each grid point j
        # s_j = -1/T_min * log( sum_k exp(-T_min * d_jk) )
        T_min = temperature
        # For numerical stability, subtract the max of the exponent terms
        # exp(-T_min * d_jk) = exp(-T_min * (d_jk - min_k d_jk)) * exp(-T_min * min_k d_jk)
        # Actually, standard log-sum-exp trick:
        # log(sum exp(x_k)) = max(x) + log(sum exp(x_k - max(x)))
        # Here x_k = -T_min * d_jk. So max(x_k) = -T_min * min(d_jk).
        # Let d_min_j = min_k d_jk.
        # sum_k exp(-T_min * d_jk) = exp(-T_min * d_min_j) * sum_k exp(-T_min * (d_jk - d_min_j))
        # log(sum ...) = -T_min * d_min_j + log(sum_k exp(-T_min * (d_jk - d_min_j)))
        # s_j = -1/T_min * log(sum ...) = d_min_j - (1/T_min) * log(sum_k exp(-T_min * (d_jk - d_min_j)))
        
        # This is stable if we compute the inner sum correctly.
        # d_jk - d_min_j >= 0.
        # exp(-T_min * (d_jk - d_min_j)) <= 1.
        
        # Compute d_min for each row
        d_min = np.min(ang_dists, axis=1)  # M
        
        # Compute shifted distances
        shifted_dists = ang_dists - d_min[:, np.newaxis]  # M x N
        exp_terms = np.exp(-T_min * shifted_dists)  # M x N
        sum_exp_terms = np.sum(exp_terms, axis=1)  # M
        
        # s_j = d_min_j - (1/T_min) * log(sum_exp_terms_j)
        soft_min_dists = d_min - (1.0 / T_min) * np.log(sum_exp_terms)  # M
        
        # Level 2: Soft-max over grid points j
        # R = max_j s_j
        # Approximate with soft-max: R_soft = max(s) + (1/T_max) * log(sum_j exp(T_max * (s_j - max(s))))
        # To keep it simple and stable, let's just use the max directly if we want exactness,
        # but for gradient, we need a smooth approximation.
        # However, the gradient of the max is just the gradient of the argmax element.
        # If we use soft-max, the gradient is a weighted average.
        # Let's use a moderate T_max for the outer max to keep it smooth but accurate.
        T_max = temperature
        
        s_max = np.max(soft_min_dists)
        shifted_s = soft_min_dists - s_max
        exp_s = np.exp(T_max * shifted_s)
        sum_exp_s = np.sum(exp_s)
        
        R_soft = s_max + (1.0 / T_max) * np.log(sum_exp_s)
        
        # Gradient calculation
        # d(R_soft)/d(pts_norm[k])
        # R_soft = s_max + (1/T_max) log(sum_j exp(T_max (s_j - s_max)))
        # Note: s_max is a constant for differentiation w.r.t x? No, s_max = max(s_j).
        # Actually, the soft-max formula is:
        # R_soft = log(sum_j exp(T_max * s_j)) / T_max
        # This is equivalent to the shifted version above.
        # Gradient:
        # d(R_soft)/d(s_j) = exp(T_max * s_j) / sum_k exp(T_max * s_k) = w_j
        # So d(R_soft)/d(x) = sum_j w_j * d(s_j)/d(x)
        
        # Weights for soft-max
        weights_max = exp_s / sum_exp_s  # M
        
        # Now we need d(s_j)/d(pts_norm[k])
        # s_j = d_min_j - (1/T_min) log(sum_l exp(-T_min (d_jl - d_min_j)))
        # d(s_j)/d(x) = d(d_min_j)/d(x) - (1/T_min) * d(log(...))/d(x)
        # This is tricky because d_min_j depends on x.
        # Alternative formulation:
        # s_j = -1/T_min log(sum_l exp(-T_min d_jl))
        # This is the standard soft-min.
        # Let's recompute using this standard form for easier gradient.
        
        # Recompute soft_min_dists using standard log-sum-exp
        # x_l = -T_min * d_jl
        # max_x = -T_min * d_min_j
        # log_sum_exp = max_x + log(sum_l exp(x_l - max_x))
        # s_j = -1/T_min * log_sum_exp = d_min_j - (1/T_min) log(sum_l exp(-T_min(d_jl - d_min_j)))
        # This matches what we have.
        
        # Gradient of s_j w.r.t pts_norm[k]:
        # s_j = -1/T_min * log( sum_l exp(-T_min * d_jl) )
        # d(s_j)/d(pts_norm[k]) = -1/T_min * (1 / sum_l exp(-T_min d_jl)) * sum_l [ exp(-T_min d_jl) * (-T_min) * d(d_jl)/d(pts_norm[k]) ]
        # = (1 / sum_l exp(-T_min d_jl)) * sum_l [ exp(-T_min d_jl) * d(d_jl)/d(pts_norm[k]) ]
        # = sum_l w_jl * d(d_jl)/d(pts_norm[k])
        # where w_jl = exp(-T_min d_jl) / sum_l' exp(-T_min d_jl')
        
        # We already computed exp_terms = exp(-T_min * (d_jl - d_min_j))
        # sum_exp_terms = sum_l exp_terms
        # w_jl = exp_terms / sum_exp_terms  (since the d_min factor cancels in the ratio)
        
        w_jl = exp_terms / sum_exp_terms[:, np.newaxis]  # M x N
        
        # Now compute d(d_jl)/d(pts_norm[k])
        # d_jl = arccos(dot(grid_j, pts_l))
        # d(d_jl)/d(pts_l) = - grid_j / sqrt(1 - dot^2)
        # Note: this gradient is only non-zero for k=l.
        
        # grad_pts_norm[k] = sum_j weights_max[j] * sum_l w_jl[j, l] * d(d_jl)/d(pts_l)
        # = sum_j weights_max[j] * w_jl[j, k] * (- grid_j / sqrt(1 - dot_jk^2))
        
        # Let's compute this efficiently.
        # For each j, we have a weight W_j = weights_max[j]
        # For each l, we have a weight V_jl = w_jl[j, l]
        # The contribution to grad_pts_norm[k] is:
        # sum_j W_j * V_jk * (- grid_j / sqrt(1 - dot_jk^2))
        
        # Compute denom = sqrt(1 - dot^2)
        dot_sq = dot * dot
        denom = np.sqrt(np.maximum(1.0 - dot_sq, 1e-10))  # M x N
        
        # Gradient vector for each (j, l) pair: g_jl = - grid_j / denom_jl
        # We need to accumulate: grad_pts_norm[l] += sum_j (weights_max[j] * w_jl[j, l] * g_jl)
        
        # Let's reshape for matrix multiplication.
        # grad_pts_norm[l] = sum_j (weights_max[j] * w_jl[j, l] * (-grid_j / denom_jl))
        
        # Combine weights: C_jl = weights_max[j] * w_jl[j, l] / denom_jl
        # Then grad_pts_norm[l] = - sum_j C_jl * grid_j
        
        C = weights_max[:, np.newaxis] * w_jl / denom  # M x N
        
        # grad_pts_norm[l] = - sum_j C_jl * grid_j
        # This is - (C.T @ grid_pts)
        
        grad_pts_norm = -(C.T @ grid_pts)  # N x 3
        
        # Now apply chain rule for normalization
        # pts_norm = pts / ||pts||
        # d(pts_norm[i])/d(pts[i]) = (I - pts_norm[i] pts_norm[i]^T) / ||pts[i]||
        
        grad = np.zeros_like(x_flat)
        for i in range(N):
            p = pts_norm[i]
            n = norms[i, 0]
            grad_i = (grad_pts_norm[i] - np.dot(grad_pts_norm[i], p) * p) / n
            grad[i*3:(i+1)*3] = grad_i
        
        return float(R_soft), grad

    def optimize_with_lbfgsb(points, grid_pts, temperature=30.0, maxiter=200, ftol=1e-8):
        """Optimize using L-BFGS-B on the flattened coordinates."""
        
        def objective(x):
            val, _ = soft_voronoi_objective_and_grad(x, grid_pts, temperature)
            return val
        
        def gradient(x):
            _, g = soft_voronoi_objective_and_grad(x, grid_pts, temperature)
            return g
        
        # Normalize initial points
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        x0 = (points / norms).flatten()
        
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 2, 'ftol': ftol, 'gtol': 1e-6}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    def optimize_with_least_squares(points, target_radius, maxiter=100):
        """
        Optimize using scipy.optimize.least_squares on facet circumradius residuals.
        """
        def residuals(x_flat):
            pts = x_flat.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1e-10
            pts_norm = pts / norms
            return facet_residuals(pts_norm, target_radius)
        
        x0 = points.flatten()
        
        try:
            result = least_squares(
                residuals, x0,
                method='lm',
                max_nfev=maxiter * 10
            )
            x_opt = result.x.reshape(N, 3)
            x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
            return x_opt, result.cost
        except Exception:
            return points.copy(), np.inf

    def make_fib_grid(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Use moderate grid sizes to balance accuracy and speed
    # The soft-Voronoi objective is smooth, so we don't need an extremely dense grid
    # for the optimizer to work well, but a reasonably dense grid is needed for the
    # approximation to be accurate.
    grid_pts_coarse = make_fib_grid(1000)
    grid_pts_fine = make_fib_grid(5000)
    grid_pts_very_fine = make_fib_grid(15000)

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion + L-BFGS-B optimization
    n_restarts = 6
    
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
            points = points + 0.5 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            rng = np.random.RandomState(200 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 4:
            # Start from a slightly perturbed Fibonacci spiral
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(400 + restart)
            points = points + 0.1 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(500 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # First, optimize with repulsion to get a good starting configuration
        points = gradient_descent_repulsion(points, 2.0, 0.1, 0.99, 100)
        
        # Then optimize with L-BFGS-B on the soft-Voronoi objective (coarse grid)
        time_left = deadline - time.time() - 10.0
        if time_left > 10.0:
            points, _ = optimize_with_lbfgsb(points, grid_pts=grid_pts_coarse, temperature=30.0, maxiter=150)
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Local refinement with smaller L-BFGS-B runs using fine grid
    if best_points is not None and time.time() < deadline - 20.0:
        for trial in range(4):
            if time.time() > deadline - 10.0:
                break
            rng = np.random.RandomState(600 + trial)
            candidate = best_points + 0.05 * rng.randn(N, 3)
            candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
            
            time_left = deadline - time.time() - 8.0
            if time_left > 5.0:
                candidate, _ = optimize_with_lbfgsb(candidate, grid_pts=grid_pts_fine, temperature=50.0, maxiter=100)
            
            rad = compute_covering_radius_exact(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()
        
        # Phase 2b: Additional refinement using least_squares on facet residuals
        if time.time() < deadline - 15.0:
            target_rad = best_rad * 0.95
            candidate = best_points.copy()
            candidate, cost = optimize_with_least_squares(candidate, target_rad, maxiter=50)
            rad = compute_covering_radius_exact(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 3: Coordinate-wise local search
    if best_points is not None and time.time() < deadline - 10.0:
        points = best_points.copy()
        improved = True
        max_rounds = 3
        round_count = 0
        
        while improved and round_count < max_rounds and time.time() < deadline - 5.0:
            improved = False
            round_count += 1
            
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                
                old_p = points[i].copy()
                
                # Try perturbing point i in random directions
                rng = np.random.RandomState(i * 1000 + round_count)
                for trial in range(3):
                    direction = rng.randn(3)
                    direction -= np.dot(direction, old_p) * old_p
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm < 1e-10:
                        continue
                    direction /= dir_norm
                    
                    for step in [0.02, 0.01, 0.005]:
                        new_p = old_p + step * direction
                        new_p /= np.linalg.norm(new_p)
                        points[i] = new_p
                        
                        rad = compute_covering_radius_exact(points)
                        if rad < best_rad - 1e-10:
                            best_rad = rad
                            best_points = points.copy()
                            improved = True
                            break
                        else:
                            points[i] = old_p
                    else:
                        continue
                    break
                
                if not improved or rad >= best_rad - 1e-10:
                    points[i] = old_p

    # Phase 4: Final polish with L-BFGS-B using very fine grid and lower temperature
    if best_points is not None and time.time() < deadline - 5.0:
        final = best_points.copy()
        time_left = deadline - time.time() - 3.0
        if time_left > 3.0:
            final, _ = optimize_with_lbfgsb(final, grid_pts=grid_pts_very_fine, temperature=100.0, maxiter=50, ftol=1e-10)
            rad_final = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final
        
        # Phase 4b: Final polish using least_squares on facet residuals
        if best_points is not None and time.time() < deadline - 3.0:
            target_rad = best_rad * 0.98
            final2 = best_points.copy()
            final2, cost2 = optimize_with_least_squares(final2, target_rad, maxiter=30)
            rad_final2 = compute_covering_radius_exact(final2)
            if rad_final2 < best_rad:
                best_rad = rad_final2
                best_points = final2

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
