"""Program for covering the sphere with 130 equal caps using repulsion-based optimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# Pre-compute a dense grid for the smooth objective
def _make_fib_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

GRID_COARSE = _make_fib_grid(2000)
GRID_FINE = _make_fib_grid(8000)

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
            dot = np.dot(GRID_FINE, pts.T)
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

    def grid_objective_and_grad(x_flat, grid_pts):
        """
        Smooth objective: soft-max of angular distances from grid points to nearest configuration point.
        Uses a vectorized, differentiable approximation.
        x_flat: flattened (N,3) array of points.
        grid_pts: precomputed grid of sphere points.
        Returns (value, gradient).
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

        # For each grid point, find the min distance to any configuration point
        # Use soft-min for differentiability
        temperature = 50.0
        min_dists = np.min(ang_dists, axis=1)  # M
        
        # Soft-max of min_dists to get a smooth approximation of the covering radius
        max_d = np.max(min_dists)
        shifted = min_dists - max_d
        exp_shifted = np.exp(shifted * temperature)
        sum_exp = np.sum(exp_shifted)
        log_sum_exp = np.log(sum_exp) / temperature + max_d
        
        # Gradient computation
        # d/dx [ softmax(max(d_i)) ]
        # Let S = sum_j exp(T * (d_j - max_d))
        # val = max_d + (1/T) log(S)
        # d(val)/d(x) = (1/T) * (1/S) * sum_j [ exp(T * (d_j - max_d)) * T * (d(d_j)/dx) * (1 - delta_{j, argmax}) ... ]
        # Actually, simpler:
        # softmax is a weighted average of the d_j's, with weights w_j = exp(T*(d_j - max_d)) / S
        # d(val)/dx = sum_j w_j * (d(d_j)/dx)
        # Note: d_j is the distance from grid point j to its NEAREST point.
        # This is where it gets tricky. The "soft-min" over points k for a fixed grid point j is:
        # min_k d(j, k) approx -1/T log(sum_k exp(-T * d(j, k)))
        # Let's use that formulation directly as it is more stable and standard.
        
        # Reset calculation with soft-min over points
        # d_j = -1/T log( sum_k exp(-T * d(j, k)) )
        # val = max_j d_j (soft-max)
        
        # This requires two levels of smoothing.
        # Level 1: soft-min over points k for each grid point j.
        # Level 2: soft-max over grid points j.
        
        # Let's stick to the previous approach but fix the gradient.
        # The previous approach used:
        # val = max_d + (1/T) log(sum exp(T*(d_j - max_d)))
        # where d_j = min_k ang_dists[j, k]
        # The gradient of d_j w.r.t x is the gradient of the min, which is 0 for non-minimizers and grad for minimizer.
        # To smooth this, we should use the soft-min gradient.
        
        # Let's implement the full soft-min/soft-max gradient properly.
        
        # 1. Soft-min over points for each grid point
        # s_j = -1/T_min * log( sum_k exp(-T_min * ang_dists[j, k]) )
        T_min = 50.0
        exp_neg_T_min_d = np.exp(-T_min * ang_dists) # M x N
        sum_exp_neg = np.sum(exp_neg_T_min_d, axis=1) # M
        soft_min_dists = -1.0 / T_min * np.log(sum_exp_neg) # M
        
        # Weights for soft-min: w_jk = exp(-T_min * d_jk) / sum_k exp(-T_min * d_jk)
        w_jk = exp_neg_T_min_d / sum_exp_neg[:, np.newaxis] # M x N
        
        # Gradient of soft_min_dists[j] w.r.t pts_norm[k]
        # d(s_j)/d(pts_norm[k]) = -1/T_min * (1/sum_exp_neg) * sum_k' [ exp(-T_min * d_jk') * d(-T_min * d_jk')/d(pts_norm[k']) ]
        # = sum_k' w_jk' * (d(d_jk')/d(pts_norm[k']))
        # d(d_jk)/d(pts_norm[k]) = - (grid_pts[j] - pts_norm[k]) / ||grid_pts[j] - pts_norm[k]||
        # Note: d_jk is arccos(dot). d/dx arccos(u) = -1/sqrt(1-u^2) * du/dx.
        # u = dot(grid_pts[j], pts_norm[k]). du/d(pts_norm[k]) = grid_pts[j].
        # So d(d_jk)/d(pts_norm[k]) = - grid_pts[j] / sqrt(1 - dot^2)
        
        # 2. Soft-max over grid points
        # val = max_d + 1/T_max * log(sum_j exp(T_max * (s_j - max_d)))
        T_max = 50.0
        max_s = np.max(soft_min_dists)
        exp_T_max_s = np.exp(T_max * (soft_min_dists - max_s)) # M
        sum_exp_pos = np.sum(exp_T_max_s)
        val = max_s + 1.0 / T_max * np.log(sum_exp_pos)
        
        # Weights for soft-max: v_j = exp(T_max * (s_j - max_s)) / sum_exp_pos
        v_j = exp_T_max_s / sum_exp_pos # M
        
        # Gradient of val w.r.t pts_norm[k]
        # d(val)/d(pts_norm[k]) = sum_j v_j * (d(s_j)/d(pts_norm[k]))
        # = sum_j v_j * sum_k' w_jk' * (d(d_jk')/d(pts_norm[k']))
        # = sum_k' (sum_j v_j * w_jk') * (d(d_jk')/d(pts_norm[k']))
        
        # Let's compute this vectorized.
        # coeff_k = sum_j v_j * w_jk
        # coeff is N vector.
        coeff_k = v_j @ w_jk # N
        
        # Now, grad_pts_norm[k] = coeff_k * (d(d_jk)/d(pts_norm[k]))
        # But wait, the term is sum_j v_j * w_jk * (-grid_pts[j] / sqrt(1-dot^2))
        # This is not simply coeff_k * something, because the denominator depends on j.
        
        # We need to accumulate:
        # grad_pts_norm[k] = sum_j v_j * w_jk * (-grid_pts[j] / sqrt(1 - dot[j,k]^2))
        
        # Let's compute this.
        # denom = sqrt(1 - dot^2) # M x N
        denom = np.sqrt(np.maximum(1.0 - dot**2, 1e-10)) # M x N
        # factor = v_j[:, np.newaxis] * w_jk / denom # M x N
        # grad_pts_norm[k] = sum_j factor[j,k] * (-grid_pts[j])
        
        factor = (v_j[:, np.newaxis] * w_jk) / denom # M x N
        grad_pts_norm = -(factor.T @ grid_pts) # N x 3
        
        # Chain rule: d(pts_norm)/d(pts) = (I - pts_norm pts_norm^T) / norm
        grad = np.zeros_like(x_flat)
        for i in range(N):
            p = pts_norm[i]
            n = norms[i, 0]
            # d(pts_norm[i])/d(pts[i]) = (I - p p^T)/n
            grad_i = (grad_pts_norm[i] - np.dot(grad_pts_norm[i], p) * p) / n
            grad[i*3:(i+1)*3] = grad_i
        
        return float(val), grad

    def optimize_with_lbfgsb(points, grid_pts, maxiter=200):
        """Optimize using L-BFGS-B on the flattened coordinates."""
        
        def objective(x):
            val, _ = grid_objective_and_grad(x, grid_pts)
            return val
        
        def gradient(x):
            _, g = grid_objective_and_grad(x, grid_pts)
            return g
        
        # Normalize initial points
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        x0 = (points / norms).flatten()
        
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion + L-BFGS-B optimization
    n_restarts = 5
    
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
        else:
            if best_points is not None:
                rng = np.random.RandomState(300 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # First, optimize with repulsion to get a good starting configuration
        points = gradient_descent_repulsion(points, 2.0, 0.1, 0.99, 100)
        
        # Then optimize with L-BFGS-B on the grid-based objective (coarse grid)
        time_left = deadline - time.time() - 10.0
        if time_left > 10.0:
            points, _ = optimize_with_lbfgsb(points, GRID_COARSE, maxiter=200)
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Local refinement with smaller L-BFGS-B runs using fine grid
    if best_points is not None and time.time() < deadline - 20.0:
        for trial in range(3):
            if time.time() > deadline - 10.0:
                break
            rng = np.random.RandomState(500 + trial)
            candidate = best_points + 0.05 * rng.randn(N, 3)
            candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
            
            time_left = deadline - time.time() - 8.0
            if time_left > 5.0:
                candidate, _ = optimize_with_lbfgsb(candidate, GRID_FINE, maxiter=100)
            
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
                    direction -= np.dot(direction, old_p) * old_p  # Project to tangent
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
                
                # If we improved, keep the new position
                # Otherwise, restore
                if not improved or rad >= best_rad - 1e-10:
                    points[i] = old_p

    # Phase 4: Final polish with L-BFGS-B using fine grid
    if best_points is not None and time.time() < deadline - 5.0:
        final = best_points.copy()
        time_left = deadline - time.time() - 3.0
        if time_left > 3.0:
            final, _ = optimize_with_lbfgsb(final, GRID_FINE, maxiter=50)
            rad_final = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
