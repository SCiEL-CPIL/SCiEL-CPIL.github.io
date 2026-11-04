"""
Circle packing in unit square: pack 26 circles, maximize sum of radii.
Combines penalty-based direct optimization with smooth soft-min polish.
Uses multi-start with SA, best-fit, subgradient, and gradient refinement.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 140.0  # Safety margin from 150s limit


def construct():
    start_time = time.time()

    def _elapsed():
        return time.time() - start_time

    def _remaining():
        return max(0.0, DEADLINE - _elapsed())

    def compute_radii(x, y):
        """Compute radii from centers: min of half-distance to neighbors and boundary."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-6)
        return r

    def penalty_objective_and_grad(params, mu=500.0):
        """Penalty-based objective for direct (x,y,r) optimization.
        params: (3N,) = [x0,y0,r0, x1,y1,r1, ...]
        Returns (energy, grad) where energy = -sum(r) + mu * penalty
        """
        n = N
        pts = params.reshape(n, 3)
        xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]

        # Objective: maximize sum(r) -> minimize -sum(r)
        obj = -np.sum(rs)

        # Boundary violations
        b_left = np.maximum(0, rs - xs)
        b_bot = np.maximum(0, rs - ys)
        b_right = np.maximum(0, rs - (1.0 - xs))
        b_top = np.maximum(0, rs - (1.0 - ys))
        bv = b_left + b_bot + b_right + b_top
        bpen = np.sum(bv ** 2)

        # Overlap violations (vectorized)
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist_sq = dx ** 2 + dy ** 2
        dist = np.sqrt(dist_sq)
        rs_sum = rs[i_idx] + rs[j_idx]
        ov = np.maximum(0, rs_sum - dist)
        open_ = np.sum(ov ** 2)

        total_penalty = bpen + open_
        energy = obj + mu * total_penalty

        # Gradient (vectorized)
        grad = np.zeros_like(params)

        # d(-sum r)/dr_i = -1
        grad[2::3] = -1.0

        # Boundary penalty gradient
        db_dxi = np.where(b_left > 0, -1.0, 0.0) + np.where(b_right > 0, 1.0, 0.0)
        db_dyi = np.where(b_bot > 0, -1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0)
        db_dri = (np.where(b_left > 0, 1.0, 0.0) + np.where(b_bot > 0, 1.0, 0.0) +
                  np.where(b_right > 0, 1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0))

        grad[0::3] += mu * 2 * bv * db_dxi
        grad[1::3] += mu * 2 * bv * db_dyi
        grad[2::3] += mu * 2 * bv * db_dri

        # Overlap penalty gradient (vectorized)
        mask = ov > 0
        if np.any(mask):
            i_m = i_idx[mask]
            j_m = j_idx[mask]
            ov_m = ov[mask]
            dist_m = dist[mask]
            safe_dist = np.where(dist_m < 1e-12, 1.0, dist_m)
            ux = dx[mask] / safe_dist
            uy = dy[mask] / safe_dist

            np.add.at(grad, 3 * i_m, -mu * 2 * ov_m * ux)
            np.add.at(grad, 3 * i_m + 1, -mu * 2 * ov_m * uy)
            np.add.at(grad, 3 * i_m + 2, mu * 2 * ov_m)
            np.add.at(grad, 3 * j_m, mu * 2 * ov_m * ux)
            np.add.at(grad, 3 * j_m + 1, mu * 2 * ov_m * uy)
            np.add.at(grad, 3 * j_m + 2, mu * 2 * ov_m)

        return energy, grad

    def smooth_objective_and_grad(params, beta=50.0):
        """Smooth soft-min surrogate for the nonsmooth min objective.
        params: (2N,) = [x0, y0, ..., x_{N-1}, y_{N-1}]
        Radii are derived from centers via soft-min.
        """
        x = params[:N]
        y = params[N:2*N]

        # Boundary constraint
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, 1e10)

        # Build constraint matrix
        c = 0.5 * dist
        np.fill_diagonal(c, r_bound)

        # Soft-min computation
        neg_beta_c = -beta * c
        max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_neg)
        sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
        log_sum = np.log(sum_exp)
        r = -(max_neg + log_sum) / beta
        r = r[:, 0]

        # Weights
        w = exp_terms / sum_exp

        # Gradient of c_ij w.r.t. parameters
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)

        dc_dx_i = 0.5 * dx / dist_safe
        dc_dy_i = 0.5 * dy / dist_safe
        dc_dx_j = -0.5 * dx / dist_safe
        dc_dy_j = -0.5 * dy / dist_safe

        # Diagonal gradients for r_bound_i
        bounds_4 = np.stack([x, 1.0-x, y, 1.0-y], axis=1)
        beta_b = 100.0
        neg_bb = -beta_b * bounds_4
        max_bb = np.max(neg_bb, axis=1, keepdims=True)
        exp_bb = np.exp(neg_bb - max_bb)
        sum_bb = np.sum(exp_bb, axis=1, keepdims=True)
        w_b = exp_bb / sum_bb

        grad_rb_dx = w_b[:, 0] - w_b[:, 1]
        grad_rb_dy = w_b[:, 2] - w_b[:, 3]

        np.fill_diagonal(dc_dx_i, grad_rb_dx)
        np.fill_diagonal(dc_dy_i, grad_rb_dy)
        np.fill_diagonal(dc_dx_j, grad_rb_dx)
        np.fill_diagonal(dc_dy_j, grad_rb_dy)

        dr_dx_self = np.sum(w * dc_dx_i, axis=1)
        dr_dy_self = np.sum(w * dc_dy_i, axis=1)
        dr_dx_cross = np.sum(w * dc_dx_j, axis=0)
        dr_dy_cross = np.sum(w * dc_dy_j, axis=0)

        dr_dx = dr_dx_self + dr_dx_cross - np.diag(w) * np.diag(dc_dx_i)
        dr_dy = dr_dy_self + dr_dy_cross - np.diag(w) * np.diag(dc_dy_i)

        grad = np.concatenate([-dr_dx, -dr_dy])

        obj_val = -np.sum(r)
        return obj_val, grad

    def check_validity(x, y, r):
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        if np.any(r < -1e-9):
            return False

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)

        if np.any(dist < min_dist - 1e-9):
            return False
        return True

    def is_valid_params(params):
        """Check if (x,y,r) parameters are valid."""
        pts = params.reshape(N, 3)
        xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
        if np.any(xs - rs < -1e-9) or np.any(xs + rs > 1 + 1e-9):
            return False
        if np.any(ys - rs < -1e-9) or np.any(ys + rs > 1 + 1e-9):
            return False
        if np.any(rs < -1e-9):
            return False

        i_idx, j_idx = np.triu_indices(N, k=1)
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist = np.sqrt(dx**2 + dy**2)
        rs_sum = rs[i_idx] + rs[j_idx]
        if np.any(rs_sum - dist > 1e-9):
            return False
        return True

    def project_feasible(params, max_iter=200):
        """Project parameters to feasibility."""
        pts = params.reshape(N, 3).copy()
        i_idx, j_idx = np.triu_indices(N, k=1)

        for _ in range(max_iter):
            if _remaining() < 2.0:
                break

            xs, ys, rs = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()

            # Fix boundary violations
            d_left = xs
            d_bot = ys
            d_right = 1.0 - xs
            d_top = 1.0 - ys
            min_d = np.minimum(np.minimum(d_left, d_bot), np.minimum(d_right, d_top))
            too_big = rs > min_d
            if np.any(too_big):
                rs[too_big] = min_d[too_big] * (1 - 1e-9)
                pts[:, 2] = rs

            # Fix overlap violations
            dx = xs[i_idx] - xs[j_idx]
            dy = ys[i_idx] - ys[j_idx]
            dist = np.sqrt(dx**2 + dy**2)
            rs_sum = rs[i_idx] + rs[j_idx]
            ov = np.maximum(0, rs_sum - dist)

            if not np.any(ov > 1e-10):
                break

            overlap_per_circle = np.zeros(N)
            np.add.at(overlap_per_circle, i_idx, ov)
            np.add.at(overlap_per_circle, j_idx, ov)

            reduction = overlap_per_circle / 2.0 + 1e-10
            rs = np.maximum(0.01, rs - reduction)
            pts[:, 2] = rs

        return pts.ravel()

    def grow_radii(params, max_iters=300, step=0.001):
        """Try to increase radii while maintaining feasibility."""
        pts = params.reshape(N, 3).copy()
        i_idx, j_idx = np.triu_indices(N, k=1)

        for _ in range(max_iters):
            if _remaining() < 2.0:
                break

            improved = False
            xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]

            for i in range(N):
                d_left = xs[i]
                d_bot = ys[i]
                d_right = 1.0 - xs[i]
                d_top = 1.0 - ys[i]
                max_r_boundary = min(d_left, d_bot, d_right, d_top)

                max_r_overlap = max_r_boundary
                for j in range(N):
                    if i == j:
                        continue
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx**2 + dy**2)
                    limit = dist - rs[j]
                    if limit < max_r_overlap:
                        max_r_overlap = limit

                new_r = min(max_r_overlap, rs[i] + step)
                if new_r > rs[i] + 1e-12:
                    old_r = rs[i]
                    rs[i] = new_r
                    pts[i, 2] = new_r

                    if is_valid_params(pts.ravel()):
                        improved = True
                    else:
                        rs[i] = old_r
                        pts[i, 2] = old_r

            if not improved:
                break

        return pts.ravel()

    def make_initial_guess(seed, mode='grid'):
        rng = np.random.RandomState(seed)
        if mode == 'grid':
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.08, 0.08, N)
            y += rng.uniform(-0.08, 0.08, N)
        elif mode == 'hex':
            # Hexagonal packing with row counts
            row_counts = [5, 5, 5, 5, 4, 2]
            r_est = 0.09
            y_spacing = r_est * np.sqrt(3)
            x_spacing = 2 * r_est

            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            for i, count in enumerate(row_counts):
                if idx >= N:
                    break
                y_base = (i + 0.5) * y_spacing
                for j in range(count):
                    if idx >= N:
                        break
                    x_offset = (j + 0.5) * x_spacing
                    if i % 2 == 1:
                        x_offset += x_spacing / 2
                    x[idx] = 0.5 + (x_offset - (count * x_spacing) / 2) * 0.8
                    y[idx] = y_base
                    idx += 1

            x += rng.uniform(-0.03, 0.03, N)
            y += rng.uniform(-0.03, 0.03, N)
        else:  # random
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def voronoi_greedy_init(seed):
        """Greedy initialization using largest empty circle."""
        rng = np.random.RandomState(seed)

        n_grid = 12
        grid_x, grid_y = np.meshgrid(np.linspace(0.1, 0.9, n_grid), np.linspace(0.1, 0.9, n_grid))
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        boundary_x = [0.5, 0.5, 0.1, 0.9, 0.1, 0.9, 0.5, 0.5]
        boundary_y = [0.1, 0.9, 0.1, 0.9, 0.5, 0.5, 0.1, 0.9]

        all_x = np.concatenate([grid_x, boundary_x])
        all_y = np.concatenate([grid_y, boundary_y])

        selected_x = []
        selected_y = []
        remaining_x = all_x.copy()
        remaining_y = all_y.copy()

        for _ in range(N):
            if len(remaining_x) == 0:
                break

            if len(selected_x) == 0:
                idx = np.argmin((remaining_x - 0.5)**2 + (remaining_y - 0.5)**2)
            else:
                min_dists = np.min(
                    np.sqrt((remaining_x[:, None] - np.array(selected_x)[None, :])**2 +
                           (remaining_y[:, None] - np.array(selected_y)[None, :])**2),
                    axis=1
                )
                idx = np.argmax(min_dists)

            selected_x.append(remaining_x[idx])
            selected_y.append(remaining_y[idx])
            remaining_x = np.delete(remaining_x, idx)
            remaining_y = np.delete(remaining_y, idx)

        for i in range(N):
            selected_x[i] += rng.uniform(-0.02, 0.02)
            selected_y[i] += rng.uniform(-0.02, 0.02)

        x = np.array(selected_x)
        y = np.array(selected_y)
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def sa_optimize(x0, y0, seed, max_iters=3000):
        """Simulated annealing to optimize center positions."""
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        r = compute_radii(x, y)
        best_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()

        T_start = 0.05
        T_end = 0.0005
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        repulse_k = 2.0
        boundary_k = 5.0

        for it in range(max_iters):
            if _remaining() < 5.0:
                break

            T = T_start * (cool_rate ** it)

            r_curr = compute_radii(x, y)
            curr_val = np.sum(r_curr)

            fx = np.zeros(N)
            fy = np.zeros(N)

            fx += boundary_k * np.clip(0.1 - x, 0, None)
            fx -= boundary_k * np.clip(x - 0.9, 0, None)
            fy += boundary_k * np.clip(0.1 - y, 0, None)
            fy -= boundary_k * np.clip(y - 0.9, 0, None)

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1e10)

            f_mag = repulse_k / (d**2 + 0.01)
            np.fill_diagonal(f_mag, 0)

            fx += np.sum(-f_mag * dx / d, axis=1)
            fy += np.sum(-f_mag * dy / d, axis=1)

            scale = 0.02 * T / T_start + 0.005
            dx_prop = fx * scale + rng.normal(0, scale, N)
            dy_prop = fy * scale + rng.normal(0, scale, N)

            x_new = x + dx_prop
            y_new = y + dy_prop

            x_new = np.clip(x_new, 0.05, 0.95)
            y_new = np.clip(y_new, 0.05, 0.95)

            r_new = compute_radii(x_new, y_new)
            new_val = np.sum(r_new)

            delta = new_val - curr_val
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                x = x_new
                y = y_new
                r = r_new
                if new_val > best_val:
                    best_val = new_val
                    best_x = x.copy()
                    best_y = y.copy()

        return best_x, best_y, compute_radii(best_x, best_y), best_val

    def best_fit_local_search(x, y, step_sizes=None, max_rounds=60):
        """Discrete best-fit local search."""
        if step_sizes is None:
            step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]

        dirs = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)],
            [-1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), -1/np.sqrt(2)]
        ])

        x = x.copy()
        y = y.copy()
        r = compute_radii(x, y)
        curr_val = np.sum(r)

        for rnd in range(max_rounds):
            if _remaining() < 5.0:
                break

            best_improvement = 1e-10
            best_i = -1
            best_dir = None
            best_step = None

            for step in step_sizes:
                x_new = x[:, None] + dirs[None, :, 0] * step
                y_new = y[:, None] + dirs[None, :, 1] * step

                valid = (x_new >= 0.01) & (x_new <= 0.99) & (y_new >= 0.01) & (y_new <= 0.99)

                for i in range(N):
                    for d in range(8):
                        if not valid[i, d]:
                            continue

                        x_temp = x.copy()
                        y_temp = y.copy()
                        x_temp[i] = x_new[i, d]
                        y_temp[i] = y_new[i, d]

                        r_new = compute_radii(x_temp, y_temp)
                        new_val = np.sum(r_new)
                        improvement = new_val - curr_val

                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_i = i
                            best_dir = d
                            best_step = step

                if best_improvement > step * 0.1:
                    break

            if best_i >= 0:
                x[best_i] += dirs[best_dir, 0] * best_step
                y[best_i] += dirs[best_dir, 1] * best_step
                r = compute_radii(x, y)
                curr_val = np.sum(r)
            else:
                break

        return x, y, compute_radii(x, y), np.sum(compute_radii(x, y))

    def penalty_polish(params0, mu_start=100.0, mu_end=2000.0, num_steps=5):
        """Polish using penalty-based optimization with increasing mu."""
        mus = np.geomspace(mu_start, mu_end, num_steps)

        current_params = params0.copy()
        best_result = None
        best_val = -np.inf

        def update_best(params):
            nonlocal best_result, best_val
            if is_valid_params(params):
                val = np.sum(params.reshape(N, 3)[:, 2])
                if val > best_val:
                    best_val = val
                    best_result = params.copy()
            else:
                # Try projecting
                proj = project_feasible(params)
                if is_valid_params(proj):
                    val = np.sum(proj.reshape(N, 3)[:, 2])
                    if val > best_val:
                        best_val = val
                        best_result = proj.copy()

        update_best(current_params)

        for mu in mus:
            if _remaining() < 5.0:
                break

            try:
                bounds = [(0.01, 0.99, 0.001, 0.5)] * N  # (x_min, x_max, r_min, r_max) per circle
                # Flatten bounds for scipy
                flat_bounds = []
                for _ in range(N):
                    flat_bounds.append((0.01, 0.99))
                    flat_bounds.append((0.01, 0.99))
                    flat_bounds.append((0.001, 0.5))

                res = minimize(penalty_objective_and_grad, current_params,
                              method='L-BFGS-B', bounds=flat_bounds,
                              args=(mu,),
                              jac=True,
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40})
                current_params = res.x
                update_best(current_params)
            except Exception:
                pass

        if best_result is not None:
            return best_result
        return None

    def smooth_polish(x0, y0, beta_start=50.0, beta_end=500.0, num_steps=5):
        """Polish using smooth soft-min surrogate."""
        params0 = np.concatenate([x0, y0])
        bounds = [(0.01, 0.99)] * (2 * N)

        betas = np.geomspace(beta_start, beta_end, num_steps)

        current_params = params0.copy()
        best_result = None
        best_val = -np.inf

        def update_best(params):
            nonlocal best_result, best_val
            x = params[:N]
            y = params[N:2*N]
            r = compute_radii(x, y)
            if check_validity(x, y, r):
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_result = (x.copy(), y.copy(), r.copy(), val)

        update_best(current_params)

        for beta in betas:
            if _remaining() < 5.0:
                break

            try:
                res = minimize(smooth_objective_and_grad, current_params,
                              method='L-BFGS-B', bounds=bounds,
                              args=(beta,),
                              jac=True,
                              options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 50})
                current_params = res.x
                update_best(current_params)
            except Exception:
                pass

        if best_result is not None:
            return best_result
        return None

    def hard_subgradient_polish(x, y, num_steps=30, step_size=0.005):
        """Refine using exact sub-gradient of hard min function."""
        x = x.copy()
        y = y.copy()

        for _ in range(num_steps):
            if _remaining() < 3.0:
                break

            r_bound = np.minimum(
                np.minimum(x, 1.0 - x),
                np.minimum(y, 1.0 - y)
            )

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)

            half_dist = 0.5 * dist

            grad_x = np.zeros(N)
            grad_y = np.zeros(N)

            for i in range(N):
                min_d = np.min(half_dist[i])
                min_j = np.argmin(half_dist[i])

                if r_bound[i] < min_d:
                    bounds = [x[i], 1.0-x[i], y[i], 1.0-y[i]]
                    min_idx = np.argmin(bounds)
                    if min_idx == 0:
                        grad_x[i] += 1.0
                    elif min_idx == 1:
                        grad_x[i] -= 1.0
                    elif min_idx == 2:
                        grad_y[i] += 1.0
                    elif min_idx == 3:
                        grad_y[i] -= 1.0
                else:
                    d = dist[i, min_j]
                    if d > 1e-10:
                        gx = 0.5 * (x[i] - x[min_j]) / d
                        gy = 0.5 * (y[i] - y[min_j]) / d
                        grad_x[i] += gx
                        grad_y[i] += gy
                        grad_x[min_j] -= gx
                        grad_y[min_j] -= gy

            norm = np.sqrt(np.sum(grad_x**2) + np.sum(grad_y**2))
            if norm < 1e-10:
                break

            alpha = step_size / max(norm, 1.0)

            x_new = x + alpha * grad_x
            y_new = y + alpha * grad_y
            x_new = np.clip(x_new, 0.05, 0.95)
            y_new = np.clip(y_new, 0.05, 0.95)

            r_new = compute_radii(x_new, y_new)
            val_new = np.sum(r_new)
            r_curr = compute_radii(x, y)
            val_curr = np.sum(r_curr)

            if val_new > val_curr:
                x = x_new
                y = y_new
            else:
                alpha *= 0.5
                x_new = x + alpha * grad_x
                y_new = y + alpha * grad_y
                x_new = np.clip(x_new, 0.05, 0.95)
                y_new = np.clip(y_new, 0.05, 0.95)
                r_new = compute_radii(x_new, y_new)
                val_new = np.sum(r_new)
                if val_new > val_curr:
                    x = x_new
                    y = y_new
                else:
                    break

        return x, y, compute_radii(x, y), np.sum(compute_radii(x, y))

    # Initial best: grid configuration
    side = int(np.ceil(np.sqrt(N)))
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            best_x[idx] = (i + 0.5) / side
            best_y[idx] = (j + 0.5) / side
            idx += 1
    best_r = compute_radii(best_x, best_y)
    best_val = np.sum(best_r)

    # Multi-start with SA + best-fit + subgradient + penalty/smooth polish
    modes = ['grid', 'hex', 'random', 'voronoi']
    num_starts = 40
    for i in range(num_starts):
        if _remaining() < 20.0:
            break

        mode = modes[i % len(modes)]
        if mode == 'voronoi':
            x0, y0 = voronoi_greedy_init(i)
        else:
            x0, y0 = make_initial_guess(i, mode)

        # Run SA
        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=3000)

        # Best-fit local search
        x_bf, y_bf, r_bf, val_bf = best_fit_local_search(x_sa, y_sa)

        # Hard subgradient polish
        if _remaining() > 15.0:
            x_hs, y_hs, r_hs, val_hs = hard_subgradient_polish(x_bf, y_bf, num_steps=30, step_size=0.005)
            if val_hs > val_bf:
                x_bf, y_bf, r_bf, val_bf = x_hs, y_hs, r_hs, val_hs

        # Penalty-based polish
        if _remaining() > 10.0:
            params0 = np.column_stack([x_bf, y_bf, r_bf]).ravel()
            polished = penalty_polish(params0, mu_start=100.0, mu_end=2000.0, num_steps=5)
            if polished is not None:
                pts = polished.reshape(N, 3)
                xp, yp, rp = pts[:, 0], pts[:, 1], pts[:, 2]
                if is_valid_params(polished):
                    vp = np.sum(rp)
                    if vp > val_bf:
                        x_sa, y_sa, r_sa, val_sa = xp, yp, rp, vp

        # Smooth polish
        if _remaining() > 8.0:
            polished2 = smooth_polish(x_bf, y_bf, beta_start=50.0, beta_end=500.0, num_steps=5)
            if polished2 is not None:
                xp, yp, rp, vp = polished2
                if vp > val_bf:
                    x_sa, y_sa, r_sa, val_sa = xp, yp, rp, vp

        # Update best
        if val_sa > best_val:
            best_val = val_sa
            best_x = x_sa.copy()
            best_y = y_sa.copy()
            best_r = r_sa.copy()

        # Also try best-fit on initial guess directly
        if i % 4 == 3 and _remaining() > 15.0:
            x_bf2, y_bf2, r_bf2, val_bf2 = best_fit_local_search(x0, y0)
            if _remaining() > 10.0:
                x_hs2, y_hs2, r_hs2, val_hs2 = hard_subgradient_polish(x_bf2, y_bf2, num_steps=30, step_size=0.005)
                if val_hs2 > val_bf2:
                    x_bf2, y_bf2, r_bf2, val_bf2 = x_hs2, y_hs2, r_hs2, val_hs2
                if _remaining() > 8.0:
                    params02 = np.column_stack([x_bf2, y_bf2, r_bf2]).ravel()
                    polished2 = penalty_polish(params02, mu_start=100.0, mu_end=2000.0, num_steps=5)
                    if polished2 is not None:
                        pts2 = polished2.reshape(N, 3)
                        xp2, yp2, rp2 = pts2[:, 0], pts2[:, 1], pts2[:, 2]
                        if is_valid_params(polished2):
                            vp2 = np.sum(rp2)
                            if vp2 > val_bf2:
                                x_bf2, y_bf2, r_bf2, val_bf2 = xp2, yp2, rp2, vp2
            if val_bf2 > best_val:
                best_val = val_bf2
                best_x = x_bf2.copy()
                best_y = y_bf2.copy()
                best_r = r_bf2.copy()

    # Final high-precision polish
    if _remaining() > 5.0:
        x_bf_final, y_bf_final, r_bf_final, val_bf_final = best_fit_local_search(best_x, best_y)
        if val_bf_final > best_val:
            best_val = val_bf_final
            best_x = x_bf_final
            best_y = y_bf_final
            best_r = r_bf_final

        x_hs_final, y_hs_final, r_hs_final, val_hs_final = hard_subgradient_polish(best_x, best_y, num_steps=50, step_size=0.003)
        if val_hs_final > best_val:
            best_val = val_hs_final
            best_x = x_hs_final
            best_y = y_hs_final
            best_r = r_hs_final

        # Grow radii
        if _remaining() > 5.0:
            params0 = np.column_stack([best_x, best_y, best_r]).ravel()
            grown = grow_radii(params0, max_iters=300, step=0.001)
            if is_valid_params(grown):
                pts = grown.reshape(N, 3)
                xp, yp, rp = pts[:, 0], pts[:, 1], pts[:, 2]
                val_gr = np.sum(rp)
                if val_gr > best_val:
                    best_val = val_gr
                    best_x = xp
                    best_y = yp
                    best_r = rp

        # Multiple gradient polish passes
        for polish_pass in range(4):
            if _remaining() < 3.0:
                break
            beta_s = 50.0 * (10 ** polish_pass)
            beta_e = 1000.0 * (10 ** polish_pass)
            final_polish = smooth_polish(best_x, best_y,
                                         beta_start=min(beta_s, 3000.0),
                                         beta_end=min(beta_e, 8000.0),
                                         num_steps=7)
            if final_polish is not None:
                xp, yp, rp, vp = final_polish
                if vp > best_val:
                    best_val = vp
                    best_x = xp
                    best_y = yp
                    best_r = rp

                if _remaining() > 3.0:
                    x_bf2, y_bf2, r_bf2, val_bf2 = best_fit_local_search(best_x, best_y)
                    if val_bf2 > best_val:
                        best_val = val_bf2
                        best_x = x_bf2
                        best_y = y_bf2
                        best_r = r_bf2

                if _remaining() > 3.0:
                    x_hs2, y_hs2, r_hs2, val_hs2 = hard_subgradient_polish(best_x, best_y, num_steps=30, step_size=0.002)
                    if val_hs2 > best_val:
                        best_val = val_hs2
                        best_x = x_hs2
                        best_y = y_hs2
                        best_r = r_hs2

                if _remaining() > 3.0:
                    params0 = np.column_stack([best_x, best_y, best_r]).ravel()
                    grown = grow_radii(params0, max_iters=200, step=0.0005)
                    if is_valid_params(grown):
                        pts = grown.reshape(N, 3)
                        xp, yp, rp = pts[:, 0], pts[:, 1], pts[:, 2]
                        val_gr = np.sum(rp)
                        if val_gr > best_val:
                            best_val = val_gr
                            best_x = xp
                            best_y = yp
                            best_r = rp

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.99, 0.90, 20):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
