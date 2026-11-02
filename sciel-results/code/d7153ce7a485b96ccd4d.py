"""Structural change: decouple center optimization (smooth repulsive-force gradient descent)
from radius assignment (LP maximization of sum of radii given fixed centers).
This allows the geometry to explore globally while the radii are set optimally for each geometry."""
N = 21
import numpy as np
from scipy.optimize import linprog, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145

    best_value = -np.inf
    best_result = None
    best_wh = None

    def compute_radii_lp(xs, ys, w, h):
        """Given fixed centers, solve LP to maximize sum of radii subject to:
        r_i >= 0
        r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
        r_i + r_j <= dist(i,j) for all i<j
        """
        n = N
        c = -np.ones(n)  # minimize -sum(r) = maximize sum(r)
        
        # Inequality constraints: A_ub @ r <= b_ub
        # Boundary constraints: r_i <= x_i, r_i <= w-x_i, r_i <= y_i, r_i <= h-y_i
        A_rows = []
        b_rows = []
        
        for i in range(n):
            # r_i <= x_i
            row = np.zeros(n); row[i] = 1.0
            A_rows.append(row); b_rows.append(xs[i])
            # r_i <= w - x_i
            row = np.zeros(n); row[i] = 1.0
            A_rows.append(row); b_rows.append(w - xs[i])
            # r_i <= y_i
            row = np.zeros(n); row[i] = 1.0
            A_rows.append(row); b_rows.append(ys[i])
            # r_i <= h - y_i
            row = np.zeros(n); row[i] = 1.0
            A_rows.append(row); b_rows.append(h - ys[i])
        
        # Pairwise constraints: r_i + r_j <= dist(i,j)
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_rows.append(row)
                b_rows.append(dist)
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        bounds = [(0, None)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success and res.x is not None:
                rs = np.maximum(res.x, 0)
                # Verify feasibility
                for i in range(n):
                    if rs[i] > xs[i] + 1e-9 or rs[i] > w - xs[i] + 1e-9:
                        return None
                    if rs[i] > ys[i] + 1e-9 or rs[i] > h - ys[i] + 1e-9:
                        return None
                    if rs[i] < -1e-9:
                        return None
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist_sq = dx * dx + dy * dy
                        if dist_sq < (rs[i] + rs[j] - 1e-8) ** 2:
                            return None
                return rs
            else:
                return None
        except Exception:
            return None

    def check_and_update(xs, ys, rs, w, h):
        nonlocal best_value, best_result, best_wh
        if time.time() > deadline:
            return
        if rs is None:
            return
        if np.any(rs < 1e-9):
            return
        if np.any(xs - rs < -1e-6) or np.any(xs + rs > w + 1e-6):
            return
        if np.any(ys - rs < -1e-6) or np.any(ys + rs > h + 1e-6):
            return
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                min_dist = rs[i] + rs[j]
                if dist_sq < (min_dist - 1e-7) ** 2:
                    return
        val = np.sum(rs)
        if val > best_value:
            best_value = val
            x = np.zeros(3 * N)
            x[0::3] = xs
            x[1::3] = ys
            x[2::3] = rs
            best_result = x.copy()
            best_wh = (w, h)

    def repulsive_force_objective(pos, w, h, k=3.0):
        """Minimize a repulsive force objective to spread centers apart.
        pos is (2N,) array: [x0, y0, x1, y1, ...]
        We use a smooth approximation: sum over pairs of min(0, dist - r_target)^2
        but since radii are decoupled, we just want centers well-separated.
        Use: -sum over pairs of log(dist(i,j) + eps) to encourage large separations,
        plus boundary penalties to keep centers in [margin, w-margin] x [margin, h-margin].
        """
        xs = pos[0::2]
        ys = pos[1::2]
        
        # Boundary penalty: keep centers at least r_min from walls
        # Use a soft lower bound on distance from walls
        r_min = 0.01
        pen = 0.0
        for i in range(N):
            if xs[i] < r_min:
                pen += (r_min - xs[i]) ** 2
            if xs[i] > w - r_min:
                pen += (xs[i] - (w - r_min)) ** 2
            if ys[i] < r_min:
                pen += (r_min - ys[i]) ** 2
            if ys[i] > h - r_min:
                pen += (ys[i] - (h - r_min)) ** 2
        
        # Repulsive term: minimize -sum log(dist(i,j) + eps)
        # This pushes centers apart
        repulsive = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                repulsive -= np.log(dist + 1e-6)
        
        return repulsive + pen

    def optimize_centers(w, h, pos0, max_iter=500):
        """Optimize center positions using L-BFGS-B on the repulsive objective."""
        bounds = [(0.01, w - 0.01), (0.01, h - 0.01)] * N
        try:
            res = minimize(lambda p: repulsive_force_objective(p, w, h), pos0,
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10})
            return res.x
        except Exception:
            return pos0

    def sa_optimize_centers(w, h, pos0, max_iters=2000, seed=42):
        """Simulated annealing on center positions with repulsive force objective."""
        pos = pos0.copy()
        xs = pos[0::2].copy()
        ys = pos[1::2].copy()
        
        T = 0.02
        cooling = 0.9997
        rng = np.random.default_rng(seed)
        
        current_cost = repulsive_force_objective(pos, w, h)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            idx = rng.integers(0, N)
            old_x, old_y = xs[idx], ys[idx]
            
            scale = T * 0.5
            xs[idx] += rng.normal(0, scale)
            ys[idx] += rng.normal(0, scale)
            xs[idx] = np.clip(xs[idx], 0.01, w - 0.01)
            ys[idx] = np.clip(ys[idx], 0.01, h - 0.01)
            
            pos[2 * idx] = xs[idx]
            pos[2 * idx + 1] = ys[idx]
            
            new_cost = repulsive_force_objective(pos, w, h)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                current_cost = new_cost
            else:
                xs[idx], ys[idx] = old_x, old_y
                pos[2 * idx] = old_x
                pos[2 * idx + 1] = old_y
            
            T *= cooling
        
        return pos

    def make_grid_init(w, h):
        """Create a grid of initial center positions."""
        cols = int(np.ceil(np.sqrt(N * w / h))) if w >= h else int(np.ceil(np.sqrt(N * h / w)))
        rows = int(np.ceil(N / cols))
        
        pos = np.zeros(2 * N)
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= N:
                    break
                x = (col + 0.5) * w / cols
                y = (row + 0.5) * h / rows
                pos[2 * idx] = x
                pos[2 * idx + 1] = y
                idx += 1
        return pos

    # Aspect ratios to try
    widths = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50]

    for w in widths:
        if time.time() > deadline - 5:
            break
        h = 2.0 - w
        
        pos0 = make_grid_init(w, h)
        
        # Phase 1: Gradient-based center optimization
        pos_opt = optimize_centers(w, h, pos0, max_iter=300)
        
        # Get radii from LP
        xs = pos_opt[0::2]
        ys = pos_opt[1::2]
        rs = compute_radii_lp(xs, ys, w, h)
        if rs is not None:
            check_and_update(xs, ys, rs, w, h)
        
        # Phase 2: SA refinement of centers
        if time.time() < deadline - 10:
            pos_sa = sa_optimize_centers(w, h, pos_opt, max_iters=1500, seed=42 + int(w * 100))
            xs_sa = pos_sa[0::2]
            ys_sa = pos_sa[1::2]
            rs_sa = compute_radii_lp(xs_sa, ys_sa, w, h)
            if rs_sa is not None:
                check_and_update(xs_sa, ys_sa, rs_sa, w, h)
            
            # Phase 3: Another SA with perturbation
            if time.time() < deadline - 8:
                rng_p = np.random.default_rng(777)
                pos_pert = pos_sa.copy()
                for i in range(N):
                    pos_pert[2 * i] += rng_p.normal(0, 0.02)
                    pos_pert[2 * i + 1] += rng_p.normal(0, 0.02)
                pos_sa2 = sa_optimize_centers(w, h, pos_pert, max_iters=1000, seed=999)
                xs_sa2 = pos_sa2[0::2]
                ys_sa2 = pos_sa2[1::2]
                rs_sa2 = compute_radii_lp(xs_sa2, ys_sa2, w, h)
                if rs_sa2 is not None:
                    check_and_update(xs_sa2, ys_sa2, rs_sa2, w, h)
        
        # Phase 4: Polish with gradient descent on a combined objective
        if time.time() < deadline - 5 and rs is not None:
            # Use the best radii as targets and refine centers to maximize min separation
            def combined_obj(pos):
                xs_c = pos[0::2]
                ys_c = pos[1::2]
                # Penalty for circles being too close relative to their radii
                pen = 0.0
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = xs_c[i] - xs_c[j]
                        dy = ys_c[i] - ys_c[j]
                        dist = np.sqrt(dx * dx + dy * dy)
                        min_dist = rs[i] + rs[j]
                        if dist < min_dist:
                            pen += (min_dist - dist) ** 2
                # Boundary penalties
                for i in range(N):
                    if xs_c[i] < rs[i]:
                        pen += (rs[i] - xs_c[i]) ** 2
                    if xs_c[i] > w - rs[i]:
                        pen += (xs_c[i] - (w - rs[i])) ** 2
                    if ys_c[i] < rs[i]:
                        pen += (rs[i] - ys_c[i]) ** 2
                    if ys_c[i] > h - rs[i]:
                        pen += (ys_c[i] - (h - rs[i])) ** 2
                return pen
            
            bounds_c = [(0, w), (0, h)] * N
            try:
                res_c = minimize(combined_obj, pos_opt, method='L-BFGS-B',
                                 bounds=bounds_c, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
                xs_c = res_c.x[0::2]
                ys_c = res_c.x[1::2]
                # Re-solve LP with refined centers
                rs_c = compute_radii_lp(xs_c, ys_c, w, h)
                if rs_c is not None:
                    check_and_update(xs_c, ys_c, rs_c, w, h)
            except Exception:
                pass

    # Fallback
    if best_result is None:
        w, h = 1.4, 0.6
        a, b = 7, 3
        r = 0.1 - 1e-4
        best_result = np.zeros(3 * N)
        for i in range(N):
            col = i % a
            row = i // a
            best_result[3 * i] = (col + 0.5) * w / a
            best_result[3 * i + 1] = (row + 0.5) * h / b
            best_result[3 * i + 2] = r

    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3 * i:3 * i + 3]
    return result
