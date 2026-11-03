"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.
Uses soft-min surrogate for minimum distance, Delaunay-accelerated constraints,
multi-phase beta annealing, and multi-start over aspect ratios and row configs.
"""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def evaluate(circles, w, h):
        """Return sum of radii if valid, else -inf."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
                return -np.inf
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if d < ri + rj - 1e-6:
                    return -np.inf
        return float(np.sum(circles[:, 2]))

    def fast_valid(circles, w, h, tol=1e-5):
        """Fast vectorized validity check."""
        xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                if d < rs[i] + rs[j] - tol:
                    return False
        return True

    def delaunay_pairs(circles, w, h, margin=0.3):
        """Get Delaunay neighbor pairs for constraint reduction."""
        pts = circles[:, :2]
        try:
            tri = Delaunay(pts)
            pairs = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = simplex[i], simplex[j]
                        pairs.add((min(a, b), max(a, b)))
            # Add boundary-adjacent pairs
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.sqrt((pts[i,0]-pts[j,0])**2 + (pts[i,1]-pts[j,1])**2)
                    if d < (circles[i,2] + circles[j,2]) * (1 + margin):
                        pairs.add((i, j))
            return list(pairs)
        except Exception:
            return [(i, j) for i in range(N) for j in range(i + 1, N)]

    def hex_lattice_init(w, h, row_counts, seed=0):
        """Generate hexagonal lattice packing."""
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        est_r_h = w / (max_cols + 0.5)
        est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1)
        r_est = min(est_r_h, est_r_v)
        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = (row_idx + 0.5) * r_est * np.sqrt(3)
            x_offset = 0.5 * r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            center_x = w / 2
            for col_idx in range(n_cols):
                x = center_x - row_width / 2 + (col_idx + 0.5) * 2 * r_est
                circles.append([x, y, r_est])
        if circles:
            ys = np.array([c[1] for c in circles])
            y_min, y_max = ys.min(), ys.max()
            y_center = (y_min + y_max) / 2
            shift = h / 2 - y_center
            for c in circles:
                c[1] += shift
        return np.array(circles)

    def grid_init(w, h, a, b, seed=0):
        """Grid initialization."""
        circles = np.zeros((N, 3))
        for i in range(N):
            col = i % a
            row = i // a
            circles[i, 0] = (col + 0.5) * w / a
            circles[i, 1] = (row + 0.5) * h / b
            circles[i, 2] = min(w / a, h / b) * 0.45
        return circles

    def random_init(w, h, seed=0):
        """Random initialization."""
        rng = np.random.default_rng(seed)
        r_max = min(w, h) / 2 * 0.3
        circles = np.zeros((N, 3))
        circles[:, 0] = rng.uniform(0, w, N)
        circles[:, 1] = rng.uniform(0, h, N)
        circles[:, 2] = rng.uniform(0.02, r_max, N)
        return circles

    def farthest_point_init(w, h, seed=0):
        """Farthest-point sampling initialization."""
        rng = np.random.default_rng(seed)
        pts = []
        # Start with a random point
        pts.append(rng.uniform([0.1*w, 0.1*h], [0.9*w, 0.9*h]))
        for _ in range(N - 1):
            # Sample many candidates, pick farthest from existing
            candidates = rng.uniform([0.1*w, 0.1*h], [0.9*w, 0.9*h], (200, 2))
            best_idx = 0
            best_min_dist = -1
            for c in candidates:
                min_d = min(np.linalg.norm(c - p) for p in pts)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = c
            pts.append(best_idx)
        pts = np.array(pts)
        # Estimate radius from nearest neighbor distance
        dists = []
        for i in range(N):
            min_d = min(np.linalg.norm(pts[i] - pts[j]) for j in range(N) if j != i)
            dists.append(min_d)
        r_est = min(dists) / 2 * 0.95
        circles = np.column_stack([pts, np.full(N, r_est)])
        return circles

    def optimize_softmin(circles0, w, h, max_iter=300, beta=100.0, use_delaunay=True):
        """Optimize using L-BFGS-B with soft-min surrogate for min distance."""
        x0 = circles0.flatten()
        n = N

        def objective_and_grad(x):
            c = x.reshape(-1, 3)
            xs, ys, rs = c[:, 0], c[:, 1], c[:, 2]
            val = -np.sum(rs)
            penalty = 0.0
            grad = np.zeros_like(x)

            # Boundary constraints (softplus)
            for i in range(n):
                v1 = rs[i] - xs[i]
                sp = np.log1p(np.exp(beta * v1)) / beta if beta * v1 < 50 else v1
                penalty += sp
                g = 1.0 / (1.0 + np.exp(-beta * v1))
                grad[3*i] += -g
                grad[3*i+2] += g

                v2 = xs[i] + rs[i] - w
                sp = np.log1p(np.exp(beta * v2)) / beta if beta * v2 < 50 else v2
                penalty += sp
                g = 1.0 / (1.0 + np.exp(-beta * v2))
                grad[3*i] += g
                grad[3*i+2] += g

                v3 = rs[i] - ys[i]
                sp = np.log1p(np.exp(beta * v3)) / beta if beta * v3 < 50 else v3
                penalty += sp
                g = 1.0 / (1.0 + np.exp(-beta * v3))
                grad[3*i+1] += -g
                grad[3*i+2] += g

                v4 = ys[i] + rs[i] - h
                sp = np.log1p(np.exp(beta * v4)) / beta if beta * v4 < 50 else v4
                penalty += sp
                g = 1.0 / (1.0 + np.exp(-beta * v4))
                grad[3*i+1] += g
                grad[3*i+2] += g

                v5 = -rs[i]
                sp = np.log1p(np.exp(beta * v5)) / beta if beta * v5 < 50 else v5
                penalty += sp
                g = 1.0 / (1.0 + np.exp(-beta * v5))
                grad[3*i+2] += -g

            # Pairwise distance constraints using soft-min
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            if use_delaunay:
                try:
                    pairs = delaunay_pairs(c, w, h)
                except Exception:
                    pass

            for (i, j) in pairs:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy + 1e-12)
                v = rs[i] + rs[j] - d
                sp = np.log1p(np.exp(beta * v)) / beta if beta * v < 50 else v
                penalty += sp
                g = 1.0 / (1.0 + np.exp(-beta * v))
                grad[3*i] += -g * dx / d
                grad[3*i+1] += -g * dy / d
                grad[3*i+2] += g
                grad[3*j] += g * dx / d
                grad[3*j+1] += g * dy / d
                grad[3*j+2] += g

            obj = val + penalty
            for i in range(n):
                grad[3*i+2] += -1.0

            return obj, grad

        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        circles = result.x.reshape(-1, 3)
        val = evaluate(circles, w, h)
        return circles, val

    def polish(circles0, w, h, max_iter=500):
        """High-precision polish with increasing beta."""
        circles = circles0
        for beta in [200.0, 500.0, 1000.0]:
            if time.time() > deadline - 5:
                break
            circles, val = optimize_softmin(circles, w, h, max_iter=max_iter, beta=beta)
        return circles, evaluate(circles, w, h)

    def sa_joint_optimize(circles0, w0, max_iters=2000, seed=42):
        """Simulated annealing jointly optimizing w, positions, and radii."""
        xs = circles0[:, 0].copy()
        ys = circles0[:, 1].copy()
        rs = circles0[:, 2].copy()
        w = w0
        h = 2.0 - w

        T = 0.08
        cooling = 0.9995
        rng = np.random.default_rng(seed)

        def cost(xs, ys, rs, w, h):
            bpen = 0.0
            bpen += np.sum(np.maximum(0, rs - xs) ** 2)
            bpen += np.sum(np.maximum(0, xs - (w - rs)) ** 2)
            bpen += np.sum(np.maximum(0, rs - ys) ** 2)
            bpen += np.sum(np.maximum(0, ys - (h - rs)) ** 2)
            overlap = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    viol = max(0, rs[i] + rs[j] - d)
                    overlap += viol ** 2
            return bpen + overlap - np.sum(rs)

        current_cost = cost(xs, ys, rs, w, h)

        for it in range(max_iters):
            if time.time() > deadline:
                break
            move_type = rng.random()
            if move_type < 0.15:
                old_w = w
                old_xs, old_ys, old_rs = xs.copy(), ys.copy(), rs.copy()
                w += rng.normal(0, T * 0.5)
                w = np.clip(w, 0.4, 1.6)
                h = 2.0 - w
                for i in range(N):
                    rs[i] = np.clip(rs[i], 1e-4, min(w, h) / 2.0)
                    xs[i] = np.clip(xs[i], rs[i], w - rs[i])
                    ys[i] = np.clip(ys[i], rs[i], h - rs[i])
                new_cost = cost(xs, ys, rs, w, h)
                delta = new_cost - current_cost
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    w = old_w
                    h = 2.0 - w
                    xs, ys, rs = old_xs, old_ys, old_rs
                    current_cost = cost(xs, ys, rs, w, h)
            else:
                idx = rng.integers(0, N)
                old_x, old_y, old_r = xs[idx], ys[idx], rs[idx]
                scale = T * 0.3
                xs[idx] += rng.normal(0, scale)
                ys[idx] += rng.normal(0, scale)
                rs[idx] += rng.normal(0, scale * 0.5)
                rs[idx] = np.clip(rs[idx], 1e-4, min(w, h) / 2.0)
                xs[idx] = np.clip(xs[idx], rs[idx], w - rs[idx])
                ys[idx] = np.clip(ys[idx], rs[idx], h - rs[idx])
                new_cost = cost(xs, ys, rs, w, h)
                delta = new_cost - current_cost
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    xs[idx], ys[idx], rs[idx] = old_x, old_y, old_r
            T *= cooling

        circles = np.column_stack([xs, ys, rs])
        return circles, w

    best_value = -np.inf
    best_result = None
    best_w = 1.0
    best_h = 1.0

    row_configs = [
        [3, 4, 4, 4, 3, 3],
        [4, 4, 4, 4, 3, 2],
        [2, 4, 4, 4, 4, 3],
        [3, 4, 4, 4, 4, 2],
        [4, 3, 4, 4, 4, 2],
        [2, 3, 4, 4, 4, 4],
        [3, 3, 4, 4, 4, 3],
        [4, 4, 3, 4, 4, 2],
        [2, 4, 3, 4, 4, 4],
        [3, 4, 3, 4, 4, 3],
        [4, 4, 4, 3, 3, 3],
        [3, 4, 4, 3, 4, 3],
    ]

    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]

    init_count = 0
    for aspect in aspect_ratios:
        if time.time() > deadline - 15:
            break
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)

        # Hex lattice inits
        for rc in row_configs[:6]:
            if time.time() > deadline - 12:
                break
            circles0 = hex_lattice_init(w, h, rc, seed=init_count)
            val = evaluate(circles0, w, h)
            if val > best_value:
                best_value = val
                best_result = circles0.copy()
                best_w, best_h = w, h

            if time.time() < deadline - 25:
                circles, val = optimize_softmin(circles0, w, h, max_iter=250, beta=80.0)
                if val > best_value:
                    best_value = val
                    best_result = circles.copy()
                    best_w, best_h = w, h

                if time.time() < deadline - 15:
                    circles_p, val_p = polish(circles, w, h, max_iter=300)
                    if val_p > best_value:
                        best_value = val_p
                        best_result = circles_p.copy()
                        best_w, best_h = w, h
            init_count += 1

        # Grid init
        if time.time() > deadline - 12:
            break
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles0 = grid_init(w, h, a, b, seed=init_count)
        val = evaluate(circles0, w, h)
        if val > best_value:
            best_value = val
            best_result = circles0.copy()
            best_w, best_h = w, h

        if time.time() < deadline - 20:
            circles, val = optimize_softmin(circles0, w, h, max_iter=250, beta=80.0)
            if val > best_value:
                best_value = val
                best_result = circles.copy()
                best_w, best_h = w, h

            if time.time() < deadline - 12:
                circles_p, val_p = polish(circles, w, h, max_iter=300)
                if val_p > best_value:
                    best_value = val_p
                    best_result = circles_p.copy()
                    best_w, best_h = w, h
        init_count += 1

        # Farthest point init
        if time.time() > deadline - 12:
            break
        circles0 = farthest_point_init(w, h, seed=init_count)
        val = evaluate(circles0, w, h)
        if val > best_value:
            best_value = val
            best_result = circles0.copy()
            best_w, best_h = w, h

        if time.time() < deadline - 18:
            circles, val = optimize_softmin(circles0, w, h, max_iter=250, beta=80.0)
            if val > best_value:
                best_value = val
                best_result = circles.copy()
                best_w, best_h = w, h

            if time.time() < deadline - 10:
                circles_p, val_p = polish(circles, w, h, max_iter=250)
                if val_p > best_value:
                    best_value = val_p
                    best_result = circles_p.copy()
                    best_w, best_h = w, h
        init_count += 1

        # Random init
        if time.time() > deadline - 12:
            break
        circles0 = random_init(w, h, seed=init_count)
        if time.time() < deadline - 18:
            circles, val = optimize_softmin(circles0, w, h, max_iter=250, beta=80.0)
            if val > best_value:
                best_value = val
                best_result = circles.copy()
                best_w, best_h = w, h

            if time.time() < deadline - 10:
                circles_p, val_p = polish(circles, w, h, max_iter=250)
                if val_p > best_value:
                    best_value = val_p
                    best_result = circles_p.copy()
                    best_w, best_h = w, h
        init_count += 1

    # SA-based refinement on best found
    if best_result is not None and time.time() < deadline - 25:
        circles_sa, w_sa = sa_joint_optimize(best_result, best_w, max_iters=2000, seed=42)
        h_sa = 2.0 - w_sa
        val = evaluate(circles_sa, w_sa, h_sa)
        if val > best_value:
            best_value = val
            best_result = circles_sa.copy()
            best_w, best_h = w_sa, h_sa

        if time.time() < deadline - 12:
            circles_p, val_p = polish(circles_sa, w_sa, h_sa, max_iter=400)
            if val_p > best_value:
                best_value = val_p
                best_result = circles_p.copy()
                best_w, best_h = w_sa, h_sa

    # Fallback
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros((N, 3))
        for i in range(a):
            for j in range(b):
                idx = i * b + j
                if idx >= N:
                    break
                best_result[idx, 0] = (i + 0.5) * w / a
                best_result[idx, 1] = (j + 0.5) * h / b
                best_result[idx, 2] = r

    return best_result
# EVOLVE-BLOCK-END
