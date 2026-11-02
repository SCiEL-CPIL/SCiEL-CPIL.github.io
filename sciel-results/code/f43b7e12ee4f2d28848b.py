"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

    def hex_lattice_init(w, h, row_counts, seed=0):
        """Generate hexagonal lattice packing for given row counts."""
        rng = np.random.default_rng(seed)
        n_rows = len(row_counts)
        max_cols = max(row_counts)

        # Estimate radius
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

        # Center vertically
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
        rng = np.random.default_rng(seed)
        circles = np.zeros((N, 3))
        for i in range(N):
            col = i % a
            row = i // a
            circles[i, 0] = (col + 0.5) * w / a
            circles[i, 1] = (row + 0.5) * h / b
            circles[i, 2] = min(w / a, h / b) * 0.45
        return circles

    def random_init(w, h, seed=0):
        """Random initialization with small radii."""
        rng = np.random.default_rng(seed)
        r_max = min(w, h) / 2 * 0.3
        circles = np.zeros((N, 3))
        circles[:, 0] = rng.uniform(0, w, N)
        circles[:, 1] = rng.uniform(0, h, N)
        circles[:, 2] = rng.uniform(0.02, r_max, N)
        return circles

    def softplus(x, beta=100.0):
        """Numerically stable softplus."""
        x_beta = beta * x
        result = np.where(x_beta > 0,
                         x + np.log1p(np.exp(-x_beta)) / beta,
                         np.log1p(np.exp(x_beta)) / beta)
        return result

    def softplus_grad(x, beta=100.0):
        """Gradient of softplus: sigmoid(beta*x)."""
        return 1.0 / (1.0 + np.exp(-beta * x))

    def optimize_lbfgs(circles0, w, h, max_iter=300, beta=100.0):
        """Optimize using L-BFGS-B with softplus penalties."""
        x0 = circles0.flatten()

        def objective_and_grad(x):
            c = x.reshape(-1, 3)
            val = -np.sum(c[:, 2])
            penalty = 0.0
            grad = np.zeros_like(x)

            for i in range(N):
                xi, yi, ri = c[i]
                # x bounds
                v1 = ri - xi
                penalty += softplus(v1, beta)
                g1 = softplus_grad(v1, beta)
                grad[3 * i] += -g1
                grad[3 * i + 2] += g1

                v2 = xi + ri - w
                penalty += softplus(v2, beta)
                g2 = softplus_grad(v2, beta)
                grad[3 * i] += g2
                grad[3 * i + 2] += g2

                # y bounds
                v3 = ri - yi
                penalty += softplus(v3, beta)
                g3 = softplus_grad(v3, beta)
                grad[3 * i + 1] += -g3
                grad[3 * i + 2] += g3

                v4 = yi + ri - h
                penalty += softplus(v4, beta)
                g4 = softplus_grad(v4, beta)
                grad[3 * i + 1] += g4
                grad[3 * i + 2] += g4

                # positive radius
                v5 = -ri
                penalty += softplus(v5, beta)
                g5 = softplus_grad(v5, beta)
                grad[3 * i + 2] += -g5

            for i in range(N):
                for j in range(i + 1, N):
                    xi, yi, ri = c[i]
                    xj, yj, rj = c[j]
                    dx = xi - xj
                    dy = yi - yj
                    d = np.sqrt(dx * dx + dy * dy + 1e-12)
                    v = ri + rj - d
                    penalty += softplus(v, beta)
                    g = softplus_grad(v, beta)
                    grad[3 * i] += -g * dx / d
                    grad[3 * i + 1] += -g * dy / d
                    grad[3 * i + 2] += g
                    grad[3 * j] += g * dx / d
                    grad[3 * j + 1] += g * dy / d
                    grad[3 * j + 2] += g

            obj = val + penalty
            for i in range(N):
                grad[3 * i + 2] += -1.0

            return obj, grad

        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        circles = result.x.reshape(-1, 3)
        val = evaluate(circles, w, h)
        return circles, val

    def polish(circles0, w, h, max_iter=500, beta=200.0):
        """High-precision polish pass."""
        return optimize_lbfgs(circles0, w, h, max_iter=max_iter, beta=beta)

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
            # Boundary penalty
            bpen = 0.0
            bpen += np.sum(np.maximum(0, rs - xs) ** 2)
            bpen += np.sum(np.maximum(0, xs - (w - rs)) ** 2)
            bpen += np.sum(np.maximum(0, rs - ys) ** 2)
            bpen += np.sum(np.maximum(0, ys - (h - rs)) ** 2)
            # Overlap penalty
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
                # Perturb aspect ratio
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
                # Perturb a random circle
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
    ]

    # Multiple aspect ratios
    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]

    init_count = 0
    for aspect in aspect_ratios:
        if time.time() > deadline - 15:
            break
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)

        # Try hex lattice inits
        for rc in row_configs[:5]:
            if time.time() > deadline - 10:
                break
            circles0 = hex_lattice_init(w, h, rc, seed=init_count)
            val = evaluate(circles0, w, h)
            if val > best_value:
                best_value = val
                best_result = circles0.copy()
                best_w, best_h = w, h

            if time.time() < deadline - 20:
                circles, val = optimize_lbfgs(circles0, w, h, max_iter=200)
                if val > best_value:
                    best_value = val
                    best_result = circles.copy()
                    best_w, best_h = w, h

                if time.time() < deadline - 10:
                    circles_p, val_p = polish(circles, w, h, max_iter=300)
                    if val_p > best_value:
                        best_value = val_p
                        best_result = circles_p.copy()
                        best_w, best_h = w, h

            init_count += 1

        # Try grid init
        if time.time() > deadline - 10:
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

        if time.time() < deadline - 15:
            circles, val = optimize_lbfgs(circles0, w, h, max_iter=200)
            if val > best_value:
                best_value = val
                best_result = circles.copy()
                best_w, best_h = w, h

            if time.time() < deadline - 8:
                circles_p, val_p = polish(circles, w, h, max_iter=300)
                if val_p > best_value:
                    best_value = val_p
                    best_result = circles_p.copy()
                    best_w, best_h = w, h

        init_count += 1

        # Try random init
        if time.time() > deadline - 10:
            break
        circles0 = random_init(w, h, seed=init_count)
        if time.time() < deadline - 15:
            circles, val = optimize_lbfgs(circles0, w, h, max_iter=200)
            if val > best_value:
                best_value = val
                best_result = circles.copy()
                best_w, best_h = w, h

        init_count += 1

    # SA-based refinement on best found aspect ratio
    if best_result is not None and time.time() < deadline - 20:
        circles_sa, w_sa = sa_joint_optimize(best_result, best_w, max_iters=1500, seed=42)
        h_sa = 2.0 - w_sa
        val = evaluate(circles_sa, w_sa, h_sa)
        if val > best_value:
            best_value = val
            best_result = circles_sa.copy()
            best_w, best_h = w_sa, h_sa

        if time.time() < deadline - 10:
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
