"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    best_value = -np.inf
    best_result = None

    def _strictly_valid(circles, w, h, tol=1e-8):
        """Full-pair strict validity check. circles: (N,3)."""
        n = circles.shape[0]
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        if np.any(rs < 1e-9):
            return False
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return False
        for i in range(n):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d2 = dx * dx + dy * dy
            need = rs[i] + rs[i + 1:] - tol
            if np.any(d2 < need * need):
                return False
        return True

    def _evaluate(circles, w, h, tol=1e-7):
        """Return sum of radii if valid, else -inf."""
        if not _strictly_valid(circles, w, h, tol):
            return -np.inf
        return float(np.sum(circles[:, 2]))

    def check_and_update(circles, w, h):
        nonlocal best_value, best_result
        if time_left() < 1.0:
            return
        val = _evaluate(circles, w, h)
        if val > best_value:
            best_value = val
            best_result = circles.copy()

    def _softplus(x, beta=100.0):
        xb = beta * x
        return np.where(xb > 0,
                        x + np.log1p(np.exp(-xb)) / beta,
                        np.log1p(np.exp(xb)) / beta)

    def _softplus_grad(x, beta=100.0):
        return 1.0 / (1.0 + np.exp(-beta * x))

    def _objective_and_grad(x, w, h, beta=100.0, eq_weight=0.05):
        """Softplus-penalized objective with analytical gradients (vectorized)."""
        c = x.reshape(-1, 3)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]

        obj = -np.sum(rs)
        grad = np.zeros_like(x)

        # Equalization pressure
        if eq_weight > 0:
            r_mean = np.mean(rs)
            r_dev = rs - r_mean
            obj += eq_weight * np.sum(r_dev * r_dev)
            grad[2::3] += -2.0 * eq_weight * r_dev

        # Boundary penalties
        v1 = rs - xs
        obj += np.sum(_softplus(v1, beta)); g1 = _softplus_grad(v1, beta)
        grad[0::3] -= g1; grad[2::3] += g1
        v2 = xs + rs - w
        obj += np.sum(_softplus(v2, beta)); g2 = _softplus_grad(v2, beta)
        grad[0::3] += g2; grad[2::3] += g2
        v3 = rs - ys
        obj += np.sum(_softplus(v3, beta)); g3 = _softplus_grad(v3, beta)
        grad[1::3] -= g3; grad[2::3] += g3
        v4 = ys + rs - h
        obj += np.sum(_softplus(v4, beta)); g4 = _softplus_grad(v4, beta)
        grad[1::3] += g4; grad[2::3] += g4
        v5 = -rs
        obj += np.sum(_softplus(v5, beta)); g5 = _softplus_grad(v5, beta)
        grad[2::3] -= g5

        # Pairwise overlap penalties
        for i in range(N):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            v = rs[i] + rs[i + 1:] - d
            obj += np.sum(_softplus(v, beta))
            g = _softplus_grad(v, beta)
            grad[3 * i]     += -np.sum(g * dx / d)
            grad[3 * i + 1] += -np.sum(g * dy / d)
            grad[3 * i + 2] += np.sum(g)
            idx = np.arange(i + 1, N)
            grad[3 * idx]     += g * dx / d
            grad[3 * idx + 1] += g * dy / d
            grad[3 * idx + 2] += g

        grad[2::3] += -1.0
        return obj, grad

    def _delaunay_neighbors(centers):
        try:
            tri = Delaunay(centers)
            nbrs = [set() for _ in range(N)]
            for i, j, k in tri.simplices:
                nbrs[i].add(j); nbrs[i].add(k)
                nbrs[j].add(i); nbrs[j].add(k)
                nbrs[k].add(i); nbrs[k].add(j)
            return [sorted(s) for s in nbrs]
        except Exception:
            return [list(range(N)) for _ in range(N)]

    def _fast_valid(circles, w, h, neighbors, tol=1e-9):
        n = circles.shape[0]
        xs = circles[:, 0]; ys = circles[:, 1]; rs = circles[:, 2]
        if np.any(rs < 1e-9):
            return False
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return False
        for i in range(n):
            for j in neighbors[i]:
                if j <= i:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d2 = dx * dx + dy * dy
                need = rs[i] + rs[j] - tol
                if d2 < need * need:
                    return False
        return True

    def _hex_init(w, h, seed):
        rng = random.Random(seed)
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
            [5, 4, 4, 4, 4],
            [4, 5, 4, 4, 4],
            [3, 5, 4, 4, 5],
            [4, 4, 5, 4, 4],
            [2, 5, 4, 5, 5],
            [5, 5, 5, 5, 1],
            [1, 5, 5, 5, 5],
            [3, 4, 5, 4, 5],
            [5, 4, 5, 4, 3],
        ]
        best = None
        best_val = -np.inf
        for row_counts in row_configs:
            if sum(row_counts) != N:
                continue
            n_rows = len(row_counts)
            max_cols = max(row_counts)
            est_r_h = w / (max_cols + 0.5)
            est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
            r_est = min(est_r_h, est_r_v) * 0.95
            if r_est < 0.01:
                continue
            circles = []
            for row_idx, n_cols in enumerate(row_counts):
                y = (row_idx + 0.5) * r_est * np.sqrt(3)
                x_offset = r_est if row_idx % 2 == 1 else 0.0
                row_width = n_cols * 2 * r_est
                center_x = w / 2
                for col_idx in range(n_cols):
                    x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                    circles.append([x, y, r_est])
            circles = np.array(circles)
            valid = True
            for i in range(N):
                xi, yi, ri = circles[i]
                if ri < 1e-9 or xi - ri < -1e-6 or xi + ri > w + 1e-6 or yi - ri < -1e-6 or yi + ri > h + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        xi, yi, ri = circles[i]
                        xj, yj, rj = circles[j]
                        dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
                        min_dist = ri + rj
                        if dist_sq < (min_dist - 1e-7) ** 2:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                val = float(np.sum(circles[:, 2]))
                if val > best_val:
                    best_val = val
                    best = circles.copy()
        if best is None:
            a = int(np.ceil(np.sqrt(N * w / h)))
            b = int(np.ceil(N / a))
            while a * b < N:
                b += 1
            circles = []
            for i in range(N):
                col = i % a
                row = i // a
                r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
                x = (col + 0.5) * w / a
                y = (row + 0.5) * h / b
                circles.append([x, y, r])
            best = np.array(circles)
        return best

    def _grid_init(w, h, seed):
        rng = random.Random(seed)
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        return np.array(circles)

    def _random_init(w, h, seed):
        rng = np.random.RandomState(seed)
        r0 = min(w, h) / 2.0 * 0.25
        circles = np.zeros((N, 3))
        circles[:, 0] = rng.uniform(r0, w - r0, N)
        circles[:, 1] = rng.uniform(r0, h - r0, N)
        circles[:, 2] = r0 * rng.uniform(0.5, 1.0, N)
        return circles

    def _polish(circles, w, h, beta_start=10.0, max_iter=30):
        """Multi-phase polish with increasing beta."""
        x = circles.flatten()
        best_x = x.copy()
        best_val = -np.inf

        if _strictly_valid(circles, w, h):
            best_val = np.sum(circles[:, 2])

        betas = [beta_start, beta_start * 10, beta_start * 100, beta_start * 1000]

        for beta in betas:
            if time_left() < 3.0:
                break

            bounds = []
            r_max = min(w, h) / 2.0
            for i in range(N):
                bounds.append((0, w))
                bounds.append((0, h))
                bounds.append((1e-4, r_max))

            try:
                res = minimize(_objective_and_grad, x, method='L-BFGS-B', bounds=bounds,
                               jac=True, args=(w, h, beta, 0.05),
                               options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10})
                if time_left() < 2.0:
                    break
                new_circles = res.x.reshape(-1, 3)
                if _strictly_valid(new_circles, w, h):
                    val = np.sum(new_circles[:, 2])
                    if val > best_val:
                        best_val = val
                        best_x = res.x.copy()
                x = res.x
            except Exception:
                pass

        return best_x.reshape(-1, 3)

    def _contact_graph_polish(circles, w, h):
        """High-precision polish using very tight beta and contact-aware tolerances."""
        if time_left() < 10.0:
            return circles

        x0 = circles.flatten()
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))

        stages = [
            (300.0, 0.005, 2000),
            (800.0, 0.002, 3000),
            (2000.0, 0.001, 5000),
            (5000.0, 0.0005, 5000),
        ]

        best_x = x0.copy()
        best_val = -np.inf
        if _strictly_valid(circles, w, h):
            best_val = float(np.sum(circles[:, 2]))

        for beta, eq_w, iters in stages:
            if time_left() < 5.0:
                break
            try:
                res = minimize(_objective_and_grad, x0, method='L-BFGS-B', bounds=bounds,
                               jac=True, args=(w, h, beta, eq_w),
                               options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-12,
                                        'maxls': 80, 'maxfun': 50000})
                if time_left() < 3.0:
                    break
                new_circles = res.x.reshape(-1, 3)
                if _strictly_valid(new_circles, w, h):
                    val = float(np.sum(new_circles[:, 2]))
                    if val > best_val:
                        best_val = val
                        best_x = res.x.copy()
                x0 = res.x
            except Exception:
                pass

        return best_x.reshape(-1, 3)

    def _physics_refine(circles, w, h, n_steps=2000, seed=42):
        """
        Physics-based refinement: treat circles as particles with repulsive forces.
        This is a different modelling family: instead of gradient-based optimization,
        we simulate a force-directed system that pushes overlapping circles apart
        and pulls them toward the walls to maximize packing density.
        """
        if time_left() < 8.0:
            return circles

        rng = np.random.RandomState(seed)
        xs = circles[:, 0].copy().astype(float)
        ys = circles[:, 1].copy().astype(float)
        rs = circles[:, 2].copy().astype(float)

        # Adaptive temperature schedule
        temp_start = 0.02
        temp_end = 0.001
        n_total = n_steps

        for step in range(n_total):
            if time_left() < 3.0:
                break

            # Annealing schedule
            t = step / max(n_total - 1, 1)
            temp = temp_start * (temp_end / temp_start) ** t

            forces_x = np.zeros(N)
            forces_y = np.zeros(N)
            forces_r = np.zeros(N)

            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d = np.sqrt(dx * dx + dy * dy + 1e-12)
                    overlap = rs[i] + rs[j] - d
                    if overlap > 0:
                        # Strong repulsive force proportional to overlap
                        f = overlap * 50.0
                        fx = f * dx / d
                        fy = f * dy / d
                        forces_x[i] += fx
                        forces_y[i] += fy
                        forces_x[j] -= fx
                        forces_y[j] -= fy
                        # Slightly shrink both to resolve overlap
                        forces_r[i] -= overlap * 0.5
                        forces_r[j] -= overlap * 0.5

            # Wall constraints: push circles away from walls
            for i in range(N):
                # Left wall
                pen = rs[i] - xs[i]
                if pen > 0:
                    forces_x[i] += pen * 80.0
                    forces_r[i] -= pen * 0.3
                # Right wall
                pen = xs[i] + rs[i] - w
                if pen > 0:
                    forces_x[i] -= pen * 80.0
                    forces_r[i] -= pen * 0.3
                # Bottom wall
                pen = rs[i] - ys[i]
                if pen > 0:
                    forces_y[i] += pen * 80.0
                    forces_r[i] -= pen * 0.3
                # Top wall
                pen = ys[i] + rs[i] - h
                if pen > 0:
                    forces_y[i] -= pen * 80.0
                    forces_r[i] -= pen * 0.3

            # Pressure to grow radii (maximize sum of radii)
            forces_r += 0.01

            # Update positions with damping and noise
            damping = 0.7
            for i in range(N):
                xs[i] += damping * forces_x[i] * 0.01 + temp * rng.randn() * 0.005
                ys[i] += damping * forces_y[i] * 0.01 + temp * rng.randn() * 0.005
                rs[i] += damping * forces_r[i] * 0.01 + temp * rng.randn() * 0.002

                # Clamp to bounds
                xs[i] = max(rs[i] + 1e-9, min(w - rs[i] - 1e-9, xs[i]))
                ys[i] = max(rs[i] + 1e-9, min(h - rs[i] - 1e-9, ys[i]))
                rs[i] = max(1e-4, min(min(w, h) / 2.0, rs[i]))

        new_circles = np.column_stack([xs, ys, rs])
        return new_circles

    def _aspect_refine(circles, w, h):
        """Try small perturbations of aspect ratio to find better fit."""
        if time_left() < 15.0:
            return circles, w, h

        best_circles = circles
        best_val = _evaluate(circles, w, h)
        best_w, best_h = w, h

        # Try small aspect ratio perturbations
        for dw in [-0.05, -0.03, -0.02, 0.02, 0.03, 0.05]:
            if time_left() < 5.0:
                break
            new_w = w + dw
            new_h = 2.0 - new_w
            if new_w < 0.3 or new_h < 0.3:
                continue

            # Scale circles to fit new rectangle
            scaled = circles.copy()
            scaled[:, 0] *= new_w / w
            scaled[:, 1] *= new_h / h
            # Keep radii the same initially, let polish adjust
            polished = _polish(scaled, new_w, new_h, beta_start=50.0, max_iter=20)
            val = _evaluate(polished, new_w, new_h)
            if val > best_val:
                best_val = val
                best_circles = polished
                best_w, best_h = new_w, new_h

        return best_circles, best_w, best_h

    # Main loop over aspect ratios
    widths = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

    for w in widths:
        if time_left() < 15.0:
            break
        h = 2.0 - w

        inits = []
        hex_circles = _hex_init(w, h, seed=42)
        inits.append(hex_circles)
        grid_circles = _grid_init(w, h, seed=42)
        inits.append(grid_circles)
        rand_circles = _random_init(w, h, seed=42)
        inits.append(rand_circles)

        for init_circles in inits:
            if time_left() < 12.0:
                break

            # Phase 1: Standard polish
            polished = _polish(init_circles, w, h, beta_start=10.0, max_iter=30)
            check_and_update(polished, w, h)

            # Phase 2: Physics-based refinement (different modelling family)
            if time_left() > 10.0:
                phys = _physics_refine(polished, w, h, n_steps=1500, seed=42)
                check_and_update(phys, w, h)

            # Phase 3: Contact-graph high-precision polish
            if time_left() > 12.0:
                # Use the best of polished and phys
                if _evaluate(phys, w, h) > _evaluate(polished, w, h):
                    start_for_cg = phys
                else:
                    start_for_cg = polished
                cg_polished = _contact_graph_polish(start_for_cg, w, h)
                check_and_update(cg_polished, w, h)

            # Phase 4: Aspect ratio refinement
            if time_left() > 10.0:
                refined, new_w, new_h = _aspect_refine(cg_polished, w, h)
                if new_w != w or new_h != h:
                    check_and_update(refined, new_w, new_h)

    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros((N, 3))
        for i in range(a):
            for j in range(b):
                idx = i * b + j
                best_result[idx, 0] = (i + 0.5) * w / a
                best_result[idx, 1] = (j + 0.5) * h / b
                best_result[idx, 2] = r

    return best_result
# EVOLVE-BLOCK-END
