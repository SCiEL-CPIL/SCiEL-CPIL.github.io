import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def elapsed():
        return time.time() - start_time

    def check_time():
        return elapsed() > deadline

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

    def softplus(x, beta=100.0):
        x_beta = beta * x
        result = np.where(x_beta > 0,
                         x + np.log1p(np.exp(-x_beta)) / beta,
                         np.log1p(np.exp(x_beta)) / beta)
        return result

    def softplus_grad(x, beta=100.0):
        return 1.0 / (1.0 + np.exp(-beta * x))

    def objective_and_grad(x, w, h, beta=100.0):
        """
        Force-directed penalty objective with analytical gradients.
        x: [x0, y0, r0, x1, y1, r1, ...]
        """
        c = x.reshape(-1, 3)
        n = len(c)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]

        obj = -np.sum(rs)
        grad = np.zeros_like(x)

        # Boundary penalties
        v1 = rs - xs
        obj += np.sum(softplus(v1, beta))
        g1 = softplus_grad(v1, beta)
        grad[0::3] -= g1
        grad[2::3] += g1

        v2 = xs + rs - w
        obj += np.sum(softplus(v2, beta))
        g2 = softplus_grad(v2, beta)
        grad[0::3] += g2
        grad[2::3] += g2

        v3 = rs - ys
        obj += np.sum(softplus(v3, beta))
        g3 = softplus_grad(v3, beta)
        grad[1::3] -= g3
        grad[2::3] += g3

        v4 = ys + rs - h
        obj += np.sum(softplus(v4, beta))
        g4 = softplus_grad(v4, beta)
        grad[1::3] += g4
        grad[2::3] += g4

        v5 = -rs
        obj += np.sum(softplus(v5, beta))
        g5 = softplus_grad(v5, beta)
        grad[2::3] -= g5

        # Pairwise overlap penalties (vectorized)
        for i in range(n):
            dx = xs[i] - xs[i+1:]
            dy = ys[i] - ys[i+1:]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)

            v = rs[i] + rs[i+1:] - d
            obj += np.sum(softplus(v, beta))
            g = softplus_grad(v, beta)

            grad[3*i] += -np.sum(g * dx / d)
            grad[3*i+1] += -np.sum(g * dy / d)
            grad[3*i+2] += np.sum(g)

            idx = np.arange(i+1, n)
            grad[3*idx] += g * dx / d
            grad[3*idx+1] += g * dy / d
            grad[3*idx+2] += g

        grad[2::3] += -1.0
        return obj, grad

    def delauany_neighbors(xs, ys):
        """Get Delaunay neighbor pairs."""
        try:
            pts = np.column_stack((xs, ys))
            tri = Delaunay(pts)
            pairs = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        pairs.add((min(a,b), max(a,b)))
            return list(pairs)
        except Exception:
            return [(i, j) for i in range(N) for j in range(i+1, N)]

    def lp_active_set_refine(circles, w, h, max_iters=50):
        """
        Use Delaunay-restricted LP to refine radii given fixed centers.
        Then use shadow prices as forces to adjust centers.
        """
        xs = circles[:, 0].copy()
        ys = circles[:, 1].copy()
        rs = circles[:, 2].copy()

        for it in range(max_iters):
            if check_time():
                break

            # Get Delaunay neighbors
            pairs = delauany_neighbors(xs, ys)

            # Build LP: maximize sum(r)
            # Variables: r_0, ..., r_{N-1}
            # Constraints:
            #   r_i >= 0
            #   r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
            #   r_i + r_j <= dist(i,j) for active pairs
            c_obj = -np.ones(N)  # minimize -sum(r)

            A_ub = []
            b_ub = []

            # Boundary constraints: r_i <= x_i, r_i <= w-x_i, r_i <= y_i, r_i <= h-y_i
            for i in range(N):
                # r_i <= x_i
                row = np.zeros(N); row[i] = 1.0
                A_ub.append(row); b_ub.append(xs[i])
                # r_i <= w - x_i
                row = np.zeros(N); row[i] = 1.0
                A_ub.append(row); b_ub.append(w - xs[i])
                # r_i <= y_i
                row = np.zeros(N); row[i] = 1.0
                A_ub.append(row); b_ub.append(ys[i])
                # r_i <= h - y_i
                row = np.zeros(N); row[i] = 1.0
                A_ub.append(row); b_ub.append(h - ys[i])

            # Pairwise constraints for Delaunay neighbors
            for (i, j) in pairs:
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                row = np.zeros(N); row[i] = 1.0; row[j] = 1.0
                A_ub.append(row); b_ub.append(d)

            bounds = [(0, None)] * N

            try:
                res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                              bounds=bounds, method='highs')
                if res.success:
                    rs_new = res.x
                    # Ensure non-negative
                    rs_new = np.maximum(rs_new, 0)
                    # Use shadow prices (dual values) as forces
                    # For now, just update radii
                    rs = rs_new
                else:
                    break
            except Exception:
                break

            # Use gradient of overlap constraints to nudge centers
            # For each active pair, if r_i + r_j > d, push apart
            force_x = np.zeros(N)
            force_y = np.zeros(N)
            for (i, j) in pairs:
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2) + 1e-12
                overlap = rs[i] + rs[j] - d
                if overlap > 0:
                    # Shadow price proxy: push proportional to overlap
                    fx = (xs[i] - xs[j]) / d * overlap
                    fy = (ys[i] - ys[j]) / d * overlap
                    force_x[i] += fx
                    force_y[i] += fy
                    force_x[j] -= fx
                    force_y[j] -= fy

            # Boundary forces
            for i in range(N):
                if xs[i] < rs[i]:
                    force_x[i] += (rs[i] - xs[i])
                if xs[i] > w - rs[i]:
                    force_x[i] -= (xs[i] - (w - rs[i]))
                if ys[i] < rs[i]:
                    force_y[i] += (rs[i] - ys[i])
                if ys[i] > h - rs[i]:
                    force_y[i] -= (ys[i] - (h - rs[i]))

            # Update centers with small step
            step = 0.05
            xs += step * force_x
            ys += step * force_y

            # Clamp to bounds
            for i in range(N):
                xs[i] = np.clip(xs[i], 0, w)
                ys[i] = np.clip(ys[i], 0, h)

        return np.column_stack((xs, ys, rs))

    def optimize_from(circles0, w, h, max_iter=400, beta=100.0, ftol=1e-12, gtol=1e-8):
        """L-BFGS-B optimization with softplus penalties."""
        x0 = circles0.flatten()

        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, r_max))

        try:
            res = minimize(
                lambda p: objective_and_grad(p, w, h, beta),
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            return res.x
        except Exception:
            return x0

    def hex_lattice_init(w, h):
        """Generate hexagonal lattice initialization."""
        best_circles = None
        best_val = -np.inf

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
            val = evaluate(circles, w, h)
            if val > best_val:
                best_val = val
                best_circles = circles.copy()

        if best_circles is None:
            a = int(np.ceil(np.sqrt(N * w / h)))
            b = int(np.ceil(N / a))
            while a * b < N:
                b += 1
            circles = []
            for i in range(N):
                col = i % a
                row = i // a
                r = min(w / (a + 0.5), h / (b + 0.5)) * 0.95
                x = (col + 0.5) * w / a
                y = (row + 0.5) * h / b
                circles.append([x, y, r])
            best_circles = np.array(circles)

        return best_circles

    def grid_init(w, h):
        """Simple grid initialization."""
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.95
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        return np.array(circles)

    best_value = -np.inf
    best_result = None

    def check_and_update(circles, w, h):
        nonlocal best_value, best_result
        if check_time():
            return
        val = evaluate(circles, w, h)
        if val > best_value:
            best_value = val
            best_result = circles.copy()

    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.3, 0.9, 1.1]

    for W in aspect_ratios:
        if check_time():
            break
        H = 2.0 - W
        if H <= 0:
            continue

        # Multiple initializations
        inits = [hex_lattice_init(W, H), grid_init(W, H)]

        # Add random init
        for _ in range(2):
            r_rand = np.random.uniform(0.05, 0.15, N)
            x_rand = np.random.uniform(0.1, W - 0.1, N)
            y_rand = np.random.uniform(0.1, H - 0.1, N)
            inits.append(np.column_stack((x_rand, y_rand, r_rand)))

        for circles0 in inits:
            if check_time():
                break

            # Phase 1: L-BFGS-B with softplus penalties
            x_opt = optimize_from(circles0, W, H, max_iter=300, beta=100.0)
            circles_opt = x_opt.reshape(-1, 3)
            check_and_update(circles_opt, W, H)

            # Phase 2: Delaunay-restricted LP refinement
            if not check_time():
                circles_lp = lp_active_set_refine(circles_opt, W, H, max_iters=30)
                check_and_update(circles_lp, W, H)

            # Phase 3: Final polish with higher beta
            if not check_time():
                x_polish = optimize_from(circles_opt, W, H, max_iter=200, beta=200.0, ftol=1e-14, gtol=1e-10)
                circles_polish = x_polish.reshape(-1, 3)
                check_and_update(circles_polish, W, H)

    # Final validation and cleanup
    if best_result is not None:
        # Find the W, H that gave best result - we need to track this
        # Re-validate with the known aspect ratios
        for W in aspect_ratios:
            H = 2.0 - W
            if H <= 0:
                continue
            val = evaluate(best_result, W, H)
            if val > -np.inf:
                # Ensure strict validity
                xs = best_result[:, 0].copy()
                ys = best_result[:, 1].copy()
                rs = best_result[:, 2].copy()

                # Fix boundary violations
                for i in range(N):
                    rs[i] = min(rs[i], xs[i] - 1e-7, W - xs[i] - 1e-7, ys[i] - 1e-7, H - ys[i] - 1e-7)
                    if rs[i] < 1e-8:
                        rs[i] = 1e-8

                # Fix overlaps
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        if d < rs[i] + rs[j]:
                            shrink = (rs[i] + rs[j] - d) / 2.0 + 1e-7
                            rs[i] -= shrink
                            rs[j] -= shrink
                            if rs[i] < 1e-8: rs[i] = 1e-8
                            if rs[j] < 1e-8: rs[j] = 1e-8

                result = np.column_stack((xs, ys, rs))
                val_final = evaluate(result, W, H)
                if val_final > -np.inf:
                    return result

    # Fallback: simple grid
    W = 1.0
    H = 1.0
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
