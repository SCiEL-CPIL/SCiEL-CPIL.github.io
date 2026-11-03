import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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
            a = int(np.ceil(np.sqrt(N)))
            b = int(np.ceil(N / a))
            circles = np.zeros((N, 3))
            for i in range(N):
                col = i % a
                row = i // a
                circles[i, 0] = (col + 0.5) * (w / a)
                circles[i, 1] = (row + 0.5) * (h / b)
                circles[i, 2] = min(w / a, h / b) * 0.4
        return best

    def _random_init(w, h, seed):
        rng = np.random.RandomState(seed)
        r_est = min(w, h) / 10.0
        circles = np.zeros((N, 3))
        circles[:, 0] = rng.uniform(r_est, w - r_est, N)
        circles[:, 1] = rng.uniform(r_est, h - r_est, N)
        circles[:, 2] = r_est
        return circles

    def _optimize(circles, w, h, seed, max_iters=5000):
        """Run L-BFGS-B optimization with softplus penalties."""
        x0 = circles.flatten()
        
        def obj_grad(x):
            return _objective_and_grad(x, w, h)
        
        result = minimize(
            obj_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        return result.x.reshape(-1, 3)

    def _polish(circles, w, h, seed):
        """High-precision polish using contact graph and local moves."""
        if time_left() < 5.0:
            return circles
        
        neighbors = _delaunay_neighbors(circles[:, :2])
        current = circles.copy()
        current_val = _evaluate(current, w, h)
        
        if current_val == -np.inf:
            return current
        
        best_local = current.copy()
        best_local_val = current_val
        
        rng = random.Random(seed)
        
        # Local search: try small perturbations
        for step in range(2000):
            if time_left() < 2.0:
                break
            
            # Pick a random circle and perturb it
            i = rng.randrange(N)
            dx = rng.uniform(-0.01, 0.01)
            dy = rng.uniform(-0.01, 0.01)
            dr = rng.uniform(-0.005, 0.005)
            
            new_circle = current.copy()
            new_circle[i, 0] = np.clip(new_circle[i, 0] + dx, 0, w)
            new_circle[i, 1] = np.clip(new_circle[i, 1] + dy, 0, h)
            new_circle[i, 2] = max(0.001, new_circle[i, 2] + dr)
            
            # Quick validity check using neighbors
            if _fast_valid(new_circle, w, h, neighbors):
                val = _evaluate(new_circle, w, h)
                if val > best_local_val:
                    best_local = new_circle.copy()
                    best_local_val = val
                    current = new_circle.copy()
                    # Update neighbors occasionally
                    if step % 100 == 0:
                        neighbors = _delaunay_neighbors(current[:, :2])
        
        return best_local

    # Main search loop
    aspect_ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 0.9, 0.8, 0.7]
    
    for W in aspect_ratios:
        H = 2.0 - W
        if H < 0.2 or W < 0.2:
            continue
        
        # Try multiple initializations
        for trial in range(8):
            if time_left() < 10.0:
                break
            
            if trial == 0:
                init_circles = _hex_init(W, H, seed=42 + trial)
            elif trial < 4:
                init_circles = _hex_init(W, H, seed=100 + trial)
            else:
                init_circles = _random_init(W, H, seed=200 + trial)
            
            if init_circles is None:
                continue
            
            # Optimize
            opt_circles = _optimize(init_circles, W, H, seed=trial)
            check_and_update(opt_circles, W, H)
            
            # Polish
            if time_left() > 5.0:
                polished = _polish(opt_circles, W, H, seed=300 + trial)
                check_and_update(polished, W, H)
    
    # Fallback if no valid solution found
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x_c = (i + 0.5) * (1.0 / side)
                    y_c = (j + 0.5) * (1.0 / side)
                    r_c = 0.5 / side - 1e-4
                    cells.append([x_c, y_c, r_c])
        return np.array(cells)
    
    return best_result
