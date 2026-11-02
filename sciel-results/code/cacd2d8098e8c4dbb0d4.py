import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    best_val = -np.inf
    best_x = None
    start_time = time.time()
    time_limit = 130  # Safety margin before 150s kill

    # Precompute pair indices for vectorized overlap
    pair_i, pair_j = np.triu_indices(N, k=1)
    n_pairs = len(pair_i)

    def objective_and_grad(x):
        w = x[0]
        h = x[1]
        xs = x[2:2+N]
        ys = x[2+N:2+2*N]
        rs = x[2+2*N:2+3*N]

        sum_r = np.sum(rs)

        # Perimeter penalty (smooth)
        perim_viol = 2.0 * (w + h) - 4.0
        pen_perim = perim_viol * perim_viol if perim_viol > 0 else 0.0

        # Vectorized overlap: dist - (r_i + r_j)
        dx = xs[pair_i] - xs[pair_j]
        dy = ys[pair_i] - ys[pair_j]
        dist = np.sqrt(dx * dx + dy * dy)
        viol = rs[pair_i] + rs[pair_j] - dist

        # Smooth penalty: use softplus-like: for viol>0, viol^2; else 0
        # Use np.where for efficiency
        pen_overlap = np.sum(np.maximum(viol, 0.0) ** 2)

        # Boundary penalties (vectorized)
        v1 = np.maximum(rs - xs, 0.0)
        v2 = np.maximum(xs + rs - w, 0.0)
        v3 = np.maximum(rs - ys, 0.0)
        v4 = np.maximum(ys + rs - h, 0.0)
        pen_bound = np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2) + np.sum(v4**2)

        val = -sum_r + 100.0 * pen_perim + 100.0 * pen_overlap + 100.0 * pen_bound

        # Gradient
        grad = np.zeros_like(x)

        # w, h from perimeter
        if perim_viol > 0:
            grad[0] = 100.0 * 2.0 * perim_viol * 2.0
            grad[1] = 100.0 * 2.0 * perim_viol * 2.0

        # Overlap gradient (vectorized)
        mask = viol > 0.0
        if np.any(mask):
            dm = mask
            dxm = dx[dm]
            dym = dy[dm]
            distm = dist[dm]
            violm = viol[dm]
            im = pair_i[dm]
            jm = pair_j[dm]

            # Avoid division by zero
            safe_dist = np.where(distm > 1e-10, distm, 1.0)
            gxi = -dxm / safe_dist
            gxi = np.where(distm > 1e-10, gxi, 0.0)
            gxi_j = -gxi  # opposite sign for j
            gyi = -dym / safe_dist
            gyi = np.where(distm > 1e-10, gyi, 0.0)
            gyi_j = -gyi

            coef = 100.0 * 2.0 * violm

            # Accumulate gradients using np.add.at for repeated indices
            np.add.at(grad, 2 + im, coef * gxi)
            np.add.at(grad, 2 + jm, coef * gxi_j)
            np.add.at(grad, 2 + N + im, coef * gyi)
            np.add.at(grad, 2 + N + jm, coef * gyi_j)
            np.add.at(grad, 2 + 2*N + im, coef)
            np.add.at(grad, 2 + 2*N + jm, coef)

        # Boundary gradients
        mask1 = rs - xs > 0.0
        if np.any(mask1):
            c1 = 100.0 * 2.0 * v1[mask1]
            idx = np.where(mask1)[0]
            grad[2 + idx] -= c1
            grad[2 + 2*N + idx] += c1

        mask2 = xs + rs - w > 0.0
        if np.any(mask2):
            c2 = 100.0 * 2.0 * v2[mask2]
            idx = np.where(mask2)[0]
            grad[2 + idx] += c2
            grad[2 + 2*N + idx] += c2
            grad[0] -= np.sum(c2)

        mask3 = rs - ys > 0.0
        if np.any(mask3):
            c3 = 100.0 * 2.0 * v3[mask3]
            idx = np.where(mask3)[0]
            grad[2 + N + idx] -= c3
            grad[2 + 2*N + idx] += c3

        mask4 = ys + rs - h > 0.0
        if np.any(mask4):
            c4 = 100.0 * 2.0 * v4[mask4]
            idx = np.where(mask4)[0]
            grad[2 + N + idx] += c4
            grad[2 + 2*N + idx] += c4
            grad[1] -= np.sum(c4)

        # Sum of radii gradient
        grad[2+2*N:2+3*N] -= 1.0

        return val, grad

    bounds = []
    bounds.append((0.1, 1.9))
    bounds.append((0.1, 1.9))
    for i in range(N):
        bounds.append((0, 2))
    for i in range(N):
        bounds.append((0, 2))
    for i in range(N):
        bounds.append((0.001, 1.0))

    def check_validity(x_opt):
        w, h = x_opt[0], x_opt[1]
        xs = x_opt[2:2+N]
        ys = x_opt[2+N:2+2*N]
        rs = x_opt[2+2*N:2+3*N]
        if 2*(w+h) > 4 + 1e-6:
            return False, 0.0
        for i in range(N):
            if xs[i] < -1e-6 or xs[i] > w + 1e-6:
                return False, 0.0
            if ys[i] < -1e-6 or ys[i] > h + 1e-6:
                return False, 0.0
            if rs[i] < 1e-6:
                return False, 0.0
        # Vectorized overlap check
        dx = xs[pair_i] - xs[pair_j]
        dy = ys[pair_i] - ys[pair_j]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(dist < rs[pair_i] + rs[pair_j] - 1e-6):
            return False, 0.0
        return True, np.sum(rs)

    # Multiple restarts with varied initializations
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break

        np.random.seed(42 + restart * 7)

        if restart < 5:
            # Grid-like init: 5x5 grid in unit square
            side = 5
            w0 = 1.0
            h0 = 1.0
            xs0 = np.zeros(N)
            ys0 = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                xs0[i] = (col + 0.5) / side
                ys0[i] = (row + 0.5) / side
            rs0 = np.full(N, 0.5/side - 0.01)
        elif restart < 10:
            # Hexagonal-like init
            w0 = 1.0
            h0 = 1.0
            rows = 5
            cols = 5
            xs0 = np.zeros(N)
            ys0 = np.zeros(N)
            for i in range(N):
                row = i // cols
                col = i % cols
                offset = 0.5 / cols if row % 2 == 1 else 0.0
                xs0[i] = (col + 0.5 + offset) / cols
                ys0[i] = (row + 0.5) / rows
            rs0 = np.full(N, 0.45 / cols)
        elif restart < 15:
            # Random positions, equal radii
            w0 = np.random.uniform(0.8, 1.2)
            h0 = 2.0 - w0
            xs0 = np.random.uniform(0.1, w0 - 0.1, N)
            ys0 = np.random.uniform(0.1, h0 - 0.1, N)
            rs0 = np.full(N, np.random.uniform(0.08, 0.12))
        else:
            # Fully random
            w0 = np.random.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            xs0 = np.random.uniform(0.05, w0 - 0.05, N)
            ys0 = np.random.uniform(0.05, h0 - 0.05, N)
            rs0 = np.random.uniform(0.05, 0.15, N)

        x0 = np.concatenate(([w0, h0], xs0, ys0, rs0))

        try:
            res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10})
            x_opt = res.x

            valid, val = check_validity(x_opt)
            if valid and val > best_val:
                best_val = val
                best_x = x_opt
        except Exception:
            pass

    # Final high-precision polish on best solution
    if best_x is not None and time.time() - start_time < time_limit - 10:
        try:
            res_polish = minimize(objective_and_grad, best_x, jac=True, method='L-BFGS-B',
                                  bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            x_polished = res_polish.x
            valid, val_polished = check_validity(x_polished)
            if valid and val_polished > best_val:
                best_val = val_polished
                best_x = x_polished
        except Exception:
            pass

    if best_x is not None:
        w, h = best_x[0], best_x[1]
        xs = best_x[2:2+N]
        ys = best_x[2+N:2+2*N]
        rs = best_x[2+2*N:2+3*N]
        rs = np.maximum(rs, 1e-6)
        # Clip centers to be within bounds
        xs = np.clip(xs, 0, w)
        ys = np.clip(ys, 0, h)
        result = np.column_stack((xs, ys, rs))
        return result
    else:
        # Fallback: 5x5 grid
        side = 5
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            cells.append(((col + 0.5) / side, (row + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)
