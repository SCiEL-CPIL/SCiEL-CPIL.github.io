"""Circle packing in a perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize
import random

N = 21
DEADLINE = 115.0  # Safety margin below 120s limit

def _is_valid(circles, w, h, tol=1e-5):
    n = len(circles)
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]

    if np.any(rs < -tol):
        return False
    if np.any(xs < -rs - tol) or np.any(xs > w + rs + tol):
        return False
    if np.any(ys < -rs - tol) or np.any(ys > h + rs + tol):
        return False

    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx*dx + dy*dy < (rs[i] + rs[j])**2 - tol:
                return False
    return True

def _penalty_objective(x, mu, w_init, h_init):
    """
    x layout: [W, x1..xn, y1..yn, r1..rn]  (1 + 3*N variables)
    Returns -sum(r_i) + penalty terms.
    """
    W = x[0]
    H = 2.0 - W
    xs = x[1:1+N]
    ys = x[1+N:1+2*N]
    rs = x[1+2*N:1+3*N]

    # Objective: maximize sum of radii => minimize -sum(r)
    obj = -np.sum(rs)

    # Penalty for boundary violations
    pen = 0.0
    # x_i >= r_i
    v1 = rs - xs
    pen += np.sum(np.maximum(v1, 0.0)**2)
    # x_i <= W - r_i
    v2 = xs + rs - W
    pen += np.sum(np.maximum(v2, 0.0)**2)
    # y_i >= r_i
    v3 = rs - ys
    pen += np.sum(np.maximum(v3, 0.0)**2)
    # y_i <= H - r_i
    v4 = ys + rs - H
    pen += np.sum(np.maximum(v4, 0.0)**2)

    # Penalty for overlap
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d2 = dx*dx + dy*dy
            rsum = rs[i] + rs[j]
            if d2 < rsum*rsum:
                pen += (rsum*rsum - d2)**2

    # Penalty for W out of reasonable range
    if W < 0.3 or W > 1.7:
        pen += 10.0 * (W - 1.0)**2

    return obj + mu * pen

def _penalty_grad(x, mu, w_init, h_init):
    """
    Gradient of penalty objective.
    """
    W = x[0]
    H = 2.0 - W
    xs = x[1:1+N]
    ys = x[1+N:1+2*N]
    rs = x[1+2*N:1+3*N]

    grad = np.zeros(1 + 3*N)

    # d(-sum(r))/dW = 0
    grad[0] = 0.0

    # Boundary penalties
    # v1 = r_i - x_i >= 0: penalty = max(v1,0)^2
    # d/dx_i = -2*max(v1,0), d/d(r_i) = 2*max(v1,0)
    v1 = np.maximum(rs - xs, 0.0)
    grad[1:1+N] -= 2.0 * v1
    grad[1+2*N:1+3*N] += 2.0 * v1

    # v2 = x_i + r_i - W >= 0: penalty = max(v2,0)^2
    # d/dW = -2*max(v2,0), d/dx_i = 2*max(v2,0), d/d(r_i) = 2*max(v2,0)
    v2 = np.maximum(xs + rs - W, 0.0)
    grad[0] -= 2.0 * np.sum(v2)
    grad[1:1+N] += 2.0 * v2
    grad[1+2*N:1+3*N] += 2.0 * v2

    # v3 = r_i - y_i >= 0
    v3 = np.maximum(rs - ys, 0.0)
    grad[1+N:1+2*N] -= 2.0 * v3
    grad[1+2*N:1+3*N] += 2.0 * v3

    # v4 = y_i + r_i - H >= 0, H = 2-W, so dH/dW = -1
    # v4 = y_i + r_i - (2-W) = y_i + r_i + W - 2
    # d/dW = 2*max(v4,0), d/dy_i = 2*max(v4,0), d/d(r_i) = 2*max(v4,0)
    v4 = np.maximum(ys + rs + W - 2.0, 0.0)
    grad[0] += 2.0 * np.sum(v4)
    grad[1+N:1+2*N] += 2.0 * v4
    grad[1+2*N:1+3*N] += 2.0 * v4

    # Overlap penalties
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d2 = dx*dx + dy*dy
            rsum = rs[i] + rs[j]
            if d2 < rsum*rsum:
                # penalty = (rsum^2 - d2)^2
                # d/d(x_i) = 2*(rsum^2-d2)*(-2*dx) = -4*(rsum^2-d2)*dx
                # d/d(x_j) = 2*(rsum^2-d2)*(2*dx) = 4*(rsum^2-d2)*dx
                # d/d(r_i) = 2*(rsum^2-d2)*2*rsum = 4*(rsum^2-d2)*rsum
                # d/d(r_j) = same
                diff = rsum*rsum - d2
                grad[1+i] += -4.0 * diff * dx
                grad[1+j] += 4.0 * diff * dx
                grad[1+N+i] += -4.0 * diff * dy
                grad[1+N+j] += 4.0 * diff * dy
                grad[1+2*N+i] += 4.0 * diff * rsum
                grad[1+2*N+j] += 4.0 * diff * rsum

    # W range penalty
    if W < 0.3 or W > 1.7:
        grad[0] += 20.0 * (W - 1.0)

    return mu * grad

def _extract_circles(x, W):
    """Extract circles from optimization vector."""
    H = 2.0 - W
    xs = x[1:1+N]
    ys = x[1+N:1+2*N]
    rs = x[1+2*N:1+3*N]

    circles = np.column_stack([xs, ys, rs])

    # Clip to valid bounds
    circles[:, 0] = np.clip(circles[:, 0], 0, W)
    circles[:, 1] = np.clip(circles[:, 1], 0, H)
    circles[:, 2] = np.maximum(0.0, circles[:, 2])

    # Ensure boundary feasibility
    for i in range(N):
        max_r = min(circles[i, 0], circles[i, 1], W - circles[i, 0], H - circles[i, 1])
        circles[i, 2] = min(circles[i, 2], max_r)

    return circles, W, H

def _grid_init(W, H, n_cols, n_rows):
    xs = (np.arange(n_cols) + 0.5) * (W / n_cols)
    ys = (np.arange(n_rows) + 0.5) * (H / n_rows)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X = X.flatten()
    Y = Y.flatten()
    X = X[:N]
    Y = Y[:N]
    r0 = min(W / n_cols, H / n_rows) / 2.0 * 0.9
    R = np.full(N, r0)
    return np.concatenate([[W], X, Y, R])

def _random_init(W, H):
    xs = np.random.uniform(0.05*W, 0.95*W, N)
    ys = np.random.uniform(0.05*H, 0.95*H, N)
    rs = np.full(N, 0.08)
    return np.concatenate([[W], xs, ys, rs])

def _force_init(W, H, iterations=80):
    """Simple force-directed initialization to spread circles."""
    xs = np.random.uniform(0.05*W, 0.95*W, N)
    ys = np.random.uniform(0.05*H, 0.95*H, N)
    rs = np.full(N, 0.08)

    for _ in range(iterations):
        for i in range(N):
            fx = 0.0
            fy = 0.0
            for j in range(N):
                if i == j:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-5:
                    fx += np.random.uniform(-1, 1)
                    fy += np.random.uniform(-1, 1)
                    continue
                f = 1.0 / (d * d + 1e-4)
                fx += f * dx
                fy += f * dy

            # Attraction to center
            fx += 0.02 * (W/2 - xs[i])
            fy += 0.02 * (H/2 - ys[i])

            # Boundary repulsion
            if xs[i] < rs[i]: fx -= 2.0
            if xs[i] > W - rs[i]: fx += 2.0
            if ys[i] < rs[i]: fy -= 2.0
            if ys[i] > H - rs[i]: fy += 2.0

            xs[i] += 0.03 * fx
            ys[i] += 0.03 * fy

            xs[i] = np.clip(xs[i], rs[i], W - rs[i])
            ys[i] = np.clip(ys[i], rs[i], H - rs[i])

    return np.concatenate([[W], xs, ys, rs])

def construct():
    t0 = time.time()
    best_circles = None
    best_val = -np.inf
    best_W = 1.0
    best_H = 1.0

    # Initial W values to try (width of rectangle, H = 2 - W)
    W_starts = [1.0, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.9, 1.1, 0.75]

    for W0 in W_starts:
        if time.time() - t0 > DEADLINE:
            break

        H0 = 2.0 - W0

        # Generate initializations
        inits = []

        # Grids
        grid_configs = [(5, 5), (6, 4), (4, 6), (7, 4), (4, 7), (5, 4)]
        for nc, nr in grid_configs:
            if nc * nr >= N:
                inits.append(_grid_init(W0, H0, nc, nr))

        # Random
        inits.append(_random_init(W0, H0))
        inits.append(_random_init(W0, H0))

        # Force directed
        inits.append(_force_init(W0, H0))

        for x0 in inits:
            if time.time() - t0 > DEADLINE:
                break

            # Multi-stage penalty optimization
            try:
                x_current = x0.copy()
                mu = 1.0

                # Stage 1: moderate penalty, find feasible region
                for stage in range(3):
                    if time.time() - t0 > DEADLINE:
                        break

                    # Increase penalty each stage
                    mu = mu * 5.0  # 5, 25, 125

                    res = minimize(
                        _penalty_objective, x_current,
                        args=(mu, W0, H0),
                        method='L-BFGS-B',
                        jac=_penalty_grad,
                        bounds=[(0.3, 1.7)] +  # W
                               [(0, W0+0.5)] * N +  # x positions
                               [(0, H0+0.5)] * N +  # y positions
                               [(0, 1.0)] * N,       # radii
                        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': 0}
                    )

                    x_current = res.x

                    # Check if we're close to feasible
                    W_cur = x_current[0]
                    H_cur = 2.0 - W_cur
                    circles, _, _ = _extract_circles(x_current, W_cur)

                    if _is_valid(circles, W_cur, H_cur):
                        val = np.sum(circles[:, 2])
                        if val > best_val:
                            best_val = val
                            best_circles = circles.copy()
                            best_W = W_cur
                            best_H = H_cur

                # Final polish with high penalty
                if time.time() - t0 < DEADLINE:
                    mu_final = 500.0
                    res = minimize(
                        _penalty_objective, x_current,
                        args=(mu_final, W0, H0),
                        method='L-BFGS-B',
                        jac=_penalty_grad,
                        bounds=[(0.3, 1.7)] +
                               [(0, 2.0)] * N +
                               [(0, 2.0)] * N +
                               [(0, 1.0)] * N,
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': 0}
                    )
                    x_current = res.x
                    W_cur = x_current[0]
                    H_cur = 2.0 - W_cur
                    circles, _, _ = _extract_circles(x_current, W_cur)

                    if _is_valid(circles, W_cur, H_cur):
                        val = np.sum(circles[:, 2])
                        if val > best_val:
                            best_val = val
                            best_circles = circles.copy()
                            best_W = W_cur
                            best_H = H_cur

            except Exception:
                continue

    # Fallback: simple grid
    if best_circles is None:
        W = H = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])

    return best_circles
