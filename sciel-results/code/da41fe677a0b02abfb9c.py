"""Circle packing in perimeter-4 rectangle: smooth epigraph SLSQP + geometric polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 120.0

def _dist(x, y):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.sqrt(dx * dx + dy * dy)

def _softmin(a, k):
    """Smooth approximation of min(a) via -logsumexp(-a)/k."""
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return 0.0
    m = a.min()
    s = np.sum(np.exp(-(a - m) * k))
    return m - np.log(s) / k

def _build_objective(K):
    def objective(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[3 * N]
        h = z[3 * N + 1]
        if w < 0.25 or h < 0.25 or np.any(r < 0):
            return 1e6
        d = _dist(x, y)
        rsum = r[:, None] + r[None, :]
        triu = np.triu_indices(N, k=1)
        pair = d[triu] - rsum[triu]
        t_pair = _softmin(pair, K)
        gap = np.concatenate([
            x - r, w - x - r, y - r, h - y - r
        ])
        t_wall = _softmin(gap, K)
        return -np.sum(r) - t_pair - t_wall
    return objective

def _constraints():
    def c_perim(z):
        return z[3 * N] + z[3 * N + 1] - 4.0
    def c_wmin(z):
        return z[3 * N] - 0.3
    def c_hmin(z):
        return z[3 * N + 1] - 0.3
    return [
        {'type': 'eq', 'fun': c_perim},
        {'type': 'ineq', 'fun': c_wmin},
        {'type': 'ineq', 'fun': c_hmin},
    ]

def _check_validity(z, tol=1e-9):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[3 * N]
    h = z[3 * N + 1]
    if w <= 1e-6 or h <= 1e-6 or np.any(r <= 0):
        return False, 0.0
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False, 0.0
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False, 0.0
    d = _dist(x, y)
    rsum = r[:, None] + r[None, :]
    triu = np.triu_indices(N, k=1)
    if np.any(d[triu] < rsum[triu] - tol):
        return False, 0.0
    return True, float(np.sum(r))

def _force_relax(z, max_iter=200, dt=0.004):
    z = z.copy().astype(float)
    n = N
    for _ in range(max_iter):
        if time.time() > DEADLINE - 4:
            break
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[3 * N]
        h = z[3 * N + 1]
        fx = np.zeros(n)
        fy = np.zeros(n)
        for i in range(n):
            if x[i] - r[i] < 0: fx[i] += (x[i] - r[i]) * 20
            if x[i] + r[i] > w: fx[i] -= (x[i] + r[i] - w) * 20
            if y[i] - r[i] < 0: fy[i] += (y[i] - r[i]) * 20
            if y[i] + r[i] > h: fy[i] -= (y[i] + r[i] - h) * 20
        d = _dist(x, y)
        rsum = r[:, None] + r[None, :]
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < rsum[i, j] and d[i, j] > 1e-12:
                    ov = rsum[i, j] - d[i, j]
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    fx[i] += (dx / d[i, j]) * ov * 10
                    fy[i] += (dy / d[i, j]) * ov * 10
                    fx[j] -= (dx / d[i, j]) * ov * 10
                    fy[j] -= (dy / d[i, j]) * ov * 10
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])
    return z

def _polish(z, budget):
    """Grow each radius to its exact geometric limit, iterate."""
    z = z.copy().astype(float)
    n = N
    start = time.time()
    for _ in range(300):
        if time.time() - start > budget:
            break
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[3 * N]
        h = z[3 * N + 1]
        d = _dist(x, y)
        rsum = r[:, None] + r[None, :]
        changed = False
        for i in range(n):
            g = min(x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i])
            for j in range(n):
                if i != j:
                    g = min(g, d[i, j] - r[i] - r[j])
            if g > 1e-9:
                r[i] += 0.6 * g
                changed = True
        if not changed:
            break
        # re-project positions to keep them feasible at new radii
        for i in range(n):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - y[i])
        # light separation to fix any induced overlaps
        for _ in range(3):
            fx = np.zeros(n)
            fy = np.zeros(n)
            for i in range(n):
                if x[i] - r[i] < 0: fx[i] += (x[i] - r[i]) * 30
                if x[i] + r[i] > w: fx[i] -= (x[i] + r[i] - w) * 30
                if y[i] - r[i] < 0: fy[i] += (y[i] - r[i]) * 30
                if y[i] + r[i] > h: fy[i] -= (y[i] + r[i] - h) * 30
            d2 = _dist(x, y)
            rs2 = r[:, None] + r[None, :]
            for i in range(n):
                for j in range(i + 1, n):
                    if d2[i, j] < rs2[i, j] and d2[i, j] > 1e-12:
                        ov = rs2[i, j] - d2[i, j]
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        fx[i] += (dx / d2[i, j]) * ov * 12
                        fy[i] += (dy / d2[i, j]) * ov * 12
                        fx[j] -= (dx / d2[i, j]) * ov * 12
                        fy[j] -= (dy / d2[i, j]) * ov * 12
            x += 0.0006 * fx
            y += 0.0006 * fy
            for i in range(n):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])
        z[0::3][:n] = x
        z[1::3][:n] = y
        z[2::3][:n] = r
    return z

def _grid_seed(w0, h0, rng):
    side = int(np.ceil(np.sqrt(N)))
    z = np.zeros(3 * N + 2)
    r0 = min(w0 / side, h0 / side) * 0.45
    for i in range(N):
        row, col = divmod(i, side)
        z[3 * i] = (col + 0.5) / side * w0
        z[3 * i + 1] = (row + 0.5) / side * h0
        z[3 * i + 2] = r0
    z[3 * N] = w0
    z[3 * N + 1] = h0
    return z

def _random_seed(w0, h0, rng):
    z = np.zeros(3 * N + 2)
    for i in range(N):
        z[3 * i] = rng.uniform(0.05, w0 - 0.05)
        z[3 * i + 1] = rng.uniform(0.05, h0 - 0.05)
        z[3 * i + 2] = rng.uniform(0.04, 0.10)
    z[3 * N] = w0
    z[3 * N + 1] = h0
    return z

def construct():
    best_value = 0.0
    best_result = None
    rng = np.random.default_rng(7)

    seeds = []
    for w0 in [1.0, 0.9, 1.1, 0.8, 1.2, 0.75, 1.25, 1.33, 0.67]:
        h0 = 4.0 - 2.0 * w0
        if h0 < 0.35:
            continue
        seeds.append(_grid_seed(w0, h0, rng))
    for _ in range(4):
        w0 = rng.uniform(0.7, 1.3)
        h0 = 4.0 - 2.0 * w0
        seeds.append(_random_seed(w0, h0, rng))
    rng.shuffle(seeds)

    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7), (0.3, 1.7)]
    cons = _constraints()
    objective = _build_objective(8.0)

    for i, z0 in enumerate(seeds):
        elapsed = time.time() - START
        if elapsed > DEADLINE - 15:
            break
        remaining = DEADLINE - elapsed - 8
        if remaining < 3:
            break

        z0r = _force_relax(z0, max_iter=120)
        maxiter = int(min(400, remaining * 40))
        if maxiter < 20:
            maxiter = 20
        try:
            res = minimize(objective, z0r, method='SLSQP', bounds=bounds,
                           constraints=cons,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 40})
            z_opt = res.x
            valid, val = _check_validity(z_opt)
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                zr = z_opt.copy()
                zr[2::3][:N] *= 0.92
                valid2, val2 = _check_validity(zr)
                if valid2 and val2 > best_value:
                    best_value = val2
                    best_result = zr.copy()
        except Exception:
            continue

    # Polish best result with geometric growth
    if best_result is not None:
        remaining = DEADLINE - (time.time() - START) - 6
        if remaining > 2:
            polished = _polish(best_result, remaining)
            valid, val = _check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    # Final tight SLSQP pass
    if best_result is not None:
        remaining = DEADLINE - (time.time() - START) - 4
        if remaining > 2:
            try:
                res = minimize(objective, best_result, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': 150, 'ftol': 1e-14, 'maxls': 30})
                z_opt = res.x
                valid, val = _check_validity(z_opt)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_opt.copy()
            except Exception:
                pass

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N] * (1 - 1e-7)
    return np.column_stack((x, y, r))
