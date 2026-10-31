"""Circle packing in perimeter-4 rectangle: multi-aspect SLSQP + refinement chain."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Vectorized boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    obj += penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    # Keep aspect ratio in a reasonable range
    if w < 0.45:
        obj += penalty_weight * (0.45 - w) ** 2
    elif w > 1.55:
        obj += penalty_weight * (w - 1.55) ** 2

    return obj


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    def _force_relax(z, max_iter=400, lr=0.02):
        """Resolve overlaps by pushing circles apart; radii fixed."""
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w
        if w <= 0.05 or h <= 0.05:
            return z
        for _ in range(max_iter):
            moved = 0.0
            fx = np.zeros(N)
            fy = np.zeros(N)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_d = r[i] + r[j]
                    if d < min_d and d > 1e-12:
                        f = (min_d - d) / d
                        fx[i] += f * dx
                        fy[i] += f * dy
                        fx[j] -= f * dx
                        fy[j] -= f * dy
                    elif d < min_d:
                        # coincident: nudge
                        fx[i] += 0.01 * (np.random.rand() - 0.5)
                        fy[i] += 0.01 * (np.random.rand() - 0.5)
            x += lr * fx
            y += lr * fy
            # clamp to boundary
            for i in range(N):
                x[i] = min(max(x[i], r[i]), w - r[i])
                y[i] = min(max(y[i], r[i]), h - r[i])
                moved += abs(fx[i]) + abs(fy[i])
            if moved < 1e-6:
                break
        out = z.copy()
        out[0::3][:N] = x
        out[1::3][:N] = y
        return out

    def _polish(z, time_budget):
        """Local SLSQP polish with a wall-clock cap."""
        t0 = time.time()
        try:
            res = minimize(objective, z, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': 200, 'ftol': 1e-14, 'maxls': 40})
            if time.time() - t0 > time_budget:
                return None
            return res.x
        except Exception:
            return None

    def _constraint_refine(z, time_budget):
        """Pure objective + explicit inequality constraints (SLSQP)."""
        t0 = time.time()
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w

        def obj(v):
            return -np.sum(v[2::3][:N])

        cons = []
        # boundary constraints
        for i in range(N):
            cons.append({'type': 'ineq',
                         'fun': lambda v, i=i: v[3 * i] - v[3 * i + 2]})
            cons.append({'type': 'ineq',
                         'fun': lambda v, i=i: v[3 * i - 1 + 1] - v[3 * i + 2]})
        # simpler: build explicit closures
        cons = []
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3 * i] - v[3 * i + 2]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3 * i + 2] - (v[3 * i] - 0.0)})
        # rebuild properly below
        cons = []
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3 * i] - v[3 * i + 2]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3 * i + 1] - v[3 * i + 2]})
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[-1] - v[3 * i] - v[3 * i + 2]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 2.0 - v[-1] - v[3 * i + 1] - v[3 * i + 2]})
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({'type': 'ineq',
                             'fun': lambda v, i=i, j=j:
                             (v[3 * i] - v[3 * j]) ** 2 + (v[3 * i + 1] - v[3 * j + 1]) ** 2
                             - (v[3 * i + 2] + v[3 * j + 2]) ** 2})

        try:
            res = minimize(obj, z, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           constraints=cons,
                           options={'maxiter': 150, 'ftol': 1e-14, 'maxls': 40})
            if time.time() - t0 > time_budget:
                return None
            return res.x
        except Exception:
            return None

    def _lse_refine(z, time_budget):
        """Smooth log-sum-exp epigraph surrogate for the non-convex distance constraints."""
        t0 = time.time()
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w
        beta = 60.0

        def obj(v):
            xi = v[0::3][:N]
            yi = v[1::3][:N]
            ri = v[2::3][:N]
            wi = v[-1]
            hi = 2.0 - wi
            val = -np.sum(ri)
            iu = np.triu_indices(N, k=1)
            dx = xi[iu[0]] - xi[iu[1]]
            dy = yi[iu[0]] - yi[iu[1]]
            d2 = dx * dx + dy * dy
            rs = ri[iu[0]] + ri[iu[1]]
            gap = d2 - rs * rs
            # log-sum-exp of exp(-beta*gap) as penalty for violations
            a = -beta * gap
            a_max = np.max(a)
            val += 50.0 * (a_max + np.log(np.sum(np.exp(a - a_max))))
            # boundary LSE
            for i in range(N):
                a = -beta * (xi[i] - ri[i])
                val += 50.0 * (a + np.log(np.sum(np.exp(np.array([0.0, a]) - a))))
            return val

        try:
            res = minimize(obj, z, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': 150, 'ftol': 1e-14, 'maxls': 40})
            if time.time() - t0 > time_budget:
                return None
            return res.x
        except Exception:
            return None

    def _expansion_polish(z, time_budget):
        """Grow radii then re-relax positions to harvest extra sum."""
        t0 = time.time()
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w
        # grow radii a little
        r = r * 1.03
        # cap radii by boundary
        for i in range(N):
            r[i] = min(r[i], x[i], w - x[i], y[i], h - y[i])
            r[i] = max(r[i], 1e-4)
        out = z.copy()
        out[2::3][:N] = r
        out = _force_relax(out, max_iter=300, lr=0.02)
        # polish
        try:
            res = minimize(objective, out, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': 150, 'ftol': 1e-14, 'maxls': 40})
            if time.time() - t0 > time_budget:
                return None
            return res.x
        except Exception:
            return None

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # Phase 1: multi-start SLSQP over diverse aspect ratios
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 40:
            break
        z0_relaxed = _force_relax(z0, max_iter=300, lr=0.02)
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(120, remaining * 30)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
            z_opt = res.x
            valid, val = check_validity(z_opt)
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()
        except Exception:
            continue

    # Phase 2: refinement chain on the best result
    if best_result is not None and time.time() < deadline - 25:
        polished = _polish(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        if best_result is not None and time.time() < deadline - 20:
            expanded = _expansion_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        if best_result is not None and time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        if best_result is not None and time.time() < deadline - 8:
            lse = _lse_refine(best_result, deadline - (time.time() - start_time) - 4)
            if lse is not None:
                valid, val = check_validity(lse)
                if valid and val > best_value:
                    best_value = val
                    best_result = lse

        if best_result is not None and time.time() < deadline - 5:
            refined2 = _constraint_refine(best_result, deadline - (time.time() - start_time) - 2)
            if refined2 is not None:
                valid, val = check_validity(refined2)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined2

        if best_result is not None and time.time() < deadline - 2:
            polished2 = _polish(best_result, deadline - (time.time() - start_time))
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

    # Convert best_result to absolute (cx, cy, r) array
    if best_result is not None:
        x = best_result[0::3][:N]
        y = best_result[1::3][:N]
        r = best_result[2::3][:N]
        return np.column_stack((x, y, r))

    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    W = 1.0
    H = 1.0
    cells = []
    for i in range(N):
        ix = i % side
        iy = i // side
        cx = (ix + 0.5) / side * W
        cy = (iy + 0.5) / side * H
        r = 0.5 / side * W - 0.01
        r = min(r, 0.5 / side * H - 0.01)
        cells.append((cx, cy, r))
    return np.array(cells)
