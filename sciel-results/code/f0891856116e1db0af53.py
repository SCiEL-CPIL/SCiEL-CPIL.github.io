"""Circle packing in perimeter-4 rectangle: radii-first SLSQP + grid-seed scan + refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_DEADLINE = 140.0  # safety margin before 150s kill


def _grid_seed(w, h):
    """Generate the best feasible grid seed for a w x h rectangle.
    
    Returns x in layout [r_0..r_{N-1}, cx_0..cx_{N-1}, cy_0..cy_{N-1}].
    Scans column counts to find the grid layout that maximizes sum of radii
    while staying feasible.
    """
    best_seed = None
    best_score = -1e9

    for cols in range(1, N + 1):
        rows = (N + cols - 1) // cols
        if cols * rows < N:
            continue
        cw = w / cols
        ch = h / rows
        r = 0.5 * min(cw, ch) * 0.999
        if r < 1e-4:
            continue

        x = np.zeros(3 * N)
        idx = 0
        for j in range(rows):
            for i in range(cols):
                if idx >= N:
                    break
                cx = (i + 0.5) * cw
                cy = (j + 0.5) * ch
                x[idx] = r
                x[N + idx] = cx
                x[2 * N + idx] = cy
                idx += 1

        ok = True
        for i in range(N):
            ri = x[i]
            cx = x[N + i]
            cy = x[2 * N + i]
            if cx < ri - 1e-9 or cx > w - ri + 1e-9 or cy < ri - 1e-9 or cy > h - ri + 1e-9:
                ok = False
                break
        if ok:
            for i in range(N):
                for j in range(i + 1, N):
                    d2 = (x[N + i] - x[N + j]) ** 2 + (x[2 * N + i] - x[2 * N + j]) ** 2
                    if d2 < (x[i] + x[j]) ** 2 - 1e-9:
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            score = np.sum(x[:N])
            if score > best_score:
                best_score = score
                best_seed = x.copy()

    if best_seed is None:
        best_seed = np.zeros(3 * N)
        best_seed[:N] = 1e-4
        xs = np.linspace(1e-4, w - 1e-4, N)
        best_seed[N:2 * N] = xs
        best_seed[2 * N:3 * N] = h / 2.0
    return best_seed


def _is_feasible(x, w, tol=1e-6):
    h = 2.0 - w
    r = x[:N]
    cx = x[N:2 * N]
    cy = x[2 * N:3 * N]
    if np.any(r < -tol):
        return False
    if np.any(cx < r - tol) or np.any(cx > w - r + tol) or np.any(cy < r - tol) or np.any(cy > h - r + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2
            if d2 < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _sum_radii(x):
    return float(np.sum(x[:N]))


def _force_directed_relax(x, w, h, iterations=100, rng=None):
    """Force-directed relaxation in the [r, cx, cy] parameterization."""
    if rng is None:
        rng = np.random.default_rng(42)

    x = x.copy()
    r = x[:N].copy()
    cx = x[N:2 * N].copy()
    cy = x[2 * N:3 * N].copy()

    for it in range(iterations):
        lr = 0.1 * (1.0 - it / iterations) + 0.01
        forces = np.zeros((N, 2))

        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-10:
                    overlap = min_dist - dist
                    fx = (dx / dist) * overlap * 5.0
                    fy = (dy / dist) * overlap * 5.0
                    forces[i] += [fx, fy]
                    forces[j] -= [fx, fy]

        for i in range(N):
            ri = r[i]
            if cx[i] - ri < 0:
                forces[i, 0] += (ri - cx[i]) * 3.0
            if cx[i] + ri > w:
                forces[i, 0] -= (cx[i] + ri - w) * 3.0
            if cy[i] - ri < 0:
                forces[i, 1] += (ri - cy[i]) * 3.0
            if cy[i] + ri > h:
                forces[i, 1] -= (cy[i] + ri - h) * 3.0

        cx += forces[:, 0] * lr
        cy += forces[:, 1] * lr
        cx = np.clip(cx, r * 0.5, w - r * 0.5)
        cy = np.clip(cy, r * 0.5, h - r * 0.5)

    cx = np.clip(cx, r, w - r)
    cy = np.clip(cy, r, h - r)

    x[N:2 * N] = cx
    x[2 * N:3 * N] = cy
    return x


def _make_constraints(w, h):
    """Build SLSQP inequality constraints for the [r, cx, cy] layout."""
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - 1e-4})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N + i] - x[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (w - x[N + i]) - x[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * N + i] - x[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (h - x[2 * N + i]) - x[i]})
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (
                    (x[N + i] - x[N + j]) ** 2 +
                    (x[2 * N + i] - x[2 * N + j]) ** 2 -
                    (x[i] + x[j]) ** 2
                ),
            })
    return cons


def _slsqp_optimize(x_init, w, h, time_budget):
    """Run SLSQP on the [r, cx, cy] parameterization."""
    start_time = time.time()

    def objective(x):
        return -np.sum(x[:N])

    cons = _make_constraints(w, h)
    bounds = []
    for i in range(N):
        bounds.append((1e-4, min(w, h) / 2.0))
    for i in range(N):
        bounds.append((0.0, w))
    for i in range(N):
        bounds.append((0.0, h))

    maxiter = int(min(300, max(50, time_budget * 30)))

    try:
        res = minimize(
            objective, x_init, method='SLSQP', bounds=bounds, constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-10},
        )
        x = res.x
        x[:N] = np.maximum(x[:N], 1e-4)
        if _is_feasible(x, w):
            return x
    except Exception:
        pass

    # Fallback: penalty method
    if time.time() - start_time < time_budget * 0.5:
        return _penalty_optimize(x_init, w, h, time_budget - (time.time() - start_time))

    return None


def _penalty_optimize(x_init, w, h, time_budget):
    """Penalty-based optimization as a fallback."""
    start_time = time.time()
    x = x_init.copy()

    def penalized_objective(x):
        r = x[:N]
        cx = x[N:2 * N]
        cy = x[2 * N:3 * N]
        obj = -np.sum(r)
        penalty = 0.0
        for i in range(N):
            ri = r[i]
            if cx[i] - ri < 0:
                penalty += (ri - cx[i]) ** 2
            if cx[i] + ri > w:
                penalty += (cx[i] + ri - w) ** 2
            if cy[i] - ri < 0:
                penalty += (ri - cy[i]) ** 2
            if cy[i] + ri > h:
                penalty += (cy[i] + ri - h) ** 2
        for i in range(N):
            for j in range(i + 1, N):
                d2 = (cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2
                min_d = r[i] + r[j]
                if d2 < min_d ** 2:
                    violation = min_d - np.sqrt(d2)
                    penalty += violation ** 2
        return obj + 10.0 * penalty

    maxiter = int(min(200, max(30, time_budget * 20)))
    try:
        res = minimize(penalized_objective, x, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-10})
        x = res.x
        x[:N] = np.maximum(x[:N], 1e-4)
        if _is_feasible(x, w):
            return x
    except Exception:
        pass
    return None


def _local_refine(x, w, h, time_budget):
    """Local refinement via small perturbations."""
    start_time = time.time()
    x = x.copy()
    r = x[:N].copy()
    cx = x[N:2 * N].copy()
    cy = x[2 * N:3 * N].copy()
    current_value = _sum_radii(x)
    rng = np.random.default_rng(12345)

    for step in range(500):
        if time.time() - start_time > time_budget:
            break
        idx = rng.integers(N)
        delta_c = rng.normal(0, 0.003, 2)
        delta_r = rng.uniform(-0.002, 0.003)
        old_cx, old_cy, old_r = cx[idx], cy[idx], r[idx]
        new_cx, new_cy, new_r = old_cx + delta_c[0], old_cy + delta_c[1], old_r + delta_r
        if new_r < 1e-4 or new_cx < new_r or new_cx > w - new_r or new_cy < new_r or new_cy > h - new_r:
            continue
        cx[idx], cy[idx], r[idx] = new_cx, new_cy, new_r
        feasible = True
        for j in range(N):
            if j == idx:
                continue
            d2 = (cx[idx] - cx[j]) ** 2 + (cy[idx] - cy[j]) ** 2
            if d2 < (r[idx] + r[j]) ** 2 - 1e-7:
                feasible = False
                break
        if feasible:
            new_value = _sum_radii(np.concatenate([r, cx, cy]))
            if new_value <= current_value:
                cx[idx], cy[idx], r[idx] = old_cx, old_cy, old_r
        else:
            cx[idx], cy[idx], r[idx] = old_cx, old_cy, old_r

    x[:N] = r
    x[N:2 * N] = cx
    x[2 * N:3 * N] = cy
    return x


def construct():
    start_time = time.time()
    best_result = None
    best_val = -1e9
    best_w = None

    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.35, 0.7, 1.45, 0.65, 1.55]

    for ar in aspect_ratios:
        if time.time() - start_time > _DEADLINE - 15:
            break
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        if h <= 0.1 or w <= 0.1:
            continue

        seed = _grid_seed(w, h)
        seed_relaxed = _force_directed_relax(seed, w, h, iterations=100)

        time_budget = _DEADLINE - (time.time() - start_time) - 20
        if time_budget < 5:
            continue

        result = _slsqp_optimize(seed_relaxed, w, h, time_budget)

        if result is not None:
            val = _sum_radii(result)
            if val > best_val:
                best_val = val
                best_result = result
                best_w = w
        elif _is_feasible(seed, w):
            val = _sum_radii(seed)
            if val > best_val:
                best_val = val
                best_result = seed
                best_w = w

        if time.time() - start_time > _DEADLINE - 10:
            break

    # Final refinement
    if best_result is not None and best_w is not None:
        if time.time() - start_time < _DEADLINE - 5:
            h = 2.0 - best_w
            refined = _local_refine(best_result, best_w, h, _DEADLINE - (time.time() - start_time))
            if _is_feasible(refined, best_w):
                val = _sum_radii(refined)
                if val > best_val:
                    best_val = val
                    best_result = refined

    if best_result is None:
        w = 1.0
        h = 1.0
        r = 1e-4
        xs = np.linspace(r, w - r, N)
        ys = np.full(N, h / 2.0)
        return np.column_stack((xs, ys, r))

    r = best_result[:N]
    cx = best_result[N:2 * N]
    cy = best_result[2 * N:3 * N]
    return np.column_stack((cx, cy, r))
