"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 21
DEADLINE = 135.0  # safety margin under 150s wall-clock


def _validate(circles, w, h, tol=1e-6):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True


def _total_radius(circles):
    return float(np.sum(circles[:, 2]))


def _relax_circles(circles, w, h, iterations=5000, tol=1e-9):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    Uses iterative pairwise separation with boundary clamping.
    """
    n = len(circles)
    pos = np.array(circles[:, :2], dtype=float)
    rad = np.array(circles[:, 2], dtype=float)

    for _ in range(iterations):
        moved = False

        # 1. Resolve boundary violations
        for i in range(n):
            if pos[i, 0] - rad[i] < 0:
                pos[i, 0] = rad[i]
                moved = True
            if pos[i, 0] + rad[i] > w:
                pos[i, 0] = w - rad[i]
                moved = True
            if pos[i, 1] - rad[i] < 0:
                pos[i, 1] = rad[i]
                moved = True
            if pos[i, 1] + rad[i] > h:
                pos[i, 1] = h - rad[i]
                moved = True

        # 2. Resolve overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]

                if dist < min_dist - tol:
                    if dist < 1e-9:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                    else:
                        push = (min_dist - dist) / 2.0
                        ux = dx / dist
                        uy = dy / dist
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[j, 0] += ux * push
                        pos[j, 1] += uy * push
                    moved = True

        if not moved:
            break

    # Final boundary clamp
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])

    new_circles = np.column_stack((pos, rad))

    if _validate(new_circles, w, h, tol=1e-5):
        return new_circles

    return circles


def _make_hex_grid(n, w, h):
    """Create a hexagonal packing initialization."""
    # For 21 circles, try a hexagonal arrangement
    # 1-2-3-4-3-2-1 = 16, not enough
    # 2-3-4-4-3-2 = 18, not enough
    # 3-4-4-4-3 = 18
    # Let's try rows: 4-5-4-4-4 = 21
    rows = [4, 5, 4, 4, 4]
    cells = []
    total_rows = len(rows)
    row_h = h / total_rows
    for ri, count in enumerate(rows):
        y = (ri + 0.5) * row_h
        for ci in range(count):
            x = (ci + 0.5) * w / count
            r = 0.5 * min(w / count, row_h) * 0.85
            cells.append([x, y, r])
    return np.array(cells[:n])


def _make_grid(n, w, h):
    """Create a grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = 0.5 * min(w / side, h / side) * 0.85
            cells.append([cx, cy, r])
    return np.array(cells[:n])


def _make_random_init(w, h, n, seed=42):
    """Create a random initialization with non-overlapping circles."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    max_r = min(w, h) / 2 * 0.4
    for i in range(n):
        attempts = 0
        while attempts < 200:
            r = rng.uniform(0.05, max_r)
            x = rng.uniform(r, w - r)
            y = rng.uniform(r, h - r)
            # Check against existing circles
            ok = True
            for j in range(i):
                dx = x - circles[j, 0]
                dy = y - circles[j, 1]
                if np.sqrt(dx*dx + dy*dy) < r + circles[j, 2] + 0.01:
                    ok = False
                    break
            if ok:
                circles[i] = [x, y, r]
                break
            attempts += 1
        if attempts >= 200:
            # Place in a corner with small radius
            r = 0.05
            circles[i] = [r, r, r]
    return circles


def _penalty_objective(x, w, h, penalty_weight=100.0):
    """
    Penalty-based objective: minimize negative sum of radii + penalty for violations.
    x = [x1, y1, r1, x2, y2, r2, ...]
    """
    n = len(x) // 3
    r = x[2::3]
    pos_x = x[0::3]
    pos_y = x[1::3]

    obj = -np.sum(r)

    # Boundary penalties
    for i in range(n):
        if pos_x[i] - r[i] < 0:
            obj += penalty_weight * (pos_x[i] - r[i]) ** 2
        if pos_x[i] + r[i] > w:
            obj += penalty_weight * (pos_x[i] + r[i] - w) ** 2
        if pos_y[i] - r[i] < 0:
            obj += penalty_weight * (pos_y[i] - r[i]) ** 2
        if pos_y[i] + r[i] > h:
            obj += penalty_weight * (pos_y[i] + r[i] - h) ** 2

    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos_x[i] - pos_x[j]
            dy = pos_y[i] - pos_y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                obj += penalty_weight * (min_dist - dist) ** 2

    return obj


def _penalty_grad(x, w, h, penalty_weight=100.0):
    """Gradient of penalty objective."""
    n = len(x) // 3
    r = x[2::3]
    pos_x = x[0::3]
    pos_y = x[1::3]

    grad = np.zeros_like(x)

    # d(-sum(r))/dr_i = -1
    for i in range(n):
        grad[2*i + 2] = -1.0

    # Boundary gradients
    for i in range(n):
        if pos_x[i] - r[i] < 0:
            grad[2*i] += 2 * penalty_weight * (pos_x[i] - r[i])
            grad[2*i + 2] += -2 * penalty_weight * (pos_x[i] - r[i])
        if pos_x[i] + r[i] > w:
            grad[2*i] += 2 * penalty_weight * (pos_x[i] + r[i] - w)
            grad[2*i + 2] += 2 * penalty_weight * (pos_x[i] + r[i] - w)
        if pos_y[i] - r[i] < 0:
            grad[2*i + 1] += 2 * penalty_weight * (pos_y[i] - r[i])
            grad[2*i + 2] += -2 * penalty_weight * (pos_y[i] - r[i])
        if pos_y[i] + r[i] > h:
            grad[2*i + 1] += 2 * penalty_weight * (pos_y[i] + r[i] - h)
            grad[2*i + 2] += 2 * penalty_weight * (pos_y[i] + r[i] - h)

    # Overlap gradients
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos_x[i] - pos_x[j]
            dy = pos_y[i] - pos_y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                pen = 2 * penalty_weight * (min_dist - dist)
                if dist > 1e-9:
                    ux = dx / dist
                    uy = dy / dist
                    grad[2*i] += pen * ux
                    grad[2*i + 1] += pen * uy
                    grad[2*j] -= pen * ux
                    grad[2*j + 1] -= pen * uy
                grad[2*i + 2] += pen
                grad[2*j + 2] += pen

    return grad


def _local_optimize(circles, w, h, t_deadline, max_iters=500):
    """
    Local optimization using L-BFGS-B with penalty method.
    """
    n = len(circles)
    x0 = np.array([v for c in circles for v in c])

    best_x = x0.copy()
    best_obj = _penalty_objective(x0, w, h)

    for it in range(max_iters):
        if time.time() > t_deadline:
            break

        # Adaptive penalty weight
        pw = 50.0 + 50.0 * (it / max_iters)

        res = minimize(
            lambda x: _penalty_objective(x, w, h, pw),
            x0,
            jac=lambda x: _penalty_grad(x, w, h, pw),
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
        )

        if res.x is not None:
            new_obj = _penalty_objective(res.x, w, h, pw)
            if new_obj < best_obj:
                best_obj = new_obj
                best_x = res.x.copy()
            x0 = res.x

    # Reconstruct circles
    new_circles = np.array([(best_x[i*3], best_x[i*3+1], best_x[i*3+2]) for i in range(n)])
    return new_circles


def _slsqp_polish(circles, w, h, t_deadline):
    """
    Final polish using SLSQP with hard constraints for high precision.
    """
    if time.time() > t_deadline:
        return circles

    n = len(circles)
    x0 = np.array([v for c in circles for v in c])

    def objective(x):
        return -np.sum(x[2::3])

    constraints = []

    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})

    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })

    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
        bounds.append((1e-6, min(w, h)/2))

    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-12}
        )

        if res.success:
            new_circles = np.array([(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)])
            if _validate(new_circles, w, h, tol=1e-5):
                return new_circles
    except Exception:
        pass

    return circles


def _high_precision_polish(circles, w, h, t_deadline):
    """
    Very tight local search: iteratively nudge each circle to maximize its radius
    while maintaining validity, using fine-grained position adjustments.
    """
    if time.time() > t_deadline:
        return circles

    n = len(circles)
    circles = circles.copy()

    for _ in range(50):
        if time.time() > t_deadline:
            break
        improved = False
        for i in range(n):
            # Try to slightly increase radius
            old_r = circles[i, 2]
            new_r = old_r * 1.001
            circles[i, 2] = new_r
            circles[i, 0] = np.clip(circles[i, 0], new_r, w - new_r)
            circles[i, 1] = np.clip(circles[i, 1], new_r, h - new_r)

            # Relax to fix any overlaps
            relaxed = _relax_circles(circles, w, h, iterations=200)
            if _validate(relaxed, w, h, tol=1e-6):
                if _total_radius(relaxed) > _total_radius(circles) - 1e-9:
                    circles = relaxed
                    improved = True
                else:
                    circles[i, 2] = old_r
            else:
                circles[i, 2] = old_r

        if not improved:
            break

    return circles


def _solve_aspect(w, h, t0, t_deadline):
    """Solve for a given aspect ratio."""
    # Try multiple initializations
    best_circles = None
    best_score = -np.inf

    inits = [
        _make_hex_grid(N, w, h),
        _make_grid(N, w, h),
        _make_random_init(w, h, N, seed=1),
        _make_random_init(w, h, N, seed=42),
    ]

    for init in inits:
        if time.time() > t_deadline:
            break

        # Relax initial overlaps
        relaxed = _relax_circles(init, w, h, iterations=3000)
        if not _validate(relaxed, w, h, tol=1e-5):
            continue

        # Local optimization
        optimized = _local_optimize(relaxed, w, h, t_deadline - 5.0, max_iters=200)

        # SLSQP polish
        polished = _slsqp_polish(optimized, w, h, t_deadline - 2.0)

        # High precision polish
        final = _high_precision_polish(polished, w, h, t_deadline - 0.5)

        score = _total_radius(final)
        if score > best_score and _validate(final, w, h, tol=1e-6):
            best_score = score
            best_circles = final.copy()

    if best_circles is None:
        # Fallback
        relaxed = _relax_circles(_make_grid(N, w, h), w, h, iterations=5000)
        if _validate(relaxed, w, h, tol=1e-5):
            return relaxed
        else:
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                     for i in range(side) for j in range(side)]
            return np.array(cells[:N])

    return best_circles


def construct():
    t0 = time.time()
    t_deadline = t0 + DEADLINE

    best = None
    best_score = -np.inf

    # Search over aspect ratios
    # Perimeter = 2(w+h) = 4 => w + h = 2
    # aspect = w/h, so w = ar * h, (ar+1)*h = 2, h = 2/(ar+1), w = 2*ar/(ar+1)
    ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 2.0, 0.4, 1.1, 0.9]

    for ar in ratios:
        if time.time() > t_deadline:
            break

        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)

        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() > t_deadline:
                break
            try:
                x = _solve_aspect(ww, hh, t0, t_deadline)
            except Exception:
                continue

            s = _total_radius(x)
            if np.isfinite(s) and s > best_score and _validate(x, ww, hh, tol=1e-6):
                best_score = s
                best = (x.copy(), ww, hh)

    if best is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, w, h = best

    # Final safety: validate and shrink if needed
    if not _validate(x, w, h, tol=1e-6):
        for i in range(N):
            x[i, 2] *= 0.999
        if not _validate(x, w, h, tol=1e-6):
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                     for i in range(side) for j in range(side)]
            return np.array(cells[:N])

    return x
