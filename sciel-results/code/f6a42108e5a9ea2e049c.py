"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # seconds, leave margin under 150s kill


def _rect_from_aspect(aspect):
    """Return (w, h) for a rectangle of perimeter 4 with given aspect ratio w/h."""
    # 2(w+h) = 4 => w+h = 2, w = aspect*h => h(aspect+1)=2
    h = 2.0 / (aspect + 1.0)
    w = aspect * h
    return w, h


def _score(x, aspect):
    """x: (n,3) array of [cx, cy, r]. Returns (sum_r, penalty)."""
    w, h = _rect_from_aspect(aspect)
    n = len(x)
    cx, cy, r = x[:, 0], x[:, 1], x[:, 2]

    # Wall violations
    wall_pen = 0.0
    wall_pen += np.sum(np.maximum(0, cx + r - w)) ** 2
    wall_pen += np.sum(np.maximum(0, w - cx + r)) ** 2  # cx - r < 0
    wall_pen += np.sum(np.maximum(0, cy + r - h)) ** 2
    wall_pen += np.sum(np.maximum(0, h - cy + r)) ** 2  # cy - r < 0

    # Circle-circle overlap: dist < r_i + r_j
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx * dx + dy * dy)
            pen = np.maximum(0.0, r[i] + r[j] - d)
            overlap_pen += pen * pen

    total_pen = wall_pen + overlap_pen
    return np.sum(r), total_pen


def _objective(x, aspect, w, h):
    """Objective to minimize: -sum_r + lambda * penalty."""
    n = len(x)
    cx, cy, r = x[:, 0], x[:, 1], x[:, 2]
    lam = 10.0

    wall_pen = 0.0
    wall_pen += np.sum(np.maximum(0, cx + r - w)) ** 2
    wall_pen += np.sum(np.maximum(0, -cx + r)) ** 2
    wall_pen += np.sum(np.maximum(0, cy + r - h)) ** 2
    wall_pen += np.sum(np.maximum(0, -cy + r)) ** 2

    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx * dx + dy * dy)
            pen = np.maximum(0.0, r[i] + r[j] - d)
            overlap_pen += pen * pen

    total_pen = wall_pen + overlap_pen
    return -np.sum(r) + lam * total_pen


def _objective_grad(x, aspect, w, h):
    n = len(x)
    cx, cy, r = x[:, 0], x[:, 1], x[:, 2]
    lam = 10.0
    gx = np.zeros_like(x)

    # Wall gradients
    for i in range(n):
        # cx + r - w > 0
        v = cx[i] + r[i] - w
        if v > 0:
            gx[i, 0] += 2 * lam * v
            gx[i, 2] += 2 * lam * v
        # -cx + r > 0
        v = -cx[i] + r[i]
        if v > 0:
            gx[i, 0] += -2 * lam * v
            gx[i, 2] += 2 * lam * v
        # cy + r - h > 0
        v = cy[i] + r[i] - h
        if v > 0:
            gx[i, 1] += 2 * lam * v
            gx[i, 2] += 2 * lam * v
        # -cy + r > 0
        v = -cy[i] + r[i]
        if v > 0:
            gx[i, 1] += -2 * lam * v
            gx[i, 2] += 2 * lam * v

    # Circle-circle
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx * dx + dy * dy)
            pen = r[i] + r[j] - d
            if pen > 0:
                if d > 1e-12:
                    ux, uy = dx / d, dy / d
                else:
                    ux, uy = 1.0, 0.0
                gx[i, 0] += 2 * lam * pen * ux
                gx[i, 1] += 2 * lam * pen * uy
                gx[j, 0] += -2 * lam * pen * ux
                gx[j, 1] += -2 * lam * pen * uy
                gx[i, 2] += 2 * lam * pen
                gx[j, 2] += 2 * lam * pen

    # Objective gradient: -sum(r) => -1 for each r
    gx[:, 2] -= 1.0
    return gx


def _grid_init(w, h, n):
    """Create n circles in a grid pattern inside (w,h)."""
    cols = int(np.ceil(np.sqrt(n * w / h)))
    rows = int(np.ceil(n / cols))
    cw = w / cols
    ch = h / rows
    r = 0.45 * min(cw, ch)
    circles = []
    for i in range(n):
        ci = i % cols
        ri = i // cols
        cx = (ci + 0.5) * cw
        cy = (ri + 0.5) * ch
        circles.append([cx, cy, r])
    return np.array(circles)


def _optimize_circles(x0, aspect, max_time=30.0):
    """Optimize circle positions/radii for fixed aspect."""
    w, h = _rect_from_aspect(aspect)
    start = time.time()
    x = x0.copy()
    best_x = x.copy()
    best_score, _ = _score(x, aspect)

    # L-BFGS-B with bounds
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0.001, min(w, h) / 2.0))

    def obj(v):
        return _objective(v.reshape(-1, 3), aspect, w, h)

    def grad(v):
        return _objective_grad(v.reshape(-1, 3), aspect, w, h).ravel()

    x_flat = x.ravel()
    result = minimize(obj, x_flat, jac=grad, method='L-BFGS-B',
                      bounds=bounds,
                      options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10,
                               'maxfun': 50000})

    if result.x is not None:
        x_new = result.x.reshape(-1, 3)
        # Clamp radii positive
        x_new[:, 2] = np.maximum(x_new[:, 2], 0.001)
        s, p = _score(x_new, aspect)
        if p < 1e-6 and s > best_score:
            best_x = x_new
            best_score = s

    # Polish with smaller step
    elapsed = time.time() - start
    if elapsed < max_time * 0.5:
        x = best_x.copy()
        x_flat = x.ravel()
        result2 = minimize(obj, x_flat, jac=grad, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12,
                                    'maxfun': 100000})
        if result2.x is not None:
            x_new = result2.x.reshape(-1, 3)
            x_new[:, 2] = np.maximum(x_new[:, 2], 0.001)
            s, p = _score(x_new, aspect)
            if p < 1e-6 and s > best_score:
                best_x = x_new
                best_score = s

    return best_x, best_score


def _anneal_aspect(x0, aspect0, max_time=60.0):
    """Simulated annealing over aspect ratio."""
    w, h = _rect_from_aspect(aspect0)
    best_aspect = aspect0
    best_x, best_score = _optimize_circles(x0, aspect0, max_time=20.0)

    start = time.time()
    T = 0.5
    T_min = 0.01
    cooling = 0.95
    aspect = aspect0

    while T > T_min and (time.time() - start) < max_time:
        # Propose new aspect
        delta_a = np.random.normal(0, 0.1)
        new_aspect = np.clip(aspect + delta_a, 0.3, 3.0)

        # Quick evaluation: scale positions to new rectangle
        new_w, new_h = _rect_from_aspect(new_aspect)
        scale_x = new_w / w
        scale_y = new_h / h
        x_prop = best_x.copy()
        x_prop[:, 0] *= scale_x
        x_prop[:, 1] *= scale_y
        # Scale radii by min scale
        scale_r = min(scale_x, scale_y)
        x_prop[:, 2] *= scale_r

        s_prop, p_prop = _score(x_prop, new_aspect)
        # If too many violations, skip expensive optimization
        if p_prop > 1.0:
            T *= cooling
            continue

        # Optimize at new aspect (shorter time for exploration)
        x_opt, s_opt = _optimize_circles(x_prop, new_aspect, max_time=15.0)
        s_opt, p_opt = _score(x_opt, new_aspect)
        if p_opt > 1e-4:
            T *= cooling
            continue

        delta_s = s_opt - best_score
        if delta_s > 0 or np.random.random() < np.exp(delta_s / T):
            aspect = new_aspect
            best_aspect = new_aspect
            best_x = x_opt
            best_score = s_opt
        T *= cooling
        w, h = _rect_from_aspect(aspect)

    return best_aspect, best_x, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Initial grid packing in a square-ish rectangle
    aspect0 = 1.0
    w, h = _rect_from_aspect(aspect0)
    x0 = _grid_init(w, h, N)

    # First pass: optimize circles at initial aspect
    best_x, best_score = _optimize_circles(x0, aspect0, max_time=30.0)
    elapsed = time.time() - start_time

    # Second pass: anneal aspect ratio if time permits
    if elapsed < DEADLINE - 40:
        aspect_best, x_best, score_best = _anneal_aspect(x0, aspect0,
                                                          max_time=DEADLINE - elapsed - 20)
        if score_best > best_score:
            best_x = x_best
            best_score = score_best
            aspect0 = aspect_best

    # Final polish pass
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        x_polished, s_polished = _optimize_circles(best_x, aspect0,
                                                    max_time=DEADLINE - elapsed - 5)
        if s_polished > best_score:
            best_x = x_polished
            best_score = s_polished

    # Final safety check
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 2:
        # Verify and fix any tiny violations
        w, h = _rect_from_aspect(aspect0)
        for i in range(N):
            best_x[i, 0] = np.clip(best_x[i, 0], best_x[i, 2], w - best_x[i, 2])
            best_x[i, 1] = np.clip(best_x[i, 1], best_x[i, 2], h - best_x[i, 2])
            best_x[i, 2] = max(best_x[i, 2], 0.001)
        # Shrink slightly to ensure no overlap
        for i in range(N):
            for j in range(i + 1, N):
                dx = best_x[i, 0] - best_x[j, 0]
                dy = best_x[i, 1] - best_x[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = best_x[i, 2] + best_x[j, 2]
                if d < min_d and d > 1e-12:
                    # Push apart
                    push = (min_d - d) / 2.0 + 1e-6
                    ux, uy = dx / d, dy / d
                    best_x[i, 0] += push * ux
                    best_x[i, 1] += push * uy
                    best_x[j, 0] -= push * ux
                    best_x[j, 1] -= push * uy
                    # Clamp to walls
                    best_x[i, 0] = np.clip(best_x[i, 0], best_x[i, 2], w - best_x[i, 2])
                    best_x[i, 1] = np.clip(best_x[i, 1], best_x[i, 2], h - best_x[i, 2])
                    best_x[j, 0] = np.clip(best_x[j, 0], best_x[j, 2], w - best_x[j, 2])
                    best_x[j, 1] = np.clip(best_x[j, 1], best_x[j, 2], h - best_x[j, 2])

    return best_x
