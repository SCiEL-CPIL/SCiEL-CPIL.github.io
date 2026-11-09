"""Circle packing in perimeter-4 rectangle.
Maximize sum of 21 disjoint circle radii inside a rectangle with perimeter <= 4.

Combined strategy:
1. Parametric layouts (grid, hexagonal, mixed-row) -> analytic optimal aspect ratio.
2. Vectorized local optimization (Nelder-Mead on [r, x, y]) to improve a layout.
3. Randomized restarts of the optimizer to escape the local optimum and beat the seed.
4. Explicit deadline guard; return best found before the wall-clock limit.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # safety margin before the 150s kill


def _hexagonal_layout(n_cols, n_rows):
    xs = []
    ys = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = col * 2.0 + (1.0 if row % 2 == 1 else 0.0)
            y = row * np.sqrt(3.0)
            xs.append(x)
            ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    xs = xs - xs.min() + 1.0
    ys = ys - ys.min() + 1.0
    W = xs.max() - xs.min() + 2.0
    H = ys.max() - ys.min() + 2.0
    return xs, ys, W, H


def _grid_layout(n_cols, n_rows):
    xs = np.array([col * 2.0 + 1.0 for row in range(n_rows) for col in range(n_cols)])
    ys = np.array([row * 2.0 + 1.0 for row in range(n_rows) for col in range(n_cols)])
    W = (n_cols - 1) * 2.0 + 2.0
    H = (n_rows - 1) * 2.0 + 2.0
    return xs, ys, W, H


def _mixed_row_layout(row_counts):
    xs = []
    ys = []
    for row_idx, n_circ in enumerate(row_counts):
        y = row_idx * np.sqrt(3.0)
        offset = 1.0 if row_idx % 2 == 1 else 0.0
        for col in range(n_circ):
            xs.append(col * 2.0 + offset)
            ys.append(y)
    if len(xs) == 0:
        return None
    xs = np.array(xs)
    ys = np.array(ys)
    xs = xs - xs.min() + 1.0
    ys = ys - ys.min() + 1.0
    W = xs.max() - xs.min() + 2.0
    H = ys.max() - ys.min() + 2.0
    return xs, ys, W, H


def _analytic_layout(xs, ys, w, h):
    """Scale a unit-radius layout (xs,ys,W,H) into rectangle w x h (w+h=2).
    Returns (x, y, r) arrays of length N."""
    xs_sub = xs[:N]
    ys_sub = ys[:N]
    W_sub = xs_sub.max() - xs_sub.min() + 2.0
    H_sub = ys_sub.max() - ys_sub.min() + 2.0
    r = 2.0 / (W_sub + H_sub)
    w_opt = 2.0 * W_sub / (W_sub + H_sub)
    h_opt = 2.0 * H_sub / (W_sub + H_sub)
    x = (xs_sub - xs_sub.min()) * r + r
    y = (ys_sub - ys_sub.min()) * r + r
    return x, y, r, w_opt, h_opt


def _is_valid(x, y, r, w, h, tol=1e-7):
    n = len(x)
    if r <= 0 or w + h > 2.0 + tol:
        return False
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    if d.min() < 2.0 * r - tol:
        return False
    return True


def _objective(params, w, h):
    r = params[0]
    if r <= 0:
        return 1e6
    x = params[1::2]
    y = params[2::2]
    n = len(x)
    # boundary violations
    bx = np.minimum(np.minimum(r - x, x - (w - r)), 0.0)
    by = np.minimum(np.minimum(r - y, y - (h - r)), 0.0)
    pen = np.sum(bx * bx) + np.sum(by * by)
    # pair violations
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    gap = 2.0 * r - d
    viol = np.maximum(gap, 0.0)
    pen += np.sum(viol * viol)
    return -r + 50.0 * pen


def _optimize(x, y, r, w, h, maxiter=300):
    n = len(x)
    params0 = np.concatenate([[r], x, y])
    try:
        res = minimize(_objective, params0, args=(w, h), method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-5, 'fatol': 1e-5})
        r2 = res.x[0]
        x2 = res.x[1::2]
        y2 = res.x[2::2]
        if r2 > r + 1e-9 and _is_valid(x2, y2, r2, w, h):
            return x2, y2, r2
    except Exception:
        pass
    return x, y, r


def construct():
    start = time.time()
    best_sum = 0.0
    best_result = None

    def consider(x, y, r, w, h):
        nonlocal best_sum, best_result
        if _is_valid(x, y, r, w, h):
            s = N * r
            if s > best_sum + 1e-12:
                best_sum = s
                best_result = np.column_stack([x, y, np.full(N, r)])

    # ---- Phase 1: analytic layouts ----
    candidates = []
    for nc in range(1, 12):
        for nr in range(1, 12):
            if nc * nr >= N:
                candidates.append(('grid', nc, nr))
                candidates.append(('hex', nc, nr))
    base_counts = [
        [4, 4, 4, 4, 3, 2], [5, 4, 4, 4, 3, 1], [3, 4, 4, 4, 4, 2],
        [4, 3, 4, 4, 4, 2], [5, 5, 4, 3, 2, 2], [6, 4, 4, 3, 2, 2],
        [4, 5, 4, 4, 3, 1], [3, 4, 4, 5, 3, 2], [5, 4, 5, 3, 2, 2],
        [4, 4, 5, 4, 3, 1], [3, 3, 3, 3, 3, 3, 3], [4, 4, 4, 4, 5],
        [5, 4, 4, 4, 4], [6, 4, 4, 4, 3], [7, 3, 3, 3, 3, 2],
    ]
    for counts in base_counts:
        if sum(counts) >= N:
            candidates.append(('mixed', counts))

    for cand in candidates:
        if time.time() > DEADLINE - 25.0:
            break
        try:
            if cand[0] == 'grid':
                xs, ys, W, H = _grid_layout(cand[1], cand[2])
            elif cand[0] == 'hex':
                xs, ys, W, H = _hexagonal_layout(cand[1], cand[2])
            else:
                res = _mixed_row_layout(cand[1])
                if res is None:
                    continue
                xs, ys, W, H = res
            x, y, r, w, h = _analytic_layout(xs, ys, W, H)
            consider(x, y, r, w, h)
        except Exception:
            continue

    # ---- Phase 2: local optimization + randomized restarts ----
    if best_result is not None and time.time() < DEADLINE - 15.0:
        bx = best_result[:, 0].copy()
        by = best_result[:, 1].copy()
        br = best_result[0, 2]
        bw = None
        bh = None
        # recover w,h from the best analytic layout: recompute from positions
        # use the layout that produced best; recompute w,h via aspect from bbox
        Wb = bx.max() - bx.min() + 2.0 * br
        Hb = by.max() - by.min() + 2.0 * br
        bw = 2.0 * Wb / (Wb + Hb)
        bh = 2.0 * Hb / (Wb + Hb)

        # first, a plain local optimization of the best layout
        x, y, r, w, h = bx.copy(), by.copy(), br, bw, bh
        x, y, r = _optimize(x, y, r, w, h, maxiter=400)
        consider(x, y, r, w, h)

        # randomized restarts: nudge positions, re-optimize
        rng = np.random.default_rng(0)
        for trial in range(200):
            if time.time() > DEADLINE - 10.0:
                break
            x0 = bx.copy()
            y0 = by.copy()
            # random perturbation scaled to a fraction of radius
            amp = 0.15 * br
            x0 = x0 + rng.uniform(-amp, amp, N)
            y0 = y0 + rng.uniform(-amp, amp, N)
            # keep inside box
            x0 = np.clip(x0, br, bw - br)
            y0 = np.clip(y0, br, bh - br)
            x2, y2, r2 = _optimize(x0, y0, br, bw, bh, maxiter=200)
            consider(x2, y2, r2, bw, bh)

    if best_result is not None:
        return best_result

    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])
