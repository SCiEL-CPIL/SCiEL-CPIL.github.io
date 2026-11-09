"""Circle packing in perimeter-4 rectangle.
Maximize sum of 21 disjoint circle radii inside a rectangle with perimeter <= 4.

Combined strategy:
1. Parametric layouts (grid, hexagonal, mixed-row) -> tight bounding rectangle.
2. Rectangle-shape optimization: for a given configuration, optimize aspect ratio.
3. Vectorized local optimization (Nelder-Mead on [r, x, y]) to improve a layout.
4. Randomized restarts of the optimizer to escape local optima.
5. Explicit deadline guard; return best found before the wall-clock limit.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # safety margin before the 150s kill


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


def _tight_rect_for_config(x, y, r):
    """Given circle centers and radius, compute tight bounding rectangle.
    w = (xmax - xmin) + 2r, h = (ymax - ymin) + 2r.
    Scale so w + h = 2."""
    W = (x.max() - x.min()) + 2.0 * r
    H = (y.max() - y.min()) + 2.0 * r
    if W + H <= 0:
        return None
    w = 2.0 * W / (W + H)
    h = 2.0 * H / (W + H)
    return w, h


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


def _optimize_positions(x, y, r, w, h, maxiter=300):
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


def _optimize_aspect(x, y, r):
    """For a given circle configuration, find the best (w,h) with w+h=2
    such that all circles fit. Returns improved (x, y, r, w, h) or None."""
    # The configuration has a bounding box: xmin, xmax, ymin, ymax
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    W0 = xmax - xmin  # width of center span
    H0 = ymax - ymin  # height of center span

    # For a given w (and h = 2-w), the max radius is:
    # r <= min(w/2, h/2, (w - W0)/2, (h - H0)/2, min_pair_dist/2)
    # Actually: r <= min(w/2, h/2) for boundary (circles must fit in box)
    # and r <= (w - W0)/2, (h - H0)/2 for the span to fit
    # and r <= min_pair_dist / 2 for non-overlap
    
    # Compute min pair distance
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    min_d = d.min()

    def max_r_for_w(w):
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return 0.0
        r_boundary = min(w / 2.0, h / 2.0)
        r_span_w = (w - W0) / 2.0
        r_span_h = (h - H0) / 2.0
        r_pair = min_d / 2.0
        return max(0.0, min(r_boundary, r_span_w, r_span_h, r_pair))

    # Optimize w to maximize max_r_for_w(w)
    # w must satisfy: w >= W0 + 2*min_r and h >= H0 + 2*min_r
    # Try a grid of w values
    w_min = max(W0 + 1e-6, 0.01)
    w_max = min(2.0 - H0 - 1e-6, 1.99)
    if w_min >= w_max:
        return None

    best_r = 0.0
    best_w = None
    # Coarse grid
    for w_try in np.linspace(w_min, w_max, 100):
        r_try = max_r_for_w(w_try)
        if r_try > best_r:
            best_r = r_try
            best_w = w_try

    if best_w is None:
        return None

    # Fine grid around best
    delta = (w_max - w_min) / 500.0
    for w_try in np.linspace(max(w_min, best_w - 5*delta), min(w_max, best_w + 5*delta), 100):
        r_try = max_r_for_w(w_try)
        if r_try > best_r:
            best_r = r_try
            best_w = w_try

    if best_r <= r + 1e-10:
        return None

    h = 2.0 - best_w
    # Shift circles so they're centered in the box
    # Current centers span [xmin, xmax] x [ymin, ymax]
    # We want them to fit in [r, w-r] x [r, h-r]
    # The tightest fit means the span exactly touches:
    # xmin_new = r, xmax_new = w - r  => shift_x = r - xmin
    # But we also need the pair distances to be >= 2*best_r
    # Since best_r <= min_d/2, the pair constraint is satisfied.
    # We need to place circles so that:
    #   all x in [best_r, w - best_r]
    #   all y in [best_r, h - best_r]
    # The span is W0 in x and H0 in y.
    # w - 2*best_r >= W0 and h - 2*best_r >= H0 by construction.
    # Center the span in the box:
    shift_x = (w - W0) / 2.0 - xmin  # so that new xmin = (w-W0)/2, xmax = (w+W0)/2
    shift_y = (h - H0) / 2.0 - ymin

    x_new = x + shift_x
    y_new = y + shift_y

    # Verify
    if _is_valid(x_new, y_new, best_r, best_w, h):
        return x_new, y_new, best_r, best_w, h
    return None


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
        [3, 4, 4, 4, 4, 2], [4, 4, 4, 3, 4, 2], [5, 4, 4, 3, 4, 1],
        [4, 3, 4, 3, 4, 3], [3, 4, 3, 4, 3, 4], [4, 4, 3, 4, 3, 3],
    ]
    for counts in base_counts:
        if sum(counts) >= N:
            candidates.append(('mixed', counts))

    for cand in candidates:
        if time.time() > DEADLINE - 30.0:
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

            # Use first N circles
            xs_sub = xs[:N]
            ys_sub = ys[:N]

            # Compute the natural radius from the layout spacing
            # The layout is defined with unit radius; scale to fit w+h=2
            W_sub = xs_sub.max() - xs_sub.min()
            H_sub = ys_sub.max() - ys_sub.min()

            # For the tight rectangle:
            # w = W_sub + 2r, h = H_sub + 2r, w + h = 2
            # => 2r = 2 - W_sub - H_sub => r = (2 - W_sub - H_sub) / 2
            # But also r must be <= min_pair_dist / 2
            dx = xs_sub[:, None] - xs_sub[None, :]
            dy = ys_sub[:, None] - ys_sub[None, :]
            d = np.sqrt(dx * dx + dy * dy)
            np.fill_diagonal(d, np.inf)
            min_d = d.min()

            r_rect = (2.0 - W_sub - H_sub) / 2.0
            r_pair = min_d / 2.0
            r = min(r_rect, r_pair)

            if r <= 0:
                continue

            w = W_sub + 2.0 * r
            h = H_sub + 2.0 * r

            # Shift circles to center in box
            x = (xs_sub - xs_sub.min()) + (w - W_sub) / 2.0
            y = (ys_sub - ys_sub.min()) + (h - H_sub) / 2.0

            consider(x, y, r, w, h)

            # Also try aspect optimization
            if time.time() < DEADLINE - 25.0:
                result = _optimize_aspect(x, y, r)
                if result is not None:
                    x2, y2, r2, w2, h2 = result
                    consider(x2, y2, r2, w2, h2)
        except Exception:
            continue

    # ---- Phase 2: local optimization + randomized restarts ----
    if best_result is not None and time.time() < DEADLINE - 15.0:
        bx = best_result[:, 0].copy()
        by = best_result[:, 1].copy()
        br = best_result[0, 2]
        # Recover w, h from the tight bounding box
        Wb = bx.max() - bx.min()
        Hb = by.max() - by.min()
        bw = Wb + 2.0 * br
        bh = Hb + 2.0 * br
        # Normalize to w+h=2
        if bw + bh > 0:
            bw = 2.0 * bw / (bw + bh)
            bh = 2.0 * bh / (bw + bh)

        # First, try aspect optimization on the best layout
        if time.time() < DEADLINE - 20.0:
            result = _optimize_aspect(bx, by, br)
            if result is not None:
                x2, y2, r2, w2, h2 = result
                consider(x2, y2, r2, w2, h2)
                if r2 > br + 1e-9:
                    bx, by, br = x2, y2, r2
                    bw, bh = w2, h2

        # Local optimization of the best layout
        if time.time() < DEADLINE - 15.0:
            x, y, r = _optimize_positions(bx.copy(), by.copy(), br, bw, bh, maxiter=400)
            consider(x, y, r, bw, bh)
            # Try aspect optimization on the optimized positions
            if time.time() < DEADLINE - 12.0:
                result = _optimize_aspect(x, y, r)
                if result is not None:
                    x2, y2, r2, w2, h2 = result
                    consider(x2, y2, r2, w2, h2)

        # Randomized restarts: nudge positions, re-optimize
        rng = np.random.default_rng(42)
        n_trials = 0
        while time.time() < DEADLINE - 8.0 and n_trials < 300:
            n_trials += 1
            x0 = bx.copy()
            y0 = by.copy()
            amp = 0.15 * br
            x0 = x0 + rng.uniform(-amp, amp, N)
            y0 = y0 + rng.uniform(-amp, amp, N)
            x0 = np.clip(x0, br, bw - br)
            y0 = np.clip(y0, br, bh - br)
            x2, y2, r2 = _optimize_positions(x0, y0, br, bw, bh, maxiter=200)
            consider(x2, y2, r2, bw, bh)
            # Also try aspect optimization
            if time.time() < DEADLINE - 8.0:
                result = _optimize_aspect(x2, y2, r2)
                if result is not None:
                    x3, y3, r3, w3, h3 = result
                    consider(x3, y3, r3, w3, h3)

    if best_result is not None:
        return best_result

    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])
