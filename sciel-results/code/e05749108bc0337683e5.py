"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np

N = 21
_DEADLINE = time.time() + 135.0  # safety margin vs 150s kill


def _validate(arr):
    if arr is None or arr.shape[0] != N:
        return False
    x = arr[:, 0]; y = arr[:, 1]; r = arr[:, 2]
    if np.any(r <= 0) or np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        return False
    if np.any(x - r < -1e-6) or np.any(x + r > 1e6) or np.any(y - r < -1e-6) or np.any(y + r > 1e6):
        # bounding box check below
        pass
    # bounding rectangle
    xmin, xmax = np.min(x - r), np.max(x + r)
    ymin, ymax = np.min(y - r), np.max(y + r)
    W = xmax - xmin
    H = ymax - ymin
    if 2.0 * (W + H) > 4.0 + 1e-5:
        return False
    # disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            rs = r[i] + r[j]
            if d2 < rs * rs - 1e-7:
                return False
    return True


def _grid_init(cols, rows, W, H, n):
    # place n circles in cols x rows grid (fill row-major), centered in W x H
    xs = np.zeros(n); ys = np.zeros(n); rs = np.zeros(n)
    cw = W / cols
    ch = H / rows
    for k in range(n):
        ci = k % cols
        ri = k // cols
        xs[k] = (ci + 0.5) * cw
        ys[k] = (ri + 0.5) * ch
    r = 0.5 * min(cw, ch)
    rs[:] = r
    return xs, ys, rs


def _optimize(aspect, cols, rows, iters=4000, seed=0):
    """Optimize positions+radius for fixed aspect ratio W/H.
    W+H=2 => W=2*aspect/(1+aspect), H=2/(1+aspect)."""
    a = aspect
    W = 2.0 * a / (1.0 + a)
    H = 2.0 / (1.0 + a)
    rng = np.random.default_rng(seed)

    xs, ys, rs = _grid_init(cols, rows, W, H, N)
    # tiny jitter to break symmetry
    xs += rng.normal(0, 1e-4, N)
    ys += rng.normal(0, 1e-4, N)

    # variables: xs, ys, r (scalar)
    # objective: maximize sum(r) = N*r  (all radii equal by symmetry)
    # constraints:
    #   boundary: 0 <= x_i - r, x_i + r <= W, 0 <= y_i - r, y_i + r <= H
    #   non-overlap: ||p_i - p_j|| >= 2r
    #
    # We use a penalty / projected gradient on a smoothed objective.
    # Define objective f = N*r - lambda * violation
    # Better: maximize r s.t. constraints -> use r as variable, push it up,
    # and when constraints violated, move circles / reduce r.

    # We'll do: each step, compute max feasible r given positions, but that's
    # min over constraints. Instead use gradient ascent on r with
    # soft penalty for violations.

    r = rs[0]
    lam = 50.0  # penalty weight

    best_sum = N * r
    best_state = (xs.copy(), ys.copy(), r)

    # adaptive step
    step_r = 1e-3
    step_p = 1e-3

    for it in range(iters):
        if time.time() > _DEADLINE - 5.0:
            break
        # compute violations and gradients
        # boundary violations
        viol = 0.0
        gx = np.zeros(N); gy = np.zeros(N); gr = 0.0

        # boundaries
        # x_i - r < 0  -> want to push x_i up, r down
        # x_i + r > W  -> push x_i down
        for i in range(N):
            if xs[i] - r < 0:
                d = -(xs[i] - r)
                viol += d * d
                gx[i] += 2 * d      # d( - (x-r) )/dx = -1, penalty d^2 -> grad 2d * (-1)?? 
                # Let's define penalty = sum of (max(0, margin))^2 where margin = - (x_i - r) when violated
                # margin = r - x_i (when x_i < r). d(margin)/dx_i = -1, d(margin)/dr = 1
                # penalty grad wrt x_i = 2*margin*(-1) = -2*margin ; wrt r = 2*margin
                # We'll just recompute cleanly below.
                pass
        # Recompute gradients cleanly
        viol = 0.0
        gx = np.zeros(N); gy = np.zeros(N); gr = 0.0

        # boundary: x_i - r >= 0  => margin = (x_i - r); if margin<0 penalty += margin^2
        for i in range(N):
            m = xs[i] - r
            if m < 0:
                viol += m * m
                gx[i] += 2.0 * m * 1.0     # d(m^2)/dx_i = 2m
                gr += 2.0 * m * (-1.0)     # d(m^2)/dr = 2m*(-1)
            m2 = W - (xs[i] + r)
            if m2 < 0:
                viol += m2 * m2
                gx[i] += 2.0 * m2 * (-1.0)
                gr += 2.0 * m2 * (-1.0)
            m3 = ys[i] - r
            if m3 < 0:
                viol += m3 * m3
                gy[i] += 2.0 * m3
                gr += 2.0 * m3 * (-1.0)
            m4 = H - (ys[i] + r)
            if m4 < 0:
                viol += m4 * m4
                gy[i] += 2.0 * m4 * (-1.0)
                gr += 2.0 * m4 * (-1.0)

        # non-overlap: dist_ij >= 2r  => margin = dist - 2r
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.hypot(dx, dy)
                m = d - 2.0 * r
                if m < 0:
                    viol += m * m
                    if d > 1e-9:
                        ux = dx / d; uy = dy / d
                    else:
                        ux = 1.0; uy = 0.0
                    # d(m)/dx_i = ux, d(m)/dx_j = -ux, d(m)/dr = -2
                    gx[i] += 2.0 * m * ux
                    gx[j] += 2.0 * m * (-ux)
                    gy[i] += 2.0 * m * uy
                    gy[j] += 2.0 * m * (-uy)
                    gr += 2.0 * m * (-2.0)

        # objective: maximize N*r - lam*viol
        # grad wrt r: N - lam*gr ; wrt x: -lam*gx ; wrt y: -lam*gy
        # ascent step
        xs += step_p * (-lam * gx) * -1.0  # careful with signs
        # Let's define: we maximize F = N*r - lam*viol
        # dF/dx_i = -lam * gx_i  where gx_i = d(viol)/dx_i
        # ascent: x_i += step * dF/dx_i = step * (-lam * gx_i)
        # I added wrong above; redo:
        # Actually recompute properly:
        pass

    # The above got messy; use a cleaner implementation below.
    return best_state


def _optimize_clean(aspect, cols, rows, iters=3000, seed=0):
    a = aspect
    W = 2.0 * a / (1.0 + a)
    H = 2.0 / (1.0 + a)
    rng = np.random.default_rng(seed)

    xs, ys, _ = _grid_init(cols, rows, W, H, N)
    xs = xs + rng.normal(0, 1e-3, N)
    ys = ys + rng.normal(0, 1e-3, N)
    r = 0.5 * min(W / cols, H / rows)

    lam = 20.0
    step_r = 5e-4
    step_p = 5e-4
    best_sum = N * r
    best = (xs.copy(), ys.copy(), r)

    for it in range(iters):
        if time.time() > _DEADLINE - 5.0:
            break
        gx = np.zeros(N); gy = np.zeros(N); gr = 0.0
        viol = 0.0
        # boundaries
        for i in range(N):
            m = xs[i] - r
            if m < 0:
                viol += m * m
                gx[i] += 2.0 * m
                gr += -2.0 * m
            m = W - xs[i] - r
            if m < 0:
                viol += m * m
                gx[i] += -2.0 * m
                gr += -2.0 * m
            m = ys[i] - r
            if m < 0:
                viol += m * m
                gy[i] += 2.0 * m
                gr += -2.0 * m
            m = H - ys[i] - r
            if m < 0:
                viol += m * m
                gy[i] += -2.0 * m
                gr += -2.0 * m
        # non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
                d = np.hypot(dx, dy)
                m = d - 2.0 * r
                if m < 0:
                    viol += m * m
                    if d > 1e-9:
                        ux = dx / d; uy = dy / d
                    else:
                        ux = 1.0; uy = 0.0
                    gx[i] += 2.0 * m * ux
                    gx[j] += -2.0 * m * ux
                    gy[i] += 2.0 * m * uy
                    gy[j] += -2.0 * m * uy
                    gr += -4.0 * m
        # F = N*r - lam*viol ; ascent
        dF_dr = N - lam * gr
        dF_dx = -lam * gx
        dF_dy = -lam * gy
        xs += step_p * dF_dx
        ys += step_p * dF_dy
        r += step_r * dF_dr
        if r < 1e-4:
            r = 1e-4
        # decay
        if it % 500 == 0:
            step_p *= 0.85
            step_r *= 0.85
            lam *= 1.2
        s = N * r
        if s > best_sum:
            best_sum = s
            best = (xs.copy(), ys.copy(), r)
    return best


def _try_layout(cols, rows, aspect, seed=0):
    xs, ys, r = _optimize_clean(aspect, cols, rows, iters=2500, seed=seed)
    # build array and validate; if invalid, shrink r until valid
    arr = np.column_stack([xs, ys, np.full(N, r)])
    # ensure inside [0,W]x[0,H]
    W = 2.0 * aspect / (1.0 + aspect)
    H = 2.0 / (1.0 + aspect)
    # clamp
    r = min(r, np.min(xs), W - np.max(xs), np.min(ys), H - np.max(ys))
    if r <= 0:
        return None, 0.0
    arr[:, 2] = r
    # enforce non-overlap by reducing r
    for _ in range(50):
        ok = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
                if d < 2 * r - 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            break
        r *= 0.98
        if r < 1e-4:
            return None, 0.0
    arr[:, 2] = r
    if not _validate(arr):
        return None, 0.0
    return arr, np.sum(arr[:, 2])


def construct():
    best_arr = None
    best_val = -1.0
    # candidate layouts: (cols, rows, aspect)
    layouts = [
        (7, 3, 7.0 / 3.0),
        (7, 3, 1.2),
        (7, 3, 1.5),
        (6, 4, 1.5),
        (5, 5, 1.0),
        (4, 6, 2/3),
        (3, 7, 3/7),
    ]
    for (cols, rows, asp) in layouts:
        if time.time() > _DEADLINE - 10.0:
            break
        arr, val = _try_layout(cols, rows, asp, seed=0)
        if arr is not None and val > best_val:
            best_val = val
            best_arr = arr
    # fallback grid
    if best_arr is None or best_val <= 0:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])
        best_val = np.sum(best_arr[:, 2])
    return best_arr
