"""Circle packing in perimeter-4 rectangle: Smooth Epigraph + Granular Contact Polish."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 21
LOGSUMEXP_TAU = 20.0  # temperature for smooth max; higher = sharper


def _logsumexp(a, axis=None):
    """Numerically stable logsumexp."""
    a = np.asarray(a)
    a_max = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    return np.squeeze(out, axis=axis) if axis is not None else out


def _smooth_objective(z, tau=LOGSUMEXP_TAU, penalty_w=500.0):
    """
    Smooth objective using epigraph formulation.
    z = [x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}, w]
    We maximize sum(r) subject to:
      - boundary: r_i <= x_i, r_i <= w-x_i, r_i <= y_i, r_i <= h-y_i
      - non-overlap: (x_i-x_j)^2+(y_i-y_j)^2 >= (r_i+r_j)^2
    Smooth surrogate: use logsumexp of squared violations / tau.
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.15 or h < 0.15:
        return 1e5
    if np.any(r < 0.001):
        return 1e5

    base_obj = -np.sum(r)

    # Boundary violations (squared, sum)
    viol_left = np.maximum(r - x, 0.0)
    viol_right = np.maximum(x + r - w, 0.0)
    viol_bottom = np.maximum(r - y, 0.0)
    viol_top = np.maximum(y + r - h, 0.0)
    boundary_pen = penalty_w * (
        np.sum(viol_left ** 2) + np.sum(viol_right ** 2) +
        np.sum(viol_bottom ** 2) + np.sum(viol_top ** 2)
    )

    # Overlap: smooth max of (r_i + r_j - d_ij) over all pairs
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    d = np.sqrt(dx * dx + dy * dy + 1e-15)
    r_sum = r[iu[0]] + r[iu[1]]
    gap = d - r_sum  # positive = no overlap

    # Smooth max of (-gap) = max violation
    # Use logsumexp: max_i a_i ≈ (1/tau) log sum exp(tau * a_i)
    neg_gap = -gap  # positive when overlapping
    max_neg_gap = (1.0 / tau) * _logsumexp(tau * neg_gap)
    # This is a smooth upper bound on max overlap
    overlap_pen = penalty_w * max_neg_gap ** 2

    # Aspect ratio penalty
    aspect_pen = 0.0
    if w < 0.5:
        aspect_pen += penalty_w * (0.5 - w) ** 2
    elif w > 1.5:
        aspect_pen += penalty_w * (w - 1.5) ** 2

    return base_obj + boundary_pen + overlap_pen + aspect_pen


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _hexagonal_seed(w, h, spacing_scale=0.95):
    """Generate hexagonal lattice positions scaled to fit rectangle."""
    positions = []
    if w <= 0 or h <= 0:
        return positions
    # Hexagonal: rows offset
    row_h = spacing_scale * 0.866
    col_w = spacing_scale
    # Number of columns and rows that fit
    n_cols = max(1, int(w / col_w) + 1)
    n_rows = max(1, int(h / row_h) + 1)
    count = 0
    for row in range(n_rows):
        y_pos = (row + 0.5) * row_h
        if y_pos > h - 0.01:
            break
        offset = 0.5 * col_w if row % 2 == 1 else 0.0
        for col in range(n_cols):
            x_pos = (col + 0.5) * col_w + offset
            if x_pos > w - 0.01:
                break
            positions.append((x_pos, y_pos))
            count += 1
            if count >= N:
                break
        if count >= N:
            break
    # Pad with random if needed
    rng = np.random.default_rng(123)
    while len(positions) < N:
        positions.append((rng.uniform(0.05, w - 0.05), rng.uniform(0.05, h - 0.05)))
    return positions[:N]


def _grid_seed(w, h, n_rows, n_cols):
    """Generate grid positions."""
    positions = []
    for i in range(n_rows):
        for j in range(n_cols):
            cx = (j + 0.5) / n_cols * w
            cy = (i + 0.5) / n_rows * h
            positions.append((cx, cy))
    rng = np.random.default_rng(456)
    while len(positions) < N:
        positions.append((rng.uniform(0.05, w - 0.05), rng.uniform(0.05, h - 0.05)))
    return positions[:N]


def _get_seeds():
    """Generate diverse initial seeds."""
    seeds = []
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 1.4, 0.7]
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.25:
            continue
        z = np.zeros(3 * N + 1)
        # Try hexagonal
        pos = _hexagonal_seed(w0, h0, spacing_scale=min(w0, h0) / 4.5)
        for i, (cx, cy) in enumerate(pos):
            z[3*i] = cx
            z[3*i+1] = cy
        z[-1] = w0
        # Initial radius estimate
        r_est = min(w0 / 5, h0 / 5) * 0.5
        for i in range(N):
            z[3*i+2] = max(0.02, min(r_est, min(z[3*i], w0-z[3*i], z[3*i+1], h0-z[3*i+1]) * 0.8))
        seeds.append(z)

    # Grid seeds
    for w0 in [1.0, 0.8, 1.2, 0.7, 1.4]:
        h0 = 2.0 - w0
        if h0 < 0.25:
            continue
        z = np.zeros(3 * N + 1)
        pos = _grid_seed(w0, h0, 5, 5)
        for i, (cx, cy) in enumerate(pos):
            z[3*i] = cx
            z[3*i+1] = cy
        z[-1] = w0
        r_est = min(w0 / 5, h0 / 5) * 0.5
        for i in range(N):
            z[3*i+2] = max(0.02, min(r_est, min(z[3*i], w0-z[3*i], z[3*i+1], h0-z[3*i+1]) * 0.8))
        seeds.append(z)

    # Random seeds
    rng = np.random.default_rng(789)
    for _ in range(8):
        w0 = rng.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        z = np.zeros(3 * N + 1)
        for i in range(N):
            r0 = rng.uniform(0.03, 0.10)
            cx = rng.uniform(r0, w0 - r0)
            cy = rng.uniform(r0, h0 - r0)
            z[3*i] = cx
            z[3*i+1] = cy
            z[3*i+2] = r0
        z[-1] = w0
        seeds.append(z)

    return seeds


def _force_resolve(z, max_iter=800):
    """Force-directed relaxation to resolve overlaps and boundary violations."""
    z = z.copy().astype(float)
    n = N
    dt = 0.003
    for it in range(max_iter):
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w
        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall forces
        fx -= np.maximum(r - x, 0.0) * 80
        fx += np.maximum(x + r - w, 0.0) * 80
        fy -= np.maximum(r - y, 0.0) * 80
        fy += np.maximum(y + r - h, 0.0) * 80

        # Pairwise repulsion
        iu = np.triu_indices(n, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d = np.sqrt(dx*dx + dy*dy + 1e-15)
        r_sum = r[iu[0]] + r[iu[1]]
        overlap = np.maximum(r_sum - d, 0.0)
        max_ov = float(np.max(overlap)) if overlap.size > 0 else 0.0
        if max_ov > 1e-10:
            f_mag = overlap * 12.0
            ux = dx / d
            uy = dy / d
            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_mag[idx]
                fy[i_idx] += uy[idx] * f_mag[idx]
                fx[j_idx] -= ux[idx] * f_mag[idx]
                fy[j_idx] -= uy[idx] * f_mag[idx]

        if max_ov < 1e-9:
            break

        # Update
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy
        for i in range(n):
            z[3*i] = np.clip(z[3*i], r[i], w - r[i])
            z[3*i+1] = np.clip(z[3*i+1], r[i], h - r[i])

    return z


def _granular_contact_polish(x, y, r, w, h, time_budget):
    """
    Granular polish: identify contact graph, then do coordinated micro-adjustments
    along the contact manifold to create room for radius growth.
    """
    n = N
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    start = time.time()

    for it in range(5000):
        if time.time() > time_budget:
            break

        # Compute all pairwise gaps
        iu = np.triu_indices(n, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d = np.sqrt(dx*dx + dy*dy + 1e-15)
        r_sum = r[iu[0]] + r[iu[1]]
        gap = d - r_sum  # >0 means separated, <0 means overlapping

        # Identify contacts: gap < 0.005
        contact_mask = gap < 0.005
        if not np.any(contact_mask):
            break

        # Build contact graph forces: push apart along contact normals
        fx = np.zeros(n)
        fy = np.zeros(n)
        for idx in np.where(contact_mask)[0]:
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            d_val = d[idx]
            if d_val < 1e-12:
                continue
            nx = dx[idx] / d_val
            ny = dy[idx] / d_val
            push = 0.002  # small push
            fx[i_idx] += nx * push
            fy[i_idx] += ny * push
            fx[j_idx] -= nx * push
            fy[j_idx] -= ny * push

        # Wall constraints: gentle pull away from walls
        fx += np.maximum(r - x, 0.0) * 20
        fx -= np.maximum(x + r - w, 0.0) * 20
        fy += np.maximum(r - y, 0.0) * 20
        fy -= np.maximum(y + r - h, 0.0) * 20

        # Apply small step
        step = 0.001
        x += step * fx
        y += step * fy
        # Clamp
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Now try to grow radii: binary search on uniform scale factor
    if time.time() < time_budget:
        lo, hi = 1.0, 1.15
        for _ in range(40):
            if time.time() > time_budget:
                break
            mid = (lo + hi) / 2.0
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        r = r * lo

    # Per-circle expansion
    if time.time() < time_budget:
        for i in range(n):
            if time.time() > time_budget:
                break
            # Max radius for circle i
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_val = np.sqrt(dx*dx + dy*dy)
                r_allowed = d_val - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-8:
                r[i] = r_max

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return x, y, r  # return best effort


def _sweep_aspect(z, time_budget):
    """
    Sweep over aspect ratio w, running a quick optimization at each.
    Returns the best z found.
    """
    best_z = z.copy()
    best_val = -np.sum(z[2::3][:N])
    start = time.time()

    w_candidates = np.linspace(0.5, 1.5, 21)
    # Sort by proximity to current w
    w_cur = z[-1]
    w_candidates = w_candidates[np.argsort(np.abs(w_candidates - w_cur))]

    for w_try in w_candidates:
        if time.time() > time_budget:
            break
        h_try = 2.0 - w_try
        if h_try < 0.2:
            continue

        # Scale positions to new rectangle
        z_try = z.copy()
        z_try[-1] = w_try
        # Rescale x and y proportionally
        scale_x = w_try / z[-1] if z[-1] > 0.01 else 1.0
        scale_y = h_try / (2.0 - z[-1]) if (2.0 - z[-1]) > 0.01 else 1.0
        z_try[0::3][:N] *= scale_x
        z_try[1::3][:N] *= scale_y
        z_try[2::3][:N] *= min(scale_x, scale_y) * 0.9  # shrink radii to avoid overlap

        # Quick force resolve
        z_try = _force_resolve(z_try, max_iter=200)

        # Quick SLSQP
        remaining = time_budget - (time.time() - start)
        if remaining < 5:
            break
        maxiter = int(min(200, remaining * 10))
        try:
            res = minimize(lambda zz: _smooth_objective(zz, tau=LOGSUMEXP_TAU),
                           z_try, method='SLSQP',
                           bounds=[(0, 2)] * (3*N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 50})
            z_opt = res.x
            x = z_opt[0::3][:N]
            y = z_opt[1::3][:N]
            r = z_opt[2::3][:N]
            w = z_opt[-1]
            h = 2.0 - w
            if _is_valid(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_val:
                    best_val = val
                    best_z = z_opt.copy()
        except Exception:
            continue

    return best_z


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    best_value = 0.0
    best_result = None

    def check_and_update(z):
        nonlocal best_value, best_result
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = z.copy()
            return val
        return 0.0

    seeds = _get_seeds()

    # Phase 1: Multi-start SLSQP with smooth objective
    for i, z0 in enumerate(seeds):
        if time.time() > deadline - 40:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 8:
            break

        # Force resolve first
        z0_res = _force_resolve(z0, max_iter=400)

        maxiter = int(min(500, max(100, remaining * 30)))
        try:
            res = minimize(lambda zz: _smooth_objective(zz, tau=LOGSUMEXP_TAU),
                           z0_res, method='SLSQP',
                           bounds=[(0, 2)] * (3*N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
            z_opt = res.x
            check_and_update(z_opt)
        except Exception:
            continue

    # Phase 2: Aspect ratio sweep on best result
    if best_result is not None and time.time() < deadline - 30:
        remaining = deadline - (time.time() - start_time) - 20
        if remaining > 10:
            swept = _sweep_aspect(best_result, remaining)
            check_and_update(swept)

    # Phase 3: Granular contact polish
    if best_result is not None and time.time() < deadline - 20:
        remaining = deadline - (time.time() - start_time) - 10
        if remaining > 5:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w
            x_new, y_new, r_new = _granular_contact_polish(x, y, r, w, h, remaining)
            z_new = np.empty(3*N + 1)
            z_new[0::3][:N] = x_new
            z_new[1::3][:N] = y_new
            z_new[2::3][:N] = r_new
            z_new[-1] = w
            check_and_update(z_new)

    # Phase 4: High-precision SLSQP refinement on best
    if best_result is not None and time.time() < deadline - 10:
        remaining = deadline - (time.time() - start_time) - 5
        if remaining > 5:
            # Use lower temperature for sharper objective
            try:
                res = minimize(lambda zz: _smooth_objective(zz, tau=LOGSUMEXP_TAU * 3),
                               best_result, method='SLSQP',
                               bounds=[(0, 2)] * (3*N) + [(0.3, 1.7)],
                               options={'maxiter': int(min(400, remaining * 50)), 'ftol': 1e-14, 'maxls': 100})
                z_opt = res.x
                check_and_update(z_opt)
            except Exception:
                pass

    # Phase 5: Final granular polish
    if best_result is not None and time.time() < deadline - 3:
        remaining = deadline - (time.time() - start_time)
        if remaining > 2:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w
            x_new, y_new, r_new = _granular_contact_polish(x, y, r, w, h, remaining)
            z_new = np.empty(3*N + 1)
            z_new[0::3][:N] = x_new
            z_new[1::3][:N] = y_new
            z_new[2::3][:N] = r_new
            z_new[-1] = w
            check_and_update(z_new)

    # Fallback
    if best_result is None:
        w = 1.0
        h = 1.0
        z = np.zeros(3*N + 1)
        pos = _grid_seed(w, h, 5, 5)
        for i, (cx, cy) in enumerate(pos):
            z[3*i] = cx
            z[3*i+1] = cy
            z[3*i+2] = 0.08
        z[-1] = w
        return np.column_stack((z[0::3][:N], z[1::3][:N], z[2::3][:N]))

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Safety shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
