"""Circle packing in perimeter-4 rectangle: L-BFGS-B + constraint repair + greedy expansion."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(x, y, r, w, h, tol=1e-7):
    """Check if packing is valid."""
    if w <= 0.05 or h <= 0.05:
        return False
    if np.any(r <= 0):
        return False
    if np.any(x < r + tol) or np.any(x > w - r - tol):
        return False
    if np.any(y < r + tol) or np.any(y > h - r - tol):
        return False
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    d2 = dx * dx + dy * dy
    r_sum = r[iu[0]] + r[iu[1]]
    if np.any(d2 < (r_sum + tol) ** 2):
        return False
    return True


def _max_radius_for_circle(i, x, y, r, w, h):
    """Compute maximum radius for circle i given current positions."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _expansion_polish(x, y, r, w, h, time_budget=5.0):
    """Iterative expansion with local position optimization."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    start = time.time()

    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    steps = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]

    for outer in range(50):
        if time.time() - start > time_budget:
            break

        improved = False
        order = np.argsort(r)

        for i in order:
            if time.time() - start > time_budget:
                break

            # Try to expand radius in place
            r_max = _max_radius_for_circle(i, x, y, r, w, h)
            if r_max > r[i] + 1e-8:
                lo = r[i]
                hi = r_max
                for _ in range(15):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
                    improved = True
                    continue

            # Try local position search to enable larger radius
            best_move = None
            best_gain = 1e-6

            for angle in angles:
                for step in steps:
                    nx = x[i] + step * np.cos(angle)
                    ny = y[i] + step * np.sin(angle)

                    if nx < r[i] - 1e-9 or nx > w - r[i] + 1e-9:
                        continue
                    if ny < r[i] - 1e-9 or ny > h - r[i] + 1e-9:
                        continue

                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        dx = nx - x[j]
                        dy = ny - y[j]
                        d2 = dx * dx + dy * dy
                        rs = r[i] + r[j]
                        if d2 < rs * rs - 1e-9:
                            valid = False
                            break
                    if not valid:
                        continue

                    r_wall_new = min(nx, w - nx, ny, h - ny)
                    r_max_new = r_wall_new
                    for j in range(N):
                        if i == j:
                            continue
                        dx = nx - x[j]
                        dy = ny - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        r_allowed = d - r[j]
                        if r_allowed < r_max_new:
                            r_max_new = r_allowed
                    r_max_new = max(0.0, r_max_new)

                    gain = r_max_new - r[i]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (nx, ny, r_max_new)

            if best_move is not None:
                x[i] = best_move[0]
                y[i] = best_move[1]
                r[i] = best_move[2]
                improved = True

        if not improved:
            break

    return x, y, r


def _lbfgs_optimize(x, y, r, w, h, time_budget=10.0):
    """Optimize using L-BFGS-B with soft constraints."""
    start = time.time()
    
    def to_vec(x, y, r, w):
        v = np.zeros(3 * N + 1)
        v[0::3][:N] = x
        v[1::3][:N] = y
        v[2::3][:N] = r
        v[-1] = w
        return v

    def from_vec(v):
        x = v[0::3][:N].copy()
        y = v[1::3][:N].copy()
        r = v[2::3][:N].copy()
        w = v[-1]
        return x, y, r, w

    def objective(v):
        x, y, r, w = from_vec(v)
        h = 2.0 - w
        # Soft penalty for violations
        penalty = 0.0
        
        # Wall violations
        penalty += 100.0 * np.sum(np.maximum(r - x, 0.0)**2)
        penalty += 100.0 * np.sum(np.maximum(x + r - w, 0.0)**2)
        penalty += 100.0 * np.sum(np.maximum(r - y, 0.0)**2)
        penalty += 100.0 * np.sum(np.maximum(y + r - h, 0.0)**2)
        
        # Overlap violations
        iu = np.triu_indices(N, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d = np.sqrt(dx*dx + dy*dy + 1e-12)
        r_sum = r[iu[0]] + r[iu[1]]
        overlap = np.maximum(r_sum - d, 0.0)
        penalty += 100.0 * np.sum(overlap**2)
        
        return -np.sum(r) + penalty

    v0 = to_vec(x, y, r, w)
    bounds = []
    for i in range(N):
        bounds.append((1e-4, 1.0))  # x
        bounds.append((1e-4, 1.0))  # y
        bounds.append((1e-4, 1.0))  # r
    bounds.append((0.3, 1.7))  # w

    maxiter = int(min(1000, time_budget * 100))
    try:
        res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
        v_opt = res.x
        xo, yo, ro, wo = from_vec(v_opt)
        ho = 2.0 - wo
        if _is_valid(xo, yo, ro, wo, ho):
            return xo, yo, ro, wo
        # Try to repair
        return None
    except Exception:
        return None


def _force_relax(x, y, r, w, h, max_iter=2000):
    """Force-directed relaxation to resolve overlaps."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    iu = np.triu_indices(N, k=1)

    for it in range(max_iter):
        fx = np.zeros(N)
        fy = np.zeros(N)

        # Wall repulsion
        fx -= np.maximum(r - x, 0.0) * 50
        fx += np.maximum(x + r - w, 0.0) * 50
        fy -= np.maximum(r - y, 0.0) * 50
        fy += np.maximum(y + r - h, 0.0) * 50

        # Pairwise repulsion
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d2 = dx * dx + dy * dy
        r_sum = r[iu[0]] + r[iu[1]]
        d = np.sqrt(np.maximum(d2, 1e-12))
        overlap = np.maximum(r_sum - d, 0.0)

        if np.max(overlap) < 1e-10:
            break

        f_pairs = overlap * 10.0
        ux = dx / d
        uy = dy / d

        for k in range(len(iu[0])):
            i = iu[0][k]
            j = iu[1][k]
            fx[i] += ux[k] * f_pairs[k]
            fy[i] += uy[k] * f_pairs[k]
            fx[j] -= ux[k] * f_pairs[k]
            fy[j] -= uy[k] * f_pairs[k]

        max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)), 1e-8)
        dt = 0.005 if max_force > 0.1 else 0.002

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y, r


def _get_fallback():
    """Generate a valid fallback packing."""
    w = 1.0
    h = 1.0
    side = 5
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    for i in range(N):
        row, col = divmod(i, side)
        x[i] = (col + 0.5) / side
        y[i] = (row + 0.5) / side
        r[i] = 0.09
    if not _is_valid(x, y, r, w, h, tol=1e-6):
        r = r * 0.9
    return np.column_stack((x, y, r))


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_value = 0.0
    best_result = None

    def get_initial_guesses():
        guesses = []
        side = 5

        # Various aspect ratios
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.3, 0.75]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                x[i] = (col + 0.5) / side * w0
                y[i] = (row + 0.5) / side * h0
                r[i] = min(w0 / side, h0 / side) * 0.4
            guesses.append((x, y, r, w0))

        # Random starts
        for seed_val in range(10):
            rng = np.random.default_rng(seed_val * 1000 + 42)
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.10)
                x[i] = rng.uniform(r0, w0 - r0)
                y[i] = rng.uniform(r0, h0 - r0)
                r[i] = r0
            guesses.append((x, y, r, w0))

        return guesses

    guesses = get_initial_guesses()

    for idx, (x0, y0, r0, w0) in enumerate(guesses):
        if time.time() > deadline - 25:
            break

        h0 = 2.0 - w0

        # Step 1: Force relax
        x, y, r = _force_relax(x0, y0, r0, w0, h0, max_iter=800)

        # Step 2: L-BFGS-B optimization
        result = _lbfgs_optimize(x, y, r, w0, h0, time_budget=min(8.0, deadline - time.time() - 15))
        if result is not None:
            x, y, r, w = result
            h = 2.0 - w
        else:
            # Fall back to force relax result
            x, y, r = _force_relax(x0, y0, r0, w0, h0, max_iter=1000)
            w = w0
            h = h0

        # Step 3: Expansion polish
        x, y, r = _expansion_polish(x, y, r, w, h, time_budget=min(6.0, deadline - time.time() - 12))

        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                z = np.empty(3 * N + 1)
                z[0::3][:N] = x
                z[1::3][:N] = y
                z[2::3][:N] = r
                z[-1] = w
                best_result = z.copy()

    # Phase 2: Polish best result
    if best_result is not None and time.time() < deadline - 10:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        x, y, r = _expansion_polish(x, y, r, w, h, time_budget=min(10.0, deadline - time.time() - 8))
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                z = np.empty(3 * N + 1)
                z[0::3][:N] = x
                z[1::3][:N] = y
                z[2::3][:N] = r
                z[-1] = w
                best_result = z.copy()

    # Phase 3: Try scaling to different aspect ratios
    if best_result is not None and time.time() < deadline - 8:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w_orig = best_result[-1]
        h_orig = 2.0 - w_orig

        for w_try in [w_orig * 0.9, w_orig * 1.1, w_orig * 0.85, w_orig * 1.15, w_orig * 0.95, w_orig * 1.05]:
            if time.time() > deadline - 5:
                break
            if w_try < 0.3 or w_try > 1.7:
                continue
            h_try = 2.0 - w_try

            sx = w_try / w_orig
            sy = h_try / h_orig
            xs = x * sx
            ys = y * sy
            rs = r * min(sx, sy)

            xs, ys, rs = _force_relax(xs, ys, rs, w_try, h_try, max_iter=500)
            xs, ys, rs = _expansion_polish(xs, ys, rs, w_try, h_try, time_budget=min(3.0, deadline - time.time() - 3))

            if _is_valid(xs, ys, rs, w_try, h_try):
                val = float(np.sum(rs))
                if val > best_value:
                    best_value = val
                    z = np.empty(3 * N + 1)
                    z[0::3][:N] = xs
                    z[1::3][:N] = ys
                    z[2::3][:N] = rs
                    z[-1] = w_try
                    best_result = z.copy()

    if best_result is None:
        return _get_fallback()

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w_final = best_result[-1]
    h_final = 2.0 - w_final

    # Final safety: shrink radii to ensure strict validity
    lo, hi = 0.8, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2.0
        r_trial = r * mid
        if _is_valid(x, y, r_trial, w_final, h_final, tol=1e-6):
            lo = mid
        else:
            hi = mid

    r_final = r * lo
    if not _is_valid(x, y, r_final, w_final, h_final, tol=1e-6):
        return _get_fallback()

    return np.column_stack((x, y, r_final))
