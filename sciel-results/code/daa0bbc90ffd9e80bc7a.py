"""Circle packing in perimeter-4 rectangle: Pebble Inflation constructive algorithm."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    best_value = 0.0
    best_output = None

    # Try multiple aspect ratios and initial configurations
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.65, 1.6, 0.6, 1.8, 0.55]
    rng = np.random.default_rng(12345)

    for w0 in aspect_ratios:
        if time.time() > deadline - 15:
            break
        h0 = 2.0 - w0
        if h0 < 0.3 or w0 < 0.3:
            continue

        # Generate initial positions: grid + jitter
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)

        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side * w0 + rng.uniform(-0.02, 0.02)
            y[i] = (row + 0.5) / side * h0 + rng.uniform(-0.02, 0.02)
            r[i] = 0.02  # start small

        # Clamp to rectangle
        x = np.clip(x, 0.01, w0 - 0.01)
        y = np.clip(y, 0.01, h0 - 0.01)

        # Pebble inflation: iteratively grow each circle to max possible radius
        result = _pebble_inflate(x, y, r, w0, h0, deadline)

        if result is not None:
            x_res, y_res, r_res, w_res = result
            val = float(np.sum(r_res))
            if val > best_value:
                best_value = val
                best_output = np.column_stack((x_res, y_res, r_res))

        # Also try random initial positions
        if time.time() > deadline - 15:
            break
        x_rand = rng.uniform(0.05, w0 - 0.05, size=N)
        y_rand = rng.uniform(0.05, h0 - 0.05, size=N)
        r_rand = np.full(N, 0.02)
        result2 = _pebble_inflate(x_rand, y_rand, r_rand, w0, h0, deadline)
        if result2 is not None:
            x_res, y_res, r_res, w_res = result2
            val = float(np.sum(r_res))
            if val > best_value:
                best_value = val
                best_output = np.column_stack((x_res, y_res, r_res))

    # Polish best result with SLSQP if time remains
    if best_output is not None and time.time() < deadline - 10:
        polished = _slsqp_polish(best_output, deadline - (time.time() - start_time))
        if polished is not None:
            val = float(np.sum(polished[:, 2]))
            if val > best_value:
                best_value = val
                best_output = polished

    if best_output is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cells.append([(col + 0.5) / side, (row + 0.5) / side, 0.5 / side - 1e-3])
        return np.array(cells)

    # Safety shrink
    best_output = best_output.copy()
    best_output[:, 2] *= (1 - 1e-6)
    return best_output


def _max_radius_for_circle(cx, cy, x, y, r, W, H, idx):
    """Compute maximum radius for circle at (cx, cy) given other circles and walls."""
    # Wall constraints
    r_wall = min(cx, W - cx, cy, H - cy)
    if r_wall <= 0:
        return 0.0

    r_max = r_wall

    # Circle-circle constraints
    for j in range(N):
        if j == idx:
            continue
        dx = cx - x[j]
        dy = cy - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        if d > 1e-12:
            # We need r + r[j] <= d, so r <= d - r[j]
            r_candidate = d - r[j]
            if r_candidate < r_max:
                r_max = r_candidate

    return max(r_max, 0.0)


def _pebble_inflate(x, y, r, W, H, deadline):
    """
    Pebble inflation: iteratively grow each circle to its maximum allowed radius.
    Multiple passes to reach a better local optimum.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)

    max_passes = 50
    for p in range(max_passes):
        if time.time() > deadline - 10:
            break

        improved = False
        # Process circles in random order to avoid bias
        order = np.random.permutation(N)
        for idx in order:
            # Compute max radius at current position
            r_max = _max_radius_for_circle(x[idx], y[idx], x, y, r, W, H, idx)
            if r_max <= 0:
                continue

            # Try to grow: new radius is min(current + growth, r_max)
            growth = 0.005
            r_new = min(r[idx] + growth, r_max)

            if r_new <= r[idx] + 1e-12:
                continue

            # Check if growing is valid (it should be by construction, but verify)
            r[idx] = r_new
            improved = True

        # After growing all, try to reposition circles to allow further growth
        if not improved and p > 5:
            break

        # Reposition pass: move each circle to a position that allows larger radius
        if p % 3 == 0:
            for idx in range(N):
                if time.time() > deadline - 10:
                    break
                # Try moving to several candidate positions
                best_r = r[idx]
                best_x, best_y = x[idx], y[idx]

                # Try positions at various offsets
                for dx in [-0.03, -0.01, 0.0, 0.01, 0.03]:
                    for dy in [-0.03, -0.01, 0.0, 0.01, 0.03]:
                        nx = x[idx] + dx
                        ny = y[idx] + dy
                        if nx < 0.01 or nx > W - 0.01 or ny < 0.01 or ny > H - 0.01:
                            continue
                        r_cand = _max_radius_for_circle(nx, ny, x, y, r, W, H, idx)
                        if r_cand > best_r + 1e-6:
                            best_r = r_cand
                            best_x, best_y = nx, ny

                if best_r > r[idx] + 1e-6:
                    x[idx], y[idx], r[idx] = best_x, best_y, best_r

    # Final validation
    if not _is_valid(x, y, r, W, H):
        # Try to repair by shrinking
        scale = 0.95
        while not _is_valid(x, y, r * scale, W, H) and scale > 0.5:
            scale -= 0.01
        if scale < 0.5:
            return None
        r *= scale

    return x, y, r, W


def _slsqp_polish(output, time_budget):
    """SLSQP polish on a valid packing to squeeze out more."""
    n = N
    x0 = output[:, 0].copy()
    y0 = output[:, 1].copy()
    r0 = output[:, 2].copy()

    # Estimate aspect ratio from bounding box
    x_min, x_max = np.min(x0 - r0), np.max(x0 + r0)
    y_min, y_max = np.min(y0 - r0), np.max(y0 + r0)
    W0 = x_max - x_min
    H0 = y_max - y_min

    # Normalize to perimeter 4
    perim = 2 * (W0 + H0)
    if perim <= 0:
        return None
    scale = 4.0 / perim
    W0 *= scale
    H0 *= scale

    # Shift to origin
    x0 = x0 - x_min
    y0 = y0 - y_min

    # Parameter vector: [x1, y1, r1, ..., xN, yN, rN, w]
    z = np.zeros(3 * N + 1)
    z[0::3][:n] = x0
    z[1::3][:n] = y0
    z[2::3][:n] = r0
    z[-1] = W0

    penalty_weight = 500.0

    def obj(z):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2:
            return 1e6

        val = -np.sum(r)

        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                dist_req = r[i] + r[j]
                if d < dist_req:
                    val += penalty_weight * (dist_req - d) ** 2

        for i in range(n):
            if x[i] < r[i]:
                val += penalty_weight * (r[i] - x[i]) ** 2
            elif x[i] > w - r[i]:
                val += penalty_weight * (x[i] - (w - r[i])) ** 2
            if y[i] < r[i]:
                val += penalty_weight * (r[i] - y[i]) ** 2
            elif y[i] > h - r[i]:
                val += penalty_weight * (y[i] - (h - r[i])) ** 2

        return val

    bounds = []
    for i in range(n):
        bounds.append((max(0, x0[i] - 0.05), min(W0, x0[i] + 0.05)))
        bounds.append((max(0, y0[i] - 0.05), min(H0, y0[i] + 0.05)))
        bounds.append((max(1e-4, r0[i] * 0.95), r0[i] * 1.05))
    bounds.append((max(0.2, W0 - 0.05), min(1.8, W0 + 0.05)))

    maxiter = int(min(200, time_budget * 20))
    try:
        res = minimize(obj, z, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
        z_opt = res.x

        x_opt = z_opt[0::3][:n]
        y_opt = z_opt[1::3][:n]
        r_opt = z_opt[2::3][:n]
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt

        if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
            return np.column_stack((x_opt, y_opt, r_opt))
    except Exception:
        pass

    return None


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(r <= 0):
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
