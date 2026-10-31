"""Circle packing in perimeter-4 rectangle: SLSQP + vectorized forces + fine grid + polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 140.0

def _compute_distances(x, y):
    """Vectorized pairwise distance matrix."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.sqrt(dx * dx + dy * dy)

def _objective_fast(z):
    """Fast objective with smooth soft-min surrogate for overlap/boundary penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.3 or h < 0.3:
        return 1e6

    obj = -np.sum(r)
    pw = 500.0
    beta = 50.0  # soft-min temperature

    # Vectorized pairwise overlap penalty using soft-min
    d = _compute_distances(x, y)
    r_sum = r[:, None] + r[None, :]
    # gap = d - r_sum; negative means overlap
    gap = d - r_sum
    # Soft-min of gaps: approximate min via log-sum-exp
    # For overlap penalty, we want to penalize when gap < 0
    # Use: -softmin(gap) as penalty when gaps are negative
    # softmin(gap) ≈ -(1/beta) * log(sum(exp(-beta * gap)))
    # We only care about pairs where gap < 0 (overlap)
    triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    gap_triu = gap[triu_mask]
    # Soft-min approximation: for negative values, penalize smoothly
    # Use quadratic on clipped soft value
    # Simpler: use smooth approximation of max(0, -gap)^2
    # max(0, -gap) ≈ softplus(-gap) = log(1 + exp(-gap)) / 1
    # But for SLSQP, quadratic on softplus is smoother
    soft_overlap = np.log1p(np.exp(np.clip(-gap_triu, -50, 50)))
    obj += pw * np.sum(soft_overlap ** 2)

    # Boundary penalties (smooth)
    bl_x = np.maximum(0, r - x)
    br_x = np.maximum(0, x - (w - r))
    bl_y = np.maximum(0, r - y)
    br_y = np.maximum(0, y - (h - r))
    # Use smooth approximation
    soft_bl_x = np.log1p(np.exp(np.clip(-(x - r), -50, 50)))
    soft_br_x = np.log1p(np.exp(np.clip(-(w - r - x), -50, 50)))
    soft_bl_y = np.log1p(np.exp(np.clip(-(y - r), -50, 50)))
    soft_br_y = np.log1p(np.exp(np.clip(-(h - r - y), -50, 50)))
    obj += pw * (np.sum(soft_bl_x ** 2) + np.sum(soft_br_x ** 2) +
                 np.sum(soft_bl_y ** 2) + np.sum(soft_br_y ** 2))

    # Aspect ratio penalty
    if w < 0.5:
        obj += pw * (0.5 - w) ** 2
    elif w > 1.5:
        obj += pw * (w - 1.5) ** 2

    return obj

def _check_validity(z, tol=1e-9):
    """Check if packing is valid, return (valid, value)."""
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

    d = _compute_distances(x, y)
    r_sum = r[:, None] + r[None, :]
    triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d[triu_mask] < r_sum[triu_mask] - tol):
        return False, 0.0

    val = float(np.sum(r))
    return True, val

def _force_relax(z, max_iter=300, dt=0.003):
    """Force-directed relaxation to resolve overlaps (fully vectorized)."""
    z = z.copy().astype(float)
    n = N

    for _ in range(max_iter):
        if time.time() > DEADLINE - 3:
            break
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion (vectorized)
        fx += np.where(x - r < 0, (x - r) * 20, 0)
        fx -= np.where(x + r > w, (x + r - w) * 20, 0)
        fy += np.where(y - r < 0, (y - r) * 20, 0)
        fy -= np.where(y + r > h, (y + r - h) * 20, 0)

        # Pairwise repulsion (fully vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, 1.0)  # avoid div by zero
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(0, r_sum - d)
        # Force direction: unit vector from j to i
        ux = dx / d
        uy = dy / d
        # Each pair contributes to both i and j; use overlap * force_mag
        force_mag = overlap * 10
        fx += np.sum(ux * force_mag, axis=1)
        fy += np.sum(uy * force_mag, axis=1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        z[0::3][:n] = np.clip(z[0::3][:n], r, w - r)
        z[1::3][:n] = np.clip(z[1::3][:n], r, h - r)

    return z

def _polish_result(z, time_budget):
    """Iterative radius growth polish with vectorized position adjustments."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    start = time.time()
    max_iter = 5000
    dt_pos = 0.001

    for it in range(max_iter):
        if time.time() - start > time_budget:
            break

        # Position adjustment for overlaps (vectorized)
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        fx += np.where(x - r < 0, (x - r) * 30, 0)
        fx -= np.where(x + r > w, (x + r - w) * 30, 0)
        fy += np.where(y - r < 0, (y - r) * 30, 0)
        fy -= np.where(y + r > h, (y + r - h) * 30, 0)

        # Pairwise repulsion (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, 1.0)
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(0, r_sum - d)
        ux = dx / d
        uy = dy / d
        force_mag = overlap * 15
        fx += np.sum(ux * force_mag, axis=1)
        fy += np.sum(uy * force_mag, axis=1)

        x += dt_pos * fx
        y += dt_pos * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        # Try to grow each circle (vectorized)
        d = _compute_distances(x, y)
        np.fill_diagonal(d, 1e10)
        # For each circle i, max growth is min of:
        #   distance to each wall, and (d[i,j] - r[i] - r[j]) for all j
        wall_growth = np.minimum(
            np.minimum(x - r, w - x - r),
            np.minimum(y - r, h - y - r)
        )
        # Pairwise constraint: d[i,j] - r[i] - r[j]
        pair_growth = d - r[:, None] - r[None, :]
        max_growth = np.minimum(wall_growth[:, None], pair_growth)
        max_growth = np.min(max_growth, axis=1)
        
        growable = max_growth > 1e-8
        if np.any(growable):
            growth = np.minimum(0.003, max_growth[growable] * 0.5)
            r[growable] += growth

    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    return z

def _fine_grid_start(w0, h0, grid_res=60):
    """Create initial guess from fine grid sampling."""
    # Place circles on a fine grid, pick best N positions
    positions = []
    step_x = w0 / grid_res
    step_y = h0 / grid_res
    r0 = min(step_x, step_y) * 0.48

    for i in range(grid_res):
        for j in range(grid_res):
            cx = (i + 0.5) * step_x
            cy = (j + 0.5) * step_y
            positions.append((cx, cy))

    # Greedy selection: pick positions that maximize min distance
    selected = []
    remaining = list(range(len(positions)))
    rng = np.random.default_rng(123)
    rng.shuffle(remaining)

    # Start with a few random picks spread out
    for _ in range(min(5, N)):
        idx = remaining.pop(0)
        selected.append(positions[idx])

    while len(selected) < N and remaining:
        best_idx = -1
        best_min_dist = -1
        # Sample a subset for speed
        sample = remaining[:min(200, len(remaining))]
        for idx in sample:
            px, py = positions[idx]
            min_d = min(np.hypot(px - sx, py - sy) for sx, sy in selected)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        if best_idx >= 0:
            remaining.remove(best_idx)
            selected.append(positions[idx])
        else:
            break

    z = []
    for cx, cy in selected[:N]:
        z.extend([cx, cy, r0])
    z.append(w0)
    return np.array(z)

def _high_precision_polish(z, time_budget):
    """High precision polish: SLSQP + coordinate-descent radius maximization."""
    z = z.copy().astype(float)
    n = N

    start = time.time()
    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]

    # SLSQP passes
    for pass_num in range(4):
        if time.time() - start > time_budget - 3:
            break

        remaining = time_budget - (time.time() - start)
        maxiter = int(min(150, remaining * 20))
        if maxiter < 10:
            maxiter = 10

        try:
            res = minimize(_objective_fast, z, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-15, 'maxls': 30})
            z_opt = res.x
            valid, val = _check_validity(z_opt)
            if valid:
                z = z_opt.copy()
        except Exception:
            break

    # Coordinate-descent radius maximization
    # For each circle, maximize its radius given all other positions/radii fixed
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    cd_start = time.time()
    for cd_iter in range(50):
        if time.time() - cd_start > time_budget - 1:
            break
        improved = False
        d = _compute_distances(x, y)
        for i in range(n):
            # Max radius for circle i given constraints
            # Wall constraints:
            max_r = min(x[i], w - x[i], y[i], h - y[i])
            # Pairwise constraints: r_i <= d[i,j] - r[j]
            for j in range(n):
                if i != j:
                    max_r = min(max_r, d[i, j] - r[j])
            max_r = max(max_r, 0.0)
            if max_r > r[i] + 1e-10:
                r[i] = max_r
                improved = True
        if not improved:
            break

    z[2::3][:n] = r
    return z

def construct():
    best_value = 0.0
    best_result = None

    # Generate diverse initial guesses
    guesses = []

    # Fine grid starts with varying aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.65, 1.4, 0.6]
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.35:
            continue
        z = _fine_grid_start(w0, h0, grid_res=60)
        guesses.append(z)

    # Standard grid starts
    side = int(np.ceil(np.sqrt(N)))
    for w0 in [1.0, 0.85, 1.18, 0.75, 1.33, 0.67]:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        z = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side * w0
            cy = (row + 0.5) / side * h0
            r0 = min(w0 / side, h0 / side) * 0.45
            z.extend([cx, cy, r0])
        z.append(w0)
        guesses.append(np.array(z))

    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(6):
        w0 = rng.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        z = []
        for i in range(N):
            cx = rng.uniform(0.05, w0 - 0.05)
            cy = rng.uniform(0.05, h0 - 0.05)
            r0 = rng.uniform(0.04, 0.10)
            z.extend([cx, cy, r0])
        z.append(w0)
        guesses.append(np.array(z))

    # Shuffle to interleave types
    rng.shuffle(guesses)

    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]

    for i, z0 in enumerate(guesses):
        elapsed = time.time() - START
        if elapsed > DEADLINE - 8:
            break

        remaining = DEADLINE - elapsed - 5
        if remaining < 2:
            break

        # Force-directed relaxation
        z0_relaxed = _force_relax(z0, max_iter=200)

        # SLSQP optimization
        maxiter = int(min(200, remaining * 25))
        if maxiter < 10:
            maxiter = 10
        try:
            res = minimize(_objective_fast, z0_relaxed, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 30})

            z_opt = res.x
            valid, val = _check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                # Repair: shrink radii
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.93
                valid, val = _check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # Phase 2: Polish best result
    if best_result is not None:
        remaining = DEADLINE - (time.time() - START) - 5
        if remaining > 3:
            polished = _polish_result(best_result, remaining)
            valid, val = _check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    # Phase 3: High precision polish (SLSQP + coordinate descent)
    if best_result is not None:
        remaining = DEADLINE - (time.time() - START) - 3
        if remaining > 2:
            hp_polished = _high_precision_polish(best_result, remaining)
            valid, val = _check_validity(hp_polished)
            if valid and val > best_value:
                best_value = val
                best_result = hp_polished

    # Phase 4: Final coordinate-descent radius squeeze
    if best_result is not None:
        x = best_result[0::3][:N]
        y = best_result[1::3][:N]
        r = best_result[2::3][:N].copy()
        # Compute max possible radius for each circle
        d = _compute_distances(x, y)
        for i in range(N):
            max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])  # placeholder, use actual w,h
        # Actually get w,h from best_result
        w_val = best_result[-1]
        h_val = 2.0 - w_val
        for i in range(N):
            max_r = min(x[i], w_val - x[i], y[i], h_val - y[i])
            for j in range(N):
                if i != j:
                    max_r = min(max_r, d[i, j] - r[j])
            r[i] = max(r[i], max_r)
        final = np.column_stack((x, y, r))
        # Verify
        valid, val = _check_validity(np.concatenate([
            np.column_stack((x, y, r)).reshape(-1)[0::3],
            np.column_stack((x, y, r)).reshape(-1)[1::3],
            np.column_stack((x, y, r)).reshape(-1)[2::3],
            [w_val]
        ]))
        # Simpler validity check
        valid = True
        for i in range(N):
            if r[i] <= 0 or x[i] < r[i] - 1e-9 or x[i] > w_val - r[i] + 1e-9:
                valid = False
                break
            if y[i] < r[i] - 1e-9 or y[i] > h_val - r[i] + 1e-9:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    if d[i,j] < r[i] + r[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            best_value = float(np.sum(r))
            best_result = np.column_stack((x, y, r))

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]

    # Safety shrink
    r = r * (1 - 1e-7)

    return np.column_stack((x, y, r))
