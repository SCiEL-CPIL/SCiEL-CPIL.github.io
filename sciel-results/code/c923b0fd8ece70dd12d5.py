"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 110.0  # safety margin before the 120s (and 150s kill) limit


def _penalty_and_objective(z, n):
    """z = [x_1..x_n, y_1..y_n, r_1..r_n]. Return (value, grad).
    value = sum(r) - K_overlap*overlap - K_boundary*boundary.
    Higher value is better; we minimize -value."""
    x = z[:n]
    y = z[n:2*n]
    r = z[2*n:3*n]

    # pairwise overlap (positive when circles overlap)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    rr = r[:, None] + r[None, :]
    d = np.sqrt(dx*dx + dy*dy)
    # overlap amount per pair (0 if separated)
    ov = np.maximum(rr - d, 0.0)
    # use only upper triangle
    iu = np.triu_indices(n, k=1)
    overlap = np.sum(ov[iu])

    # boundary violation: circle must be inside [0,1]^2
    b = np.maximum(0.0, r - x) + np.maximum(0.0, r - y) \
      + np.maximum(0.0, r - (1.0 - x)) + np.maximum(0.0, r - (1.0 - y))
    boundary = np.sum(b)

    K_ov = 50.0
    K_b = 50.0
    value = np.sum(r) - K_ov * overlap - K_b * boundary

    # gradients
    gx = np.zeros(n); gy = np.zeros(n); gr = np.zeros(n)

    # overlap gradient
    for i, j in zip(iu[0], iu[1]):
        if d[i, j] > 1e-12:
            s = ov[i, j]
            if s > 0:
                ux = dx[i, j] / d[i, j]
                uy = dy[i, j] / d[i, j]
                gx[i] -= K_ov * ux
                gy[i] -= K_ov * uy
                gx[j] += K_ov * ux
                gy[j] += K_ov * uy
                gr[i] -= K_ov
                gr[j] -= K_ov
        else:
            # degenerate: push apart along x
            gx[i] -= K_ov
            gx[j] += K_ov
            gr[i] -= K_ov
            gr[j] -= K_ov

    # boundary gradient
    for i in range(n):
        if r[i] > x[i]:
            gx[i] += K_b
            gr[i] -= K_b
        if r[i] > y[i]:
            gy[i] += K_b
            gr[i] -= K_b
        if r[i] > (1.0 - x[i]):
            gx[i] -= K_b
            gr[i] -= K_b
        if r[i] > (1.0 - y[i]):
            gy[i] -= K_b
            gr[i] -= K_b

    # objective gradient (minimize -value)
    gr += 1.0  # d(sum r)/dr = 1, so d(-value)/dr = -1; we return grad of -value
    grad = np.concatenate([-gx, -gy, -gr])
    return -value, grad


def _is_valid(arr, tol=1e-7):
    """Check disjoint interiors and containment in [0,1]^2."""
    x = arr[:, 0]; y = arr[:, 1]; r = arr[:, 2]
    n = len(arr)
    if np.any(r < 0):
        return False
    if np.any(x < -tol) or np.any(y < -tol) or np.any(x > 1+tol) or np.any(y > 1+tol):
        return False
    # containment with radius
    if np.any(x - r < -tol) or np.any(y - r < -tol) or np.any(x + r > 1+tol) or np.any(y + r > 1+tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx*dx + dy*dy)
    rr = r[:, None] + r[None, :]
    # interior disjoint: distance >= r_i + r_j (allow tiny tolerance)
    if np.any(d < rr - tol):
        return False
    return True


def _clamp_valid(arr):
    """Shrink radii to guarantee validity; never move centers out of box.
    Returns a valid array (possibly with smaller radii)."""
    x = arr[:, 0].copy(); y = arr[:, 1].copy(); r = arr[:, 2].copy()
    n = len(arr)
    # clamp centers to [0,1]
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    # boundary clamp
    for i in range(n):
        r[i] = min(r[i], x[i], y[i], 1.0 - x[i], 1.0 - y[i])
        r[i] = max(r[i], 0.0)
    # iterative overlap resolution by shrinking the smaller circle
    for _ in range(200):
        changed = False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx*dx + dy*dy)
        rr = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        for i, j in zip(iu[0], iu[1]):
            if d[i, j] < rr[i, j] - 1e-10:
                # need to shrink so that r_i + r_j <= d
                # reduce both proportionally to current radii
                need = rr[i, j] - d[i, j]
                if need > 0:
                    tot = r[i] + r[j]
                    if tot <= 1e-12:
                        r[i] = 0.0; r[j] = 0.0
                    else:
                        # scale down the pair
                        scale = max(0.0, (d[i, j]) / tot)
                        r[i] *= scale
                        r[j] *= scale
                    changed = True
        if not changed:
            break
    # final boundary re-clamp after shrink
    for i in range(n):
        r[i] = min(r[i], x[i], y[i], 1.0 - x[i], 1.0 - y[i])
        r[i] = max(r[i], 0.0)
    return np.column_stack([x, y, r])


def _grid_init(n, rng):
    side = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(side) + 0.5) / side
    ys = (np.arange(side) + 0.5) / side
    pts = [(a, b) for a in xs for b in ys]
    # shuffle
    idx = rng.permutation(len(pts))[:n]
    x = np.array([pts[i][0] for i in idx])
    y = np.array([pts[i][1] for i in idx])
    r = np.full(n, 0.5/side)
    return np.concatenate([x, y, r])


def _random_init(n, rng):
    x = rng.uniform(0.1, 0.9, n)
    y = rng.uniform(0.1, 0.9, n)
    r = rng.uniform(0.02, 0.08, n)
    return np.concatenate([x, y, r])


def construct():
    t0 = time.time()
    best_arr = None
    best_val = -np.inf
    n = N

    # baseline grid (always valid-ish)
    base = _grid_init(n, np.random.default_rng(0))
    base = _clamp_valid(base)
    if _is_valid(base):
        best_arr = base.copy()
        best_val = np.sum(base[:, 2])

    rng = np.random.default_rng(12345)
    lb = np.zeros(3*n)
    ub = np.zeros(3*n)
    ub[:n] = 1.0; ub[n:2*n] = 1.0
    ub[2*n:] = 0.5  # radius upper bound

    starts = []
    starts.append(_grid_init(n, rng))
    for _ in range(3):
        starts.append(_random_init(n, rng))

    for s, z0 in enumerate(starts):
        if time.time() - t0 > _DEADLINE:
            break
        try:
            res = minimize(
                _penalty_and_objective, z0,
                args=(n,),
                method='L-BFGS-B',
                jac=True,
                bounds=list(zip(lb, ub)),
                options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-6, 'maxfun': 6000},
            )
        except Exception:
            continue
        z = res.x
        arr = np.column_stack([z[:n], z[n:2*n], z[2*n:3*n]])
        arr = _clamp_valid(arr)
        if _is_valid(arr):
            val = np.sum(arr[:, 2])
            if val > best_val:
                best_val = val
                best_arr = arr.copy()

    if best_arr is None:
        # fallback: tiny circles
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i+0.5)/side, (j+0.5)/side, 0.4/side) for i in range(side) for j in range(side)]
        best_arr = _clamp_valid(np.array(cells[:n]))

    # final safety: ensure valid
    best_arr = _clamp_valid(best_arr)
    if not _is_valid(best_arr):
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i+0.5)/side, (j+0.5)/side, 0.4/side) for i in range(side) for j in range(side)]
        best_arr = _clamp_valid(np.array(cells[:n]))
    return best_arr
# EVOLVE-BLOCK-END
