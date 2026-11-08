"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii.

Combines a force-based (PBD-style) physics optimizer with a final L-BFGS-B local
refinement. The physics pass handles global rearrangement from multiple diverse
starts (grid + random + symmetric corner starts); the polish pass uses a smooth
penalized objective to extract the last fraction of radius from the best packing.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_START = time.time()
DEADLINE = _START + 116.0          # safety margin before 150 s wall-clock kill
RNG = np.random.default_rng(20240517)

# --- force parameters ---
K_WALL = 500.0
K_SEP = 40.0
K_R = 10.0
EPS = 1e-9
STEP = 0.03
R_MIN = 1e-6
R_MAX = 0.5


def _is_valid(c, tol=1e-6):
    """Return True if packing is inside [0,1]^2 and circles are disjoint."""
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    if r.min() < R_MIN:
        return False
    if (x.min() < -tol) or (x.max() > 1 + tol) or (y.min() < -tol) or (y.max() > 1 + tol):
        return False
    if np.any(x + r > 1 + tol) or np.any(x - r < -tol):
        return False
    if np.any(y + r > 1 + tol) or np.any(y - r < -tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - tol:
                return False
    return True


def _score(c):
    return float(np.sum(c[:, 2])) if _is_valid(c) else -np.inf


def _clamp(c):
    """Clamp positions/radii to a feasible box (keeps circles inside square)."""
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    r = np.clip(r, R_MIN, R_MAX)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    return np.column_stack([x, y, r])


def _forces(c):
    x = c[:, 0].copy(); y = c[:, 1].copy(); r = c[:, 2].copy()
    fx = np.zeros(N); fy = np.zeros(N); fr = np.zeros(N)

    # wall forces
    fx += K_WALL * np.where(x < r, 1.0, 0.0) - K_WALL * np.where(x > 1 - r, 1.0, 0.0)
    fy += K_WALL * np.where(y < r, 1.0, 0.0) - K_WALL * np.where(y > 1 - r, 1.0, 0.0)
    fr += K_WALL * (np.where(x < r, 1.0, 0.0) + np.where(x > 1 - r, 1.0, 0.0)
                    + np.where(y < r, 1.0, 0.0) + np.where(y > 1 - r, 1.0, 0.0))

    # separation forces
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]; dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            s = r[i] + r[j]
            if d < s:
                if d < 1e-9:
                    dx = RNG.standard_normal(); dy = RNG.standard_normal()
                    d = np.hypot(dx, dy)
                ux = dx / d; uy = dy / d
                f = K_SEP * (s - d)
                fx[i] += f * ux; fy[i] += f * uy
                fx[j] -= f * ux; fy[j] -= f * uy
                fr[i] += f; fr[j] += f

    # maximize radii
    fr += K_R
    return fx, fy, fr


def _pbd_step(c):
    """One force-based step with adaptive radius growth and clamping."""
    c = _clamp(c)
    fx, fy, fr = _forces(c)

    # adaptive step: smaller when forces are large
    fmag = np.hypot(fx, fy)
    step = STEP * 0.5 / (1.0 + 0.02 * fmag.max())

    x = c[:, 0] + step * fx
    y = c[:, 1] + step * fy

    # grow radii gently toward the wall-limited capacity
    # capacity for each circle given current position
    cap = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    new_r = c[:, 2] + step * fr * 0.05
    new_r = np.minimum(new_r, cap)          # never exceed wall capacity
    new_r = np.clip(new_r, R_MIN, R_MAX)

    return np.column_stack([x, y, new_r])


def _objective_polish(params):
    """Penalized objective for L-BFGS-B polish: minimize -sum(r) + penalties."""
    coords = params[:2 * N].reshape(-1, 2)
    radii = np.exp(params[2 * N:])           # positive via log-param
    radii = np.clip(radii, R_MIN, R_MAX)
    x = coords[:, 0]; y = coords[:, 1]

    pen = 0.0
    pen += np.sum(np.maximum(0.0, r := radii - x) ** 2)
    pen += np.sum(np.maximum(0.0, radii - (1.0 - x)) ** 2)
    pen += np.sum(np.maximum(0.0, radii - y) ** 2)
    pen += np.sum(np.maximum(0.0, radii - (1.0 - y)) ** 2)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            s = radii[i] + radii[j]
            if d < s:
                pen += (s - d) ** 2
    return -np.sum(radii) + 50.0 * pen


def _polish(c, time_left):
    """Local L-BFGS-B refinement of a valid packing. Returns improved packing or c."""
    if time_left < 3.0:
        return c
    c = _clamp(c)
    x0 = np.concatenate([c[:, :2].flatten(), np.log(np.maximum(c[:, 2], R_MIN))])

    def _check(res_params):
        coords = res_params[:2 * N].reshape(-1, 2)
        radii = np.clip(np.exp(res_params[2 * N:]), R_MIN, R_MAX)
        cand = _clamp(np.column_stack([coords, radii]))
        return cand, _score(cand)

    best = c
    best_s = _score(c)
    try:
        res = minimize(_objective_polish, x0, method='L-BFGS-B',
                       options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-9})
        cand, s = _check(res.x)
        if s > best_s:
            best, best_s = cand, s
        # also check the polished-but-clamped starting point trajectory
    except Exception:
        pass
    return best


def _grid_start():
    side = int(np.ceil(np.sqrt(N)))
    xs = np.linspace(0.5 / side, 1 - 0.5 / side, side)
    ys = np.linspace(0.5 / side, 1 - 0.5 / side, side)
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= N:
                break
            pts.append([xs[i], ys[j], 0.4 / side])
    return np.array(pts[:N])


def _random_start():
    r = np.full(N, 0.04 + 0.03 * RNG.random())
    x = np.clip(RNG.uniform(r, 1 - r, N), R_MIN, 1 - R_MIN)
    y = np.clip(RNG.uniform(r, 1 - r, N), R_MIN, 1 - R_MIN)
    return np.column_stack([x, y, r])


def _corner_start():
    """Symmetric starts exploiting the square's 4-fold symmetry."""
    # four corner clusters + center
    centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.5)]
    pts = []
    per = N // len(centers)
    rem = N % len(centers)
    idx = 0
    for ci, (cx, cy) in enumerate(centers):
        k = per + (1 if ci < rem else 0)
        for _ in range(k):
            if idx >= N:
                break
            ang = 2 * np.pi * RNG.random()
            rad = 0.08 * RNG.random()
            px = np.clip(cx + rad * np.cos(ang), 0.05, 0.95)
            py = np.clip(cy + rad * np.sin(ang), 0.05, 0.95)
            pts.append([px, py, 0.05 + 0.02 * RNG.random()])
            idx += 1
    return np.array(pts[:N])


def construct():
    start_time = time.time()
    deadline = start_time + 114.0

    best = _grid_start()
    best_s = _score(best)

    # build a list of diverse starts
    starts = [_grid_start(), _corner_start()]
    for _ in range(6):
        starts.append(_random_start())

    max_iters = 3000

    for s_idx, start in enumerate(starts):
        if time.time() > deadline - 8.0:
            break
        c = start.copy()
        # run physics pass
        for it in range(max_iters):
            c = _pbd_step(c)
            s = _score(c)
            if s > best_s:
                best_s = s
                best = c.copy()
            # convergence heuristic: if radii stopped growing meaningfully
            if it > 200 and it % 100 == 0:
                # check time within this start
                if time.time() > deadline - 6.0:
                    break
            # early stop if clearly stuck (all radii at wall capacity and no motion)
            if it % 250 == 250:
                cap = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]), np.minimum(c[:, 1], 1 - c[:, 1]))
                if np.allclose(c[:, 2], cap, atol=1e-4):
                    break

        # polish the best from this start
        if time.time() < deadline - 4.0:
            polished = _polish(c, deadline - time.time())
            ps = _score(polished)
            if ps > best_s:
                best_s = ps
                best = polished

    # final global polish on the overall best
    if time.time() < deadline - 2.0:
        best = _polish(best, deadline - time.time())

    if not _is_valid(best):
        # fallback to a guaranteed-valid grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best
