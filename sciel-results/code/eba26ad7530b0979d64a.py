"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.

STRUCTURALLY DIFFERENT STRATEGY: a single coupled continuous optimization over
all 3*N variables (positions AND radii) using a smooth quadratic barrier
penalty for all constraints, solved with L-BFGS-B from a diverse set of
well-structured initial seeds.  This treats the problem as one large smooth
NLP rather than an alternating position-fix / radius-max loop, which can
escape local minima that plague the alternating scheme.  A final exact
radius-expansion pass (iteratively setting each radius to its maximal
feasible value given the others) polishes the result.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
EPS = 1e-9


def _max_radius(pos, r):
    """Maximal radius for each circle given current positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                dmin = min(dmin, d - r[j])
        mr[i] = max(0.0, dmin)
    return mr


def _valid(cands):
    """Return (arr, score) for a valid packing of N circles, else (None, -1)."""
    valid = []
    for x, y, r in cands:
        if r <= 0:
            continue
        if x < r + EPS or x > 1 - r - EPS or y < r + EPS or y > 1 - r - EPS:
            continue
        ok = True
        for x2, y2, r2 in valid:
            if (x - x2) ** 2 + (y - y2) ** 2 < (r + r2) ** 2 - 1e-9:
                ok = False
                break
        if ok:
            valid.append((x, y, r))
    if len(valid) < N:
        return None, -1.0
    arr = np.array(valid[:N])
    return arr, float(arr[:, 2].sum())


def _exact_expand(pos, rad, iters=20):
    """Iteratively expand each radius to its exact maximum given the others."""
    rad = np.array(rad, dtype=float).copy()
    pos = np.array(pos, dtype=float)
    for _ in range(iters):
        for i in range(N):
            dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(N):
                if i != j:
                    d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                    dmin = min(dmin, d - rad[j])
            rad[i] = max(0.0, dmin)
    return pos, rad


def _barrier_obj(z, lam=5000.0):
    """Smooth barrier objective: -sum(r) + lam * (pair violations + wall violations)."""
    pos = z[:2 * N].reshape(N, 2)
    rad = z[2 * N:]
    # pair violations
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    viol_pair = np.maximum(rad[:, None] + rad[None, :] - d, 0.0)
    # wall violations
    viol_wall = np.maximum(rad - pos, 0.0) + np.maximum(rad - (1.0 - pos), 0.0)
    return -np.sum(rad) + lam * (np.sum(viol_pair ** 2) + np.sum(viol_wall ** 2))


def _barrier_grad(z, lam=5000.0):
    pos = z[:2 * N].reshape(N, 2)
    rad = z[2 * N:]
    g = np.zeros_like(z)

    # pair part
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    d_safe = np.maximum(d, EPS)
    diff = pos[:, None, :] - pos[None, :, :]
    viol = np.maximum(rad[:, None] + rad[None, :] - d, 0.0)
    # d/dpos_i of sum viol^2: 2*viol * (-diff/d)
    dp = 2.0 * lam * viol * (-diff / d_safe[:, :, None])
    # d/drad_i of sum viol^2: 2*viol * (indicator j>i or i>j)
    # For each pair (i,j), viol_ij contributes to rad_i and rad_j
    # grad rad_i = 2*lam*sum_j viol_ij (for j != i)
    g_pair_r = 2.0 * lam * viol.sum(axis=1)
    # set diagonal to 0 (viol_ii = 0 anyway)
    np.fill_diagonal(viol, 0.0)

    # wall part
    a = np.maximum(rad - pos, 0.0)
    b = np.maximum(rad - (1.0 - pos), 0.0)
    # d/dpos_i of wall: -2*lam*a + 2*lam*b  (per axis)
    gw = (-2.0 * lam * a + 2.0 * lam * b)
    # d/drad_i of wall: 2*lam*(a + b)
    g_wall_r = 2.0 * lam * (a + b)

    g[:2 * N] = (dp.sum(axis=1) + gw).reshape(-1)
    g[2 * N:] = -1.0 + g_pair_r + g_wall_r
    return g


def _seeds():
    """Generate diverse initial configurations."""
    seeds = []
    rng = np.random.default_rng(42)

    # 1) Grid-based seeds with varying aspect ratios
    for side in range(3, 9):
        cells = [((i + 0.5) / side, (j + 0.5) / side)
                 for i in range(side) for j in range(side)][:N]
        pos = np.array(cells, dtype=float)
        rad = np.full(N, 0.45 / side)
        z = np.concatenate([pos.ravel(), rad])
        seeds.append(z)

    # 2) Hexagonal packing seed
    pts = []
    for row in range(6):
        y = (row + 0.5) / 6
        offset = 0.5 / 6 if row % 2 else 0.0
        for col in range(6):
            x = (col + 0.5) / 6 + offset
            if 0 <= x <= 1:
                pts.append((x, y))
        if len(pts) >= N:
            break
    pos = np.array(pts[:N], dtype=float)
    rad = np.full(N, 0.45 / 6)
    seeds.append(np.concatenate([pos.ravel(), rad]))

    # 3) Ring / concentric seed
    pts = [(0.5, 0.5)]
    for r_ring, cnt in [(0.15, 6), (0.3, 12), (0.42, 7)]:
        for k in range(cnt):
            a = 2 * np.pi * k / cnt
            pts.append((0.5 + r_ring * np.cos(a), 0.5 + r_ring * np.sin(a)))
        if len(pts) >= N:
            break
    pos = np.array(pts[:N], dtype=float)
    rad = np.full(N, 0.08)
    seeds.append(np.concatenate([pos.ravel(), rad]))

    # 4) Random uniform seeds
    for _ in range(8):
        pos = rng.random((N, 2)) * 0.8 + 0.1
        rad = rng.random(N) * 0.05 + 0.02
        seeds.append(np.concatenate([pos.ravel(), rad]))

    # 5) One big circle + small ones around
    pos = np.zeros((N, 2))
    pos[0] = [0.5, 0.5]
    rad = np.full(N, 0.03)
    rad[0] = 0.2
    for k in range(1, N):
        a = 2 * np.pi * (k - 1) / (N - 1)
        pos[k] = [0.5 + 0.25 * np.cos(a), 0.5 + 0.25 * np.sin(a)]
    seeds.append(np.concatenate([pos.ravel(), rad]))

    # 6) Corner + edge biased seed
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    pts = list(corners)
    for _ in range(4):
        pts.append((0.5, 0.5 + rng.random() * 0.6 - 0.3))
    for _ in range(N - len(pts)):
        pts.append((rng.random() * 0.8 + 0.1, rng.random() * 0.8 + 0.1))
    pos = np.array(pts[:N], dtype=float)
    rad = np.full(N, 0.06)
    rad[:4] = 0.1
    seeds.append(np.concatenate([pos.ravel(), rad]))

    return seeds


def construct():
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before the 150s kill

    best = None
    best_score = -1.0

    def consider(pos, rad):
        nonlocal best, best_score
        rad = _exact_expand(pos, rad, iters=10)
        arr, s = _valid(list(zip(pos[:, 0], pos[:, 1], rad)))
        if arr is not None and s > best_score:
            best_score = s
            best = arr

    seeds = _seeds()

    for si, z0 in enumerate(seeds):
        if time.time() > DEADLINE - 10:
            break

        # Try a few barrier strengths (coarse to fine)
        for lam in [2000.0, 5000.0, 10000.0]:
            if time.time() > DEADLINE - 10:
                break
            try:
                res = minimize(
                    _barrier_obj, z0, jac=_barrier_grad, args=(lam,),
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N,
                    options={'maxiter': 150, 'maxfun': 3000, 'gtol': 1e-8}
                )
                z = res.x
                pos = z[:2 * N].reshape(N, 2)
                rad = z[2 * N:]
                # Clip to valid range
                pos = np.clip(pos, 0.0, 1.0)
                rad = np.clip(rad, 0.0, 0.5)
                consider(pos, rad)
            except Exception:
                pass

    # Final polish on the best found: run a tighter barrier optimization
    if best is not None:
        pos = best[:, :2]
        rad = best[:, 2]
        z0 = np.concatenate([pos.ravel(), rad])
        for lam in [10000.0, 20000.0]:
            if time.time() > DEADLINE - 5:
                break
            try:
                res = minimize(
                    _barrier_obj, z0, jac=_barrier_grad, args=(lam,),
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N,
                    options={'maxiter': 200, 'maxfun': 4000, 'gtol': 1e-10}
                )
                z = res.x
                pos = z[:2 * N].reshape(N, 2)
                rad = z[2 * N:]
                pos = np.clip(pos, 0.0, 1.0)
                rad = np.clip(rad, 0.0, 0.5)
                consider(pos, rad)
            except Exception:
                pass

    if best is None:
        # Fallback: 5x5 grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
