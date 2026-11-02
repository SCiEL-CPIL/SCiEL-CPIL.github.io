"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.

STRATEGY: Delaunay contact-graph-aware joint optimization.
- Build Delaunay triangulation of centers to get sparse contact pairs.
- Joint L-BFGS-B optimization of positions AND radii with soft penalty
  over only Delaunay-neighbor pairs (not all N*(N-1)/2 pairs).
- Iterative radius growth: maximize each radius given current positions,
  then re-optimize positions with the new radii.
- Multiple diverse initializations (hex, grid, random, corner-weighted).
- Final exact radius maximization pass.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-9


def _max_radius(pos, rad):
    """Maximal radius for each circle given current positions and radii."""
    n = len(rad)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                dmin = min(dmin, d - rad[j])
        mr[i] = max(0.0, dmin)
    return mr


def _valid_check(cands):
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


def _delaunay_edges(pos):
    """Get Delaunay edges (pairs of indices) from center positions."""
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = int(simplex[i]), int(simplex[j])
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        # Fallback: all pairs (shouldn't happen for 26 points)
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _soft_penalty_objective(z, edges, lam):
    """
    z = [x0,y0, x1,y1, ..., xN-1,yN-1, r0,r1,...,rN-1]
    Objective: -sum(r_i) + lam * (sum of overlap violations^2 + wall violations^2)
    Only uses Delaunay edges for pair constraints.
    """
    pos = z[:2 * N].reshape(N, 2)
    rad = z[2 * N:]

    # Negative sum of radii (we minimize, so maximize sum)
    obj = -np.sum(rad)

    # Pair overlap penalties (only Delaunay edges)
    for i, j in edges:
        d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
        overlap = rad[i] + rad[j] - d
        if overlap > 0:
            obj += lam * overlap * overlap

    # Wall penalties
    for i in range(N):
        for ax in range(2):
            dist = pos[i, ax] if ax == 0 else pos[i, 1]
            # distance to wall
            wl = dist - rad[i]
            wr = (1.0 - dist) - rad[i]
            if wl < 0:
                obj += lam * wl * wl
            if wr < 0:
                obj += lam * wr * wr

    return obj


def _soft_penalty_grad(z, edges, lam):
    pos = z[:2 * N].reshape(N, 2)
    rad = z[2 * N:]
    grad = np.zeros_like(z)

    # Gradient of -sum(rad)
    grad[2 * N:] = -1.0

    # Pair overlap gradients
    for i, j in edges:
        diff = pos[i] - pos[j]
        d = np.hypot(diff[0], diff[1])
        if d < EPS:
            continue
        overlap = rad[i] + rad[j] - d
        if overlap > 0:
            g = 2.0 * lam * overlap
            # d(overlap)/d(pos[i]) = -diff/d
            grad[2 * i] += -g * diff[0] / d
            grad[2 * i + 1] += -g * diff[1] / d
            grad[2 * j] += g * diff[0] / d
            grad[2 * j + 1] += g * diff[1] / d
            # d(overlap)/d(rad[i]) = 1, d(overlap)/d(rad[j]) = 1
            grad[2 * N + i] += g
            grad[2 * N + j] += g

    # Wall gradients
    for i in range(N):
        for ax in range(2):
            dist = pos[i, ax]
            wl = dist - rad[i]
            wr = (1.0 - dist) - rad[i]
            if wl < 0:
                g = 2.0 * lam * wl
                grad[2 * i + ax] += g  # d(wl)/d(pos) = 1
                grad[2 * N + i] += -g  # d(wl)/d(rad) = -1
            if wr < 0:
                g = 2.0 * lam * wr
                grad[2 * i + ax] += -g  # d(wr)/d(pos) = -1
                grad[2 * N + i] += -g  # d(wr)/d(rad) = -1

    return grad


def _joint_optimize(pos, rad, n_iter=200, max_lam=500.0):
    """
    Joint optimization of positions and radii using Delaunay contact graph.
    Uses adaptive lambda (penalty parameter) that increases if constraints violated.
    """
    pos = np.array(pos, dtype=float).copy()
    rad = np.array(rad, dtype=float).copy()

    # Initial radii: small to start
    rad = np.maximum(rad, 0.01)

    lam = 10.0
    best_pos, best_rad = pos.copy(), rad.copy()
    best_score = -np.inf

    for outer in range(n_iter):
        # Rebuild Delaunay edges
        edges = _delaunay_edges(pos)

        # Build initial z
        z0 = np.concatenate([pos.ravel(), rad])

        # Bounds: positions in [0,1], radii in [0, 0.5]
        lb = np.concatenate([np.zeros(2 * N), np.zeros(N)])
        ub = np.concatenate([np.ones(2 * N), np.full(N, 0.5)])

        try:
            res = minimize(
                _soft_penalty_objective, z0,
                jac=lambda z: _soft_penalty_grad(z, edges, lam),
                args=(edges, lam),
                method='L-BFGS-B',
                bounds=list(zip(lb, ub)),
                options={'maxiter': 80, 'maxfun': 2000, 'gtol': 1e-6}
            )
            z = res.x
            pos = z[:2 * N].reshape(N, 2)
            rad = z[2 * N:]
        except Exception:
            break

        # Check validity
        cands = list(zip(pos[:, 0], pos[:, 1], rad))
        arr, s = _valid_check(cands)
        if arr is not None and s > best_score:
            best_score = s
            best_pos = pos.copy()
            best_rad = rad.copy()

        # If still violations, increase lambda
        viol = 0.0
        for i, j in edges:
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            v = rad[i] + rad[j] - d
            if v > 1e-6:
                viol += v
        for i in range(N):
            for ax in range(2):
                d = pos[i, ax] - rad[i]
                if d < -1e-6:
                    viol += -d
                d2 = (1.0 - pos[i, ax]) - rad[i]
                if d2 < -1e-6:
                    viol += -d2

        if viol < 1e-4:
            # Constraints satisfied, try to grow radii
            rad_new = _max_radius(pos, rad)
            rad = np.maximum(rad, rad_new)
            # Don't increase lambda
        else:
            lam = min(lam * 1.5, max_lam)

    return best_pos, best_rad


def _grow_and_relax(pos, rad, deadline):
    """
    Iterative growth: maximize radii, then re-optimize positions, repeat.
    """
    pos = np.array(pos, dtype=float).copy()
    rad = np.array(rad, dtype=float).copy()
    best_pos, best_rad = pos.copy(), rad.copy()
    best_score = -np.inf

    for it in range(30):
        if time.time() > deadline - 5:
            break

        # Maximize radii given positions
        rad = _max_radius(pos, rad)

        # Check validity
        cands = list(zip(pos[:, 0], pos[:, 1], rad))
        arr, s = _valid_check(cands)
        if arr is not None and s > best_score:
            best_score = s
            best_pos = pos.copy()
            best_rad = rad.copy()

        # Re-optimize positions with current radii (fix radii, move positions)
        def pos_obj(p_flat, rad_fixed):
            p = p_flat.reshape(N, 2)
            edges = _delaunay_edges(p)
            obj = 0.0
            for i, j in edges:
                d = np.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1])
                overlap = rad_fixed[i] + rad_fixed[j] - d
                if overlap > 0:
                    obj += overlap * overlap
            for i in range(N):
                for ax in range(2):
                    wl = p[i, ax] - rad_fixed[i]
                    wr = (1.0 - p[i, ax]) - rad_fixed[i]
                    if wl < 0:
                        obj += wl * wl
                    if wr < 0:
                        obj += wr * wr
            return obj

        def pos_grad(p_flat, rad_fixed):
            p = p_flat.reshape(N, 2)
            edges = _delaunay_edges(p)
            g = np.zeros_like(p_flat)
            for i, j in edges:
                diff = p[i] - p[j]
                d = np.hypot(diff[0], diff[1])
                if d < EPS:
                    continue
                overlap = rad_fixed[i] + rad_fixed[j] - d
                if overlap > 0:
                    gv = 2.0 * overlap
                    g[2 * i] += -gv * diff[0] / d
                    g[2 * i + 1] += -gv * diff[1] / d
                    g[2 * j] += gv * diff[0] / d
                    g[2 * j + 1] += gv * diff[1] / d
            for i in range(N):
                for ax in range(2):
                    wl = p[i, ax] - rad_fixed[i]
                    wr = (1.0 - p[i, ax]) - rad_fixed[i]
                    if wl < 0:
                        g[2 * i + ax] += 2.0 * wl
                    if wr < 0:
                        g[2 * i + ax] += -2.0 * wr
            return g

        try:
            res = minimize(pos_obj, pos.ravel(), jac=pos_grad, args=(rad,),
                           method='L-BFGS-B',
                           bounds=[(0.0, 1.0)] * (2 * N),
                           options={'maxiter': 60, 'maxfun': 1500, 'gtol': 1e-7})
            pos = res.x.reshape(N, 2)
        except Exception:
            pass

    return best_pos, best_rad


def _polish(pos, rad, deadline):
    """Final polish: alternating max-radius and position optimization."""
    pos = np.array(pos, dtype=float).copy()
    rad = np.array(rad, dtype=float).copy()

    for it in range(10):
        if time.time() > deadline - 3:
            break
        rad = _max_radius(pos, rad)
        # Position optimization
        def pos_obj(p_flat, rad_fixed):
            p = p_flat.reshape(N, 2)
            obj = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1])
                    overlap = rad_fixed[i] + rad_fixed[j] - d
                    if overlap > 0:
                        obj += overlap * overlap
                for ax in range(2):
                    wl = p[i, ax] - rad_fixed[i]
                    wr = (1.0 - p[i, ax]) - rad_fixed[i]
                    if wl < 0:
                        obj += wl * wl
                    if wr < 0:
                        obj += wr * wr
            return obj

        def pos_grad(p_flat, rad_fixed):
            p = p_flat.reshape(N, 2)
            g = np.zeros_like(p_flat)
            for i in range(N):
                for j in range(i + 1, N):
                    diff = p[i] - p[j]
                    d = np.hypot(diff[0], diff[1])
                    if d < EPS:
                        continue
                    overlap = rad_fixed[i] + rad_fixed[j] - d
                    if overlap > 0:
                        gv = 2.0 * overlap
                        g[2 * i] += -gv * diff[0] / d
                        g[2 * i + 1] += -gv * diff[1] / d
                        g[2 * j] += gv * diff[0] / d
                        g[2 * j + 1] += gv * diff[1] / d
                for ax in range(2):
                    wl = p[i, ax] - rad_fixed[i]
                    wr = (1.0 - p[i, ax]) - rad_fixed[i]
                    if wl < 0:
                        g[2 * i + ax] += 2.0 * wl
                    if wr < 0:
                        g[2 * i + ax] += -2.0 * wr
            return g

        try:
            res = minimize(pos_obj, pos.ravel(), jac=pos_grad, args=(rad,),
                           method='L-BFGS-B',
                           bounds=[(0.0, 1.0)] * (2 * N),
                           options={'maxiter': 40, 'maxfun': 1000, 'gtol': 1e-8})
            pos = res.x.reshape(N, 2)
        except Exception:
            pass

    rad = _max_radius(pos, rad)
    return pos, rad


def _anneal(pos0, deadline, seed):
    """
    Simulated annealing on positions with Delaunay-aware radius updates.
    Different from the current program's force-directed approach:
    uses Metropolis acceptance with temperature schedule and Delaunay
    contact graph for fast radius evaluation.
    """
    rng = np.random.default_rng(seed)
    pos = np.array(pos0, dtype=float).copy()
    rad = _max_radius(pos, np.full(N, 0.01))
    best_pos, best_rad = pos.copy(), rad.copy()
    best_score = float(rad.sum())

    n_iter = 300
    t0, t1 = 0.05, 0.001

    for it in range(n_iter):
        if time.time() > deadline - 5:
            break
        frac = it / max(1, n_iter - 1)
        t = t0 + (t1 - t0) * frac

        # Random perturbation of one or two positions
        n_perturb = 1 if rng.random() < 0.5 else 2
        idx = rng.choice(N, size=n_perturb, replace=False)
        pos_new = pos.copy()
        for i in idx:
            pos_new[i] += t * rng.standard_normal(2)
            pos_new[i] = np.clip(pos_new[i], 0.01, 0.99)

        rad_new = _max_radius(pos_new, np.full(N, 0.01))
        score_new = float(rad_new.sum())
        score_old = float(rad.sum())

        # Accept if better or with Metropolis probability
        if score_new >= score_old:
            pos, rad = pos_new, rad_new
        else:
            if rng.random() < np.exp((score_new - score_old) / max(t, 1e-10)):
                pos, rad = pos_new, rad_new

        if score_new > best_score:
            best_score = score_new
            best_pos = pos_new.copy()
            best_rad = rad_new.copy()

    return best_pos, best_rad


def construct():
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    best = None
    best_score = -1.0

    def consider(pos, rad):
        nonlocal best, best_score
        rad = _max_radius(np.array(pos, dtype=float), np.array(rad, dtype=float))
        arr, s = _valid_check(list(zip(pos[:, 0], pos[:, 1], rad)))
        if arr is not None and s > best_score:
            best_score = s
            best = arr

    # ---------- diverse initializations ----------
    rng = np.random.default_rng(42)
    starts = []

    # Hexagonal packing
    hex_pts = []
    rows = 6
    for row in range(rows):
        y = (row + 0.5) / rows
        offset = 0.5 / rows if row % 2 else 0.0
        for col in range(rows):
            x = (col + 0.5) / rows + offset
            if 0 <= x <= 1:
                hex_pts.append((x, y))
        if len(hex_pts) >= N:
            break
    starts.append(np.array(hex_pts[:N], dtype=float))

    # Grid layouts
    for side in [4, 5]:
        cells = [((i + 0.5) / side, (j + 0.5) / side)
                 for i in range(side) for j in range(side)][:N]
        starts.append(np.array(cells, dtype=float))

    # Corner-weighted random
    for _ in range(3):
        pts = []
        for _ in range(N):
            # Bias toward corners
            if rng.random() < 0.3:
                cx, cy = rng.choice([0.15, 0.85]), rng.choice([0.15, 0.85])
            else:
                cx, cy = rng.random(), rng.random()
            x = cx + 0.05 * rng.standard_normal()
            y = cy + 0.05 * rng.standard_normal()
            pts.append((np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)))
        starts.append(np.array(pts, dtype=float))

    # Uniform random
    for _ in range(3):
        starts.append(rng.random((N, 2)) * 0.8 + 0.1)

    # ---------- main search loop ----------
    for r, pos0 in enumerate(starts):
        if time.time() > DEADLINE - 10:
            break

        rad0 = np.full(N, 0.02)

        # Strategy A: Joint optimization (Delaunay-aware)
        if time.time() < DEADLINE - 20:
            try:
                pp, rr = _joint_optimize(pos0, rad0, n_iter=50, max_lam=200.0)
                consider(pp, rr)
            except Exception:
                pass

        # Strategy B: Grow and relax
        if time.time() < DEADLINE - 15:
            try:
                pp, rr = _grow_and_relax(pos0, rad0, DEADLINE)
                consider(pp, rr)
            except Exception:
                pass

        # Strategy C: Annealing
        if time.time() < DEADLINE - 10:
            try:
                pp, rr = _anneal(pos0, DEADLINE, seed=1000 + 7 * r)
                consider(pp, rr)
            except Exception:
                pass

    # ---------- final polish on best ----------
    if best is not None:
        pp, rr = _polish(best[:, :2], best[:, 2], DEADLINE)
        consider(pp, rr)

    # One more polish pass
    if best is not None:
        pp, rr = _polish(best[:, :2], best[:, 2], DEADLINE)
        consider(pp, rr)

    if best is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
