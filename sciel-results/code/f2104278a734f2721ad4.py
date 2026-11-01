"""Circle packing in unit square n=32 — radius-space local search with multi-restart."""
import time
import numpy as np

N = 32

_DEADLINE = 115.0  # seconds, safety margin before 120s kill


def _clearances(pos, rad, n):
    """For each circle, clearance to nearest other circle and to nearest wall."""
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    c_nei = d - rad[:, None] - rad[None, :]
    c_nei = np.min(c_nei, axis=1)
    c_wall = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                        np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    return c_nei, c_wall


def _objective(pos, rad, n):
    """Sum of radii (the score). Returns -inf if infeasible."""
    c_nei, c_wall = _clearances(pos, rad, n)
    if np.any(c_nei < -1e-9) or np.any(c_wall < -1e-9):
        return -np.inf
    return float(np.sum(rad))


def _make_feasible(pos, rad, n, iters=400):
    """Shrink radii and nudge positions until feasible (if possible)."""
    pos = pos.copy()
    rad = rad.copy()
    for _ in range(iters):
        c_nei, c_wall = _clearances(pos, rad, n)
        if np.all(c_nei >= -1e-9) and np.all(c_wall >= -1e-9):
            break
        # shrink the most-violated circle
        worst = np.minimum(c_nei, c_wall)
        i = int(np.argmin(worst))
        rad[i] = max(1e-5, rad[i] * 0.9)
        # also nudge position toward center if wall violation
        if c_wall[i] < -1e-9:
            pos[i, 0] = 0.5 + (pos[i, 0] - 0.5) * 0.8
            pos[i, 1] = 0.5 + (pos[i, 1] - 0.5) * 0.8
    # final feasibility check
    c_nei, c_wall = _clearances(pos, rad, n)
    if np.any(c_nei < -1e-9) or np.any(c_wall < -1e-9):
        # last resort: shrink all
        for _ in range(200):
            c_nei, c_wall = _clearances(pos, rad, n)
            if np.all(c_nei >= -1e-9) and np.all(c_wall >= -1e-9):
                break
            rad *= 0.95
    c_nei, c_wall = _clearances(pos, rad, n)
    if np.any(c_nei < -1e-9) or np.any(c_wall < -1e-9):
        return None
    return pos, rad


def _local_search(pos, rad, n, budget, rng):
    """Local search: alternate random radius perturbations + position relaxation.
    Accept if feasible and improves objective."""
    pos = pos.copy()
    rad = rad.copy()
    best = _objective(pos, rad, n)
    if not np.isfinite(best):
        return None
    t0 = time.time()
    step = 0.05
    for it in range(20000):
        if time.time() - t0 > budget:
            break
        # 1) propose a radius change on a random circle
        i = int(rng.integers(n))
        delta = (rng.random() - 0.5) * step * max(rad[i], 0.02)
        new_rad = rad[i] + delta
        if new_rad < 1e-5:
            new_rad = 1e-5
        if new_rad > 0.45:
            new_rad = 0.45
        rad[i] = new_rad
        # 2) position relaxation: push i away from violations
        for _ in range(20):
            c_nei, c_wall = _clearances(pos, rad, n)
            viol = np.minimum(c_nei, c_wall)
            if viol[i] >= -1e-9:
                break
            # move i away from its nearest neighbor / toward center
            d = np.linalg.norm(pos[i] - pos, axis=1)
            np.fill_diagonal(d, np.inf)
            j = int(np.argmin(d))
            if c_nei[i] < c_wall[i]:
                # neighbor violation: move away from j
                dirv = (pos[i] - pos[j])
                norm = np.linalg.norm(dirv)
                if norm > 1e-12:
                    dirv = dirv / norm
                else:
                    dirv = np.array([rng.random() - 0.5, rng.random() - 0.5])
                    dirv = dirv / (np.linalg.norm(dirv) + 1e-12)
                pos[i] = pos[i] + dirv * max(0.005, -viol[i] * 0.5)
            else:
                # wall violation: move toward center
                center = np.array([0.5, 0.5])
                dirv = center - pos[i]
                norm = np.linalg.norm(dirv)
                if norm > 1e-12:
                    dirv = dirv / norm
                pos[i] = pos[i] + dirv * max(0.005, -viol[i] * 0.5)
            pos[i, 0] = np.clip(pos[i, 0], rad[i], 1.0 - rad[i])
            pos[i, 1] = np.clip(pos[i, 1], rad[i], 1.0 - rad[i])
        # 3) check feasibility of the whole configuration
        c_nei, c_wall = _clearances(pos, rad, n)
        if np.any(c_nei < -1e-9) or np.any(c_wall < -1e-9):
            # revert radius change
            rad[i] = rad[i] - delta
            # revert position: recompute from scratch is expensive; instead try to fix
            # Simple: shrink i a bit more
            rad[i] = max(1e-5, rad[i] * 0.9)
            continue
        obj = _objective(pos, rad, n)
        if np.isfinite(obj) and obj > best:
            best = obj
            # anneal step size slowly
            step *= 0.9995
        else:
            # revert
            rad[i] = rad[i] - delta
    return pos, rad, best


def _initial_grid(n, rng):
    """Create an initial grid configuration with small random jitter."""
    side = int(np.ceil(np.sqrt(n)))
    idx = 0
    pos = []
    rad = []
    for i in range(side):
        for j in range(side):
            if idx >= n:
                break
            x = (i + 0.5) / side + (rng.random() - 0.5) * 0.02
            y = (j + 0.5) / side + (rng.random() - 0.5) * 0.02
            r = 0.5 / side - 0.005
            pos.append([x, y])
            rad.append(r)
            idx += 1
    pos = np.array(pos[:n])
    rad = np.array(rad[:n])
    # clip positions
    for k in range(n):
        pos[k, 0] = np.clip(pos[k, 0], rad[k], 1.0 - rad[k])
        pos[k, 1] = np.clip(pos[k, 1], rad[k], 1.0 - rad[k])
    return pos, rad


def _initial_random(n, rng):
    """Random scatter, then shrink to feasible."""
    pos = rng.random((n, 2))
    rad = np.full(n, 0.05)
    feas = _make_feasible(pos, rad, n, iters=200)
    if feas is None:
        return None
    return feas


def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)

    # Start from grid
    pos0, rad0 = _initial_grid(N, rng)
    feas = _make_feasible(pos0, rad0, N, iters=200)
    if feas is None:
        # fallback: pure grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    pos, rad = feas

    best_pos = pos.copy()
    best_rad = rad.copy()
    best_score = _objective(pos, rad, N)
    if not np.isfinite(best_score):
        best_score = 0.0

    # Multi-restart local search with budget splitting
    n_restarts = 8
    per_budget = (_DEADLINE - (time.time() - t_start) - 5.0) / max(1, n_restarts)
    per_budget = max(5.0, min(15.0, per_budget))

    for r in range(n_restarts):
        if time.time() - t_start > _DEADLINE - 5.0:
            break
        # Perturb current best to create diversity
        if r == 0:
            p, rd = pos.copy(), rad.copy()
        else:
            p = best_pos.copy()
            rd = best_rad.copy()
            # random perturbation
            for k in range(N):
                rd[k] *= (1.0 + (rng.random() - 0.5) * 0.3)
                p[k, 0] += (rng.random() - 0.5) * 0.05
                p[k, 1] += (rng.random() - 0.5) * 0.05
            feas = _make_feasible(p, rd, N, iters=300)
            if feas is None:
                # try random scatter
                feas = _initial_random(N, rng)
                if feas is None:
                    continue
            p, rd = feas

        result = _local_search(p, rd, N, per_budget, rng)
        if result is None:
            continue
        p, rd, score = result
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_pos = p.copy()
            best_rad = rd.copy()

    # Final feasibility check
    c_nei, c_wall = _clearances(best_pos, best_rad, N)
    if np.any(c_nei < -1e-9) or np.any(c_wall < -1e-9):
        # fall back to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    out = np.column_stack([best_pos, best_rad])
    return out


# EVOLVE-BLOCK-END
