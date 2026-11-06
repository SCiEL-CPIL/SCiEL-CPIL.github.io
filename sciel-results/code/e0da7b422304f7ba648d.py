"""Fresh design for 50 equal spheres in a unit cube: structured seed + vectorized
repulsion/wall solver + simulated annealing with deadline guard."""
import numpy as np
import time

N = 50
HALF = 0.5


def _radius(c):
    """Common radius for a placement: min of half inter-centre distances and
    half distance to nearest wall."""
    c = np.asarray(c, dtype=np.float64)
    # pair distances (vectorized)
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt((diff * diff).sum(-1))
    np.fill_diagonal(d, np.inf)
    r_pairs = 0.5 * d.min()
    # walls
    r_wall = 0.5 * (HALF - np.abs(c)).min()
    return float(min(r_pairs, r_wall))


def _score(c):
    """Objective to maximize: the common radius. Negative if invalid."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 3) or not np.all(np.isfinite(c)):
        return -np.inf
    # keep inside (hard clamp used by caller); here just compute
    diff = c[:, None, :] - c[None, :, :]
    d2 = (diff * diff).sum(-1)
    np.fill_diagonal(d2, np.inf)
    r_pairs = 0.5 * np.sqrt(d2.min())
    r_wall = 0.5 * (HALF - np.abs(c)).min()
    return float(min(r_pairs, r_wall))


def _energy(c, r_target):
    """Energy to minimize: penalty for any pair closer than 2*r_target and any
    centre outside the shrunken box [-(HALF-r_target), HALF-r_target]."""
    lo = -(HALF - r_target)
    hi = HALF - r_target
    diff = c[:, None, :] - c[None, :, :]
    d2 = (diff * diff).sum(-1)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    # pair penalty: sum of squared overlap amounts
    overlap = np.maximum(2.0 * r_target - d, 0.0)
    pair_pen = (overlap * overlap).sum()
    # wall penalty
    out_lo = np.maximum(lo - c, 0.0)
    out_hi = np.maximum(c - hi, 0.0)
    wall_pen = (out_lo * out_lo).sum() + (out_hi * out_hi).sum()
    return float(pair_pen + wall_pen)


def _clamp(c):
    return np.clip(c, -HALF, HALF)


def _seed():
    """Structured seed: a 4x4x4 lattice (64 sites) trimmed to 50, jittered."""
    rng = np.random.default_rng(12345)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # take a well-spread subset of 50
    idx = rng.permutation(len(grid))[:N]
    c = grid[idx].astype(np.float64)
    # small jitter to break symmetry
    c += rng.normal(0, 0.01, size=c.shape)
    return _clamp(c)


def _sa_optimize(c0, time_budget):
    """Simulated annealing on the energy with a target radius that is slowly
    increased. Returns best placement found."""
    c = c0.copy()
    best_c = c.copy()
    best_r = _radius(c)
    r_target = best_r * 0.999
    rng = np.random.default_rng(98765)

    T = 0.05
    T_min = 1e-7
    cool = 0.9995

    start = time.time()
    deadline = start + time_budget
    step = 0.02

    while time.time() < deadline:
        # move a few random centres
        n_move = rng.integers(1, 4)
        idx = rng.integers(0, N, size=n_move)
        old = c[idx].copy()
        c[idx] += rng.normal(0, step, size=(n_move, 3))
        c = _clamp(c)

        # slowly raise target radius
        r_target = min(r_target * 1.0005, best_r + 0.02)

        e_new = _energy(c, r_target)
        e_old = _energy(old_full := c.copy(), r_target)
        # recompute old energy properly using saved old positions
        c_save = c.copy()
        c[idx] = old
        e_old = _energy(c, r_target)
        c = c_save

        if e_new < e_old or rng.random() < np.exp(-(e_new - e_old) / T):
            # accept
            cur_r = _radius(c)
            if cur_r > best_r:
                best_r = cur_r
                best_c = c.copy()
        else:
            c[idx] = old
            c = _clamp(c)

        T = max(T * cool, T_min)
        # adapt step
        if T < 0.01:
            step = 0.005
        elif T < 0.001:
            step = 0.001

    # final local polish
    best_c = _polish(best_c, time_budget * 0.3)
    return best_c, _radius(best_c)


def _polish(c, time_budget):
    """Coordinate-descent style polish: nudge each centre to reduce the energy
    at a fixed target radius equal to current best."""
    c = c.copy()
    best_r = _radius(c)
    start = time.time()
    deadline = start + time_budget
    r_target = best_r
    step = 0.01
    rng = np.random.default_rng(4242)
    while time.time() < deadline:
        improved = False
        for i in range(N):
            e0 = _energy(c, r_target)
            for axis in range(3):
                for sgn in (1, -1):
                    c2 = c.copy()
                    c2[i, axis] += sgn * step
                    c2 = _clamp(c2)
                    e1 = _energy(c2, r_target)
                    if e1 < e0:
                        c = c2
                        e0 = e1
                        improved = True
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
        cur = _radius(c)
        if cur > best_r:
            best_r = cur
            r_target = cur
    return c


def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin under 150s

    c = _seed()
    best_c = c.copy()
    best_r = _radius(c)

    # a couple of restarts from different seeds, each with a time slice
    seeds = [_seed(), _seed(), _seed()]
    seeds[0] = c
    per = (deadline - time.time()) / len(seeds)
    for s in seeds:
        if time.time() > deadline - 15:
            break
        cand, r = _sa_optimize(s, per)
        if r > best_r:
            best_r = r
            best_c = cand

    # final polish with remaining time
    if time.time() < deadline - 5:
        best_c = _polish(best_c, deadline - time.time() - 5)
        best_r = _radius(best_c)

    return _clamp(best_c)
