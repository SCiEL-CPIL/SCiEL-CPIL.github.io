"""Circle packing in unit square - hard-disk parallel tempering + greedy inflation."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 26
DEADLINE = 110.0  # safety margin below 120s evaluator / 150s kill
EPS = 1e-9
R_MIN = 1e-4
R_MAX = 0.5


def _valid(x, y, r):
    n = len(x)
    if n == 0:
        return True
    if np.any(r < R_MIN) or np.any(r > R_MAX + 1e-9):
        return False
    if np.any(x < -r - EPS) or np.any(x > 1.0 + r + EPS):
        return False
    if np.any(y < -r - EPS) or np.any(y > 1.0 + r + EPS):
        return False
    d = cKDTree(np.column_stack([x, y])).query_pairs(r + r, eps=1e-9)
    for i, j in d:
        if (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 < (r[i] + r[j]) ** 2 - 1e-9:
            return False
    return True


def _score(x, y, r):
    return float(np.sum(r))


def _init(seed):
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    base_r = 0.5 / side - 1e-3
    x = np.array([c[0] for c in cells[:N]], dtype=float)
    y = np.array([c[1] for c in cells[:N]], dtype=float)
    r = np.full(N, base_r, dtype=float)
    return x, y, r


def _refine(x, y, r, budget):
    """Deterministic greedy local refinement: grow radii / nudge to relieve overlaps."""
    order = np.argsort(-r)
    for _ in range(4000):
        if time.time() > budget:
            break
        improved = False
        for i in order:
            # try to grow circle i
            base = r[i]
            # find limiting expansion
            lim = (1.0 - x[i]) if (1.0 - x[i]) < x[i] else x[i]
            lim = min(lim, (1.0 - y[i]) if (1.0 - y[i]) < y[i] else y[i])
            lim = min(lim, R_MAX)
            for j in range(N):
                if i == j:
                    continue
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                d = np.hypot(dx, dy)
                if d > 1e-12:
                    lim = min(lim, (d - r[j]) / 2.0)
            newr = min(lim, base * 1.02)
            if newr > base + 1e-7:
                r[i] = newr
                improved = True
        if not improved:
            # try small nudges to break symmetry / relieve tight contacts
            moved = False
            for i in order:
                for _ in range(4):
                    ang = np.random.rand() * 2 * np.pi
                    step = 0.002 * r[i]
                    nx = np.clip(x[i] + step * np.cos(ang), 0, 1)
                    ny = np.clip(y[i] + step * np.sin(ang), 0, 1)
                    x[i], y[i] = nx, ny
                    if _valid(x, y, r):
                        moved = True
                        break
                    else:
                        x[i], y[i] = nx - step * np.cos(ang), ny - step * np.sin(ang)
            if not moved:
                break
    # final inflation sweep
    for _ in range(200):
        if time.time() > budget:
            break
        improved = False
        for i in range(N):
            base = r[i]
            lim = (1.0 - x[i]) if (1.0 - x[i]) < x[i] else x[i]
            lim = min(lim, (1.0 - y[i]) if (1.0 - y[i]) < y[i] else y[i])
            lim = min(lim, R_MAX)
            for j in range(N):
                if i == j:
                    continue
                d = np.hypot(x[j] - x[i], y[j] - y[i])
                if d > 1e-12:
                    lim = min(lim, (d - r[j]) / 2.0)
            newr = min(lim, base * 1.01)
            if newr > base + 1e-7:
                r[i] = newr
                improved = True
        if not improved:
            break
    return x, y, r


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE

    # replicas at different "temperatures" (step sizes)
    n_rep = 8
    # temperatures: higher = larger moves (exploration), lower = fine (exploitation)
    temps = np.linspace(0.02, 0.15, n_rep)

    # initialize each replica from a grid with slight randomization
    states = []  # each: (x, y, r, score)
    for k in range(n_rep):
        x, y, r = _init(seed=1000 + k)
        rng = np.random.default_rng(2000 + k)
        # randomize a bit to decorrelate replicas
        x = np.clip(x + rng.normal(0, 0.02, N), 0, 1)
        y = np.clip(y + rng.normal(0, 0.02, N), 0, 1)
        r = np.clip(r * (1 + rng.normal(0, 0.05, N)), R_MIN, R_MAX)
        if not _valid(x, y, r):
            x, y, r = _init(seed=1000 + k)
        states.append([x.copy(), y.copy(), r.copy(), _score(x, y, r)])

    best = states[0][3]
    best_state = (states[0][0].copy(), states[0][1].copy(), states[0][2].copy())

    rng = np.random.default_rng(42)

    # parallel tempering loop
    while time.time() < deadline - 5.0:
        # propose moves on each replica
        for k in range(n_rep):
            x, y, r, s = states[k]
            temp = temps[k]
            # pick a random circle
            i = rng.integers(N)
            move_type = rng.integers(3)
            x0, y0, r0 = x[i], y[i], r[i]
            if move_type == 0:
                # move center
                dx = rng.normal(0, temp * 0.5)
                dy = rng.normal(0, temp * 0.5)
                x[i] = np.clip(x0 + dx, 0, 1)
                y[i] = np.clip(y0 + dy, 0, 1)
            elif move_type == 1:
                # change radius
                r[i] = np.clip(r0 * (1 + rng.normal(0, temp)), R_MIN, R_MAX)
            else:
                # grow radius greedily if possible
                lim = min(x0, 1.0 - x0, y0, 1.0 - y0, R_MAX)
                for j in range(N):
                    if i == j:
                        continue
                    d = np.hypot(x[j] - x[i], y[j] - y[i])
                    if d > 1e-12:
                        lim = min(lim, (d - r[j]) / 2.0)
                r[i] = min(r0 * 1.03, lim)
            if _valid(x, y, r):
                s = _score(x, y, r)
                states[k] = [x, y, r, s]
                if s > best:
                    best = s
                    best_state = (x.copy(), y.copy(), r.copy())
            else:
                # revert
                x[i], y[i], r[i] = x0, y0, r0

        # replica exchange (swap states between adjacent temps if it improves total)
        for k in range(n_rep - 1):
            if rng.rand() < 0.5:
                # swap states k and k+1 (they are independent configurations)
                states[k], states[k + 1] = states[k + 1], states[k]
                # update best
                for st in states:
                    if st[3] > best:
                        best = st[3]
                        best_state = (st[0].copy(), st[1].copy(), st[2].copy())

    # refine the best state found
    x, y, r = best_state
    x, y, r = _refine(x, y, r, deadline - 0.5)
    if _valid(x, y, r):
        s = _score(x, y, r)
        if s > best:
            best = s
            best_state = (x.copy(), y.copy(), r.copy())
    else:
        # fallback to a valid state
        for st in states:
            if _valid(st[0], st[1], st[2]):
                if st[3] > best:
                    best = st[3]
                    best_state = (st[0].copy(), st[1].copy(), st[2].copy())

    x, y, r = best_state
    # final safety: ensure validity
    if not _valid(x, y, r):
        x, y, r = _init(0)
        x, y, r = _refine(x, y, r, deadline - 0.1)

    return np.column_stack([x, y, r])
