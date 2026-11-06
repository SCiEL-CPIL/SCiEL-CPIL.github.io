"""Packing 50 equal circles in the unit square [-1/2,1/2]^2.
Strong grid seed + differential evolution + cheap local relaxation,
with an explicit wall-clock deadline guard.
"""
import time
import numpy as np

N = 50
DEADLINE = 138.0  # safety margin before 150 s evaluator kill


def _radius(pts):
    """Score: min(pair-dist/2, wall-dist). O(N^2), vectorized."""
    pts = np.asarray(pts)
    wall_min = (0.5 - np.abs(pts)).min()
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    pair_min = dist.min() / 2.0
    return min(wall_min, pair_min)


def _radius_fast_move(pts, i, newp):
    """Radius after moving point i to newp, assuming other constraints unchanged except those involving i."""
    r_wall = (0.5 - np.abs(newp)).min()
    # distances from newp to all others
    d = pts - newp
    d[i] = 0
    dist = np.sqrt((d ** 2).sum(-1))
    dist[i] = np.inf
    r_pair_i = dist.min() / 2.0
    # other pair min (precomputed outside for speed; here approximate by full)
    return min(r_wall, r_pair_i)


def _seed():
    """8x8 grid minus 14 interior points -> 50 points, radius ~0.0713."""
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # remove 14 interior points symmetrically
    remove = set()
    for i in range(2, 6):
        for j in range(2, 6):
            if len(remove) < 14:
                remove.add((i, j))
                if len(remove) < 14:
                    remove.add((m - 1 - i, m - 1 - j))
    remove = list(remove)[:14]
    keep = [k for k in range(m * m) if (k // m, k % m) not in remove]
    return grid[keep[:N]].copy()


def _local_relax(pts, iters=40):
    """Cheap local relaxation: for each point, push away from nearest neighbour if too close,
    clip to walls. Accept only if global radius doesn't decrease (approx via full check every few)."""
    pts = pts.copy()
    r = _radius(pts)
    for _ in range(iters):
        improved = False
        for i in range(N):
            p = pts[i]
            d = pts - p
            dist = np.sqrt((d ** 2).sum(-1))
            dist[i] = np.inf
            j = dist.argmin()
            gap = dist[j]
            target = 2.0 * r + 1e-5
            if gap < target:
                push = (p - pts[j]) / max(gap, 1e-12)
                step = 0.3 * (target - gap)
                np_ = p + push * step
                np_ = np.clip(np_, -0.5 + r, 0.5 - r)
                trial = pts.copy()
                trial[i] = np_
                rr = _radius(trial)
                if rr >= r - 1e-12:
                    pts = trial
                    if rr > r:
                        r = rr
                    improved = True
        if not improved:
            break
    return pts, r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)

    def now():
        return time.time() - t0

    best = _seed()
    best_r = _radius(best)

    # Differential evolution
    pop_size = 30
    F = 0.8
    CR = 0.7
    pop = np.empty((pop_size, N, 2))
    pop[0] = best
    for k in range(1, pop_size):
        pop[k] = np.clip(best + rng.normal(0, 0.008, (N, 2)), -0.5, 0.5)

    def eval_pop(p):
        return np.array([_radius(p[k]) for k in range(p.shape[0])])

    scores = eval_pop(pop)
    bi = scores.argmax()
    best, best_r = pop[bi].copy(), scores[bi]

    while now() < DEADLINE:
        # DE step
        if now() >= DEADLINE:
            break
        for i in range(pop_size):
            others = [x for x in range(pop_size) if x != i]
            a, b, c = rng.choice(others, 3, replace=False)
            v = pop[a] + F * (pop[b] - pop[c])
            trial = np.empty((N, 2))
            for j in range(N):
                for d in range(2):
                    trial[j, d] = v[j, d] if rng.random() < CR else pop[i][j, d]
            trial = np.clip(trial, -0.5, 0.5)
            tr = _radius(trial)
            if tr >= scores[i]:
                pop[i] = trial
                scores[i] = tr
        bi = scores.argmax()
        if scores[bi] > best_r:
            best, best_r = pop[bi].copy(), scores[bi]

        # periodic local relaxation on best
        if now() < DEADLINE - 8:
            rel, rr = _local_relax(best, iters=20)
            if rr > best_r:
                best, best_r = rel, rr
                # inject into population
                pop[rng.integers(pop_size)] = best

    # final relaxation
    if now() < DEADLINE - 2:
        rel, rr = _local_relax(best, iters=40)
        if rr > best_r:
            best, best_r = rel, rr

    return best
