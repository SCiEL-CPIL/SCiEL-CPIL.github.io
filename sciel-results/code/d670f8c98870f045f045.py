"""Fresh design: vectorized greedy set-cover + local reduction for q-ary covering codes."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def _time_ok(deadline):
    return time.time() < deadline


def _tetracode_words(n):
    tet = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n == 4:
        return [list(h) for h in tet]
    return [h + list(tail) for h in tet for tail in itertools.product(range(3), repeat=n - 4)]


def _full_space(q, n):
    return [list(w) for w in itertools.product(range(q), repeat=n)]


def _build_spheres(q, n, radius, deadline):
    total = q ** n
    digits = np.empty((total, n), dtype=np.int8)
    tmp = np.arange(total, dtype=np.int64).copy()
    for i in range(n - 1, -1, -1):
        digits[:, i] = tmp % q
        tmp //= q
    spheres = []
    for c in range(total):
        cd = digits[c]
        d = np.sum(digits != cd, axis=1)
        spheres.append(d <= radius)
        if not _time_ok(deadline):
            return None
    return total, np.array(spheres, dtype=bool)


def _greedy_cover(q, n, radius, deadline, rng):
    total = q ** n
    built = _build_spheres(q, n, radius, deadline)
    if built is None:
        return None
    total, spheres = built
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    while uncovered.any() and _time_ok(deadline):
        gains = (spheres & uncovered).sum(axis=1)
        mx = gains.max()
        if mx == 0:
            break
        cands = np.where(gains == mx)[0]
        pick = int(rng.choice(cands))
        chosen.append(pick)
        uncovered &= ~spheres[pick]
    if uncovered.any():
        return None
    return chosen


def _remove_redundant(q, n, radius, chosen, deadline):
    built = _build_spheres(q, n, radius, deadline)
    if built is None:
        return chosen
    total, spheres = built
    chosen = list(chosen)
    changed = True
    while changed and _time_ok(deadline):
        changed = False
        for i in range(len(chosen)):
            if not _time_ok(deadline):
                break
            others = [spheres[chosen[j]] for j in range(len(chosen)) if j != i]
            if not others:
                continue
            union = np.logical_or.reduce(others)
            if not np.any(spheres[chosen[i]] & ~union):
                chosen.pop(i)
                changed = True
                break
    return chosen


def _is_valid_cover(q, n, radius, words):
    if not words:
        return q ** n == 0
    total = q ** n
    digits = np.empty((total, n), dtype=np.int8)
    tmp = np.arange(total, dtype=np.int64).copy()
    for i in range(n - 1, -1, -1):
        digits[:, i] = tmp % q
        tmp //= q
    covered = np.zeros(total, dtype=bool)
    for w in words:
        cd = np.array(w, dtype=np.int8)
        d = np.sum(digits != cd, axis=1)
        covered |= (d <= radius)
        if covered.all():
            break
    return bool(covered.all())


def construct(q, n, radius):
    deadline = time.time() + 115.0
    total = q ** n
    if total <= 1:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    if radius >= n:
        return [[0] * n]

    best = None
    if q == 3 and n >= 4:
        cand = _tetracode_words(n)
        if _is_valid_cover(q, n, radius, cand):
            best = cand

    if total <= 20000 and _time_ok(deadline):
        for seed in range(8):
            if not _time_ok(deadline - 5):
                break
            rng = random.Random(1000 + seed)
            chosen = _greedy_cover(q, n, radius, deadline, rng)
            if chosen is None:
                continue
            reduced = _remove_redundant(q, n, radius, chosen, deadline)
            if not _is_valid_cover(q, n, radius, reduced):
                continue
            if best is None or len(reduced) < len(best):
                best = reduced
            if not _time_ok(deadline - 15):
                break

    if best is not None:
        return best
    return _full_space(q, n)
# EVOLVE-BLOCK-END
