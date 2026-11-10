"""Construct integral point sets (no three collinear, no four concyclic).

Combines:
  - current program: known-set seeds + greedy extension + D=1 + exact verify
  - inspiration 2: pool of candidates, expanding radius, local swap/repair,
    power-based concyclicity test
"""
from fractions import Fraction
import random
import time
import math

_START = time.time()
_DEADLINE = 135.0  # safety margin vs 150s kill


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _fmt(num, den):
    if den <= 0:
        num, den = -num, -den
    if den == 1:
        return [int(num), 1]
    return [int(num), int(den)]


def _dist_sq_D(a, b, D):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + D * dy * dy


def _is_perfect_square_frac(f):
    if f < 0:
        return False
    n, d = f.numerator, f.denominator
    if n == 0:
        return True
    sn = math.isqrt(n)
    sd = math.isqrt(d)
    return sn * sn == n and sd * sd == d


def _distance_rational(a, b, D):
    return _is_perfect_square_frac(_dist_sq_D(a, b, D))


def _collinear(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]) == 0


def _concyclic4(p, q, r, s, D):
    """Four points (x_i, y_i*sqrt(D)) concyclic iff (x_i, x_i^2 + D y_i^2)
    are collinear in the (x, w) plane."""
    x1, y1 = p[0], p[1]
    x2, y2 = q[0], q[1]
    x3, y3 = r[0], r[1]
    x4, y4 = s[0], s[1]
    w1 = x1 * x1 + D * y1 * y1
    w2 = x2 * x2 + D * y2 * y2
    w3 = x3 * x3 + D * y3 * y3
    w4 = x4 * x4 + D * y4 * y4
    d123 = (x2 - x1) * (w3 - w1) - (x3 - x1) * (w2 - w1)
    if d123 != 0:
        return False
    d124 = (x2 - x1) * (w4 - w1) - (x4 - x1) * (w2 - w1)
    return d124 == 0


def _valid_add(pt, pts, D):
    for b in pts:
        if not _distance_rational(pt, b, D):
            return False
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if _collinear(pt, pts[i], pts[j]):
                return False
            for k in range(j + 1, len(pts)):
                if _concyclic4(pt, pts[i], pts[j], pts[k], D):
                    return False
    return True


# Known integral sets (D=1): (x, y) with integer pairwise distances.
_KNOWN_SETS = [
    [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
    [(0, 0), (5, 0), (0, 5), (3, 4), (4, 3)],
    [(0, 0), (13, 0), (0, 13), (13, 13), (8, 13), (0, 8)],
    [(0, 0), (13, 0), (0, 13), (13, 13), (8, 13), (0, 8), (13, 8)],
    [(0, 0), (15, 0), (0, 15), (15, 15), (9, 15), (0, 9)],
    [(0, 0), (5, 0), (0, 5), (3, 4)],
    # additional 6-point candidates
    [(0, 0), (12, 0), (0, 12), (12, 12), (6, 12), (0, 6)],
    [(0, 0), (20, 0), (0, 20), (20, 20), (12, 20), (0, 12)],
    [(0, 0), (9, 0), (0, 9), (9, 9), (5, 9), (0, 5)],
]


def _verify_set(pts, D=1):
    n = len(pts)
    if n < 3:
        return False
    if len(set(pts)) != n:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if not _distance_rational(pts[i], pts[j], D):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(pts[i], pts[j], pts[k]):
                    return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _concyclic4(pts[i], pts[j], pts[k], pts[l], D):
                        return False
    return True


def _build_pool(rng, N, D):
    """Pool of integer points within [-N, N]^2 that have at least one
    integer distance to the origin (cheap prefilter)."""
    pool = set()
    for x in range(-N, N + 1):
        for y in range(-N, N + 1):
            # integer distance to origin?
            if (x * x + y * y) ** 0.5 == int((x * x + y * y) ** 0.5):
                pool.add((Fraction(x), Fraction(y)))
    return pool


def _try_build(seed):
    rng = random.Random(seed)
    D = 1
    best_pts = None
    best_len = 0

    for base_set in _KNOWN_SETS:
        if _time_left() < 12:
            break
        pts = [(Fraction(x), Fraction(y)) for x, y in base_set]
        if not _verify_set(pts, D):
            continue

        # Greedy extension with pool + expanding radius.
        N = 40
        pool = _build_pool(rng, N, D)
        max_iters = 30000
        iters = 0
        stuck_rounds = 0
        while iters < max_iters and _time_left() > 10 and len(pts) < 12:
            iters += 1
            cand = rng.choice(tuple(pool))
            if any(cand == p for p in pts):
                continue
            if _valid_add(cand, pts):
                pts.append(cand)
                stuck_rounds = 0
            else:
                stuck_rounds += 1
            if stuck_rounds > 300:
                # expand pool
                N += 15
                new_pool = _build_pool(rng, N, D)
                for p in new_pool:
                    if p not in pool:
                        pool.add(p)
                stuck_rounds = 0

        # Repair / swap pass: try removing one point and adding a better one.
        if _time_left() > 12 and len(pts) < 12:
            for _ in range(60):
                if _time_left() < 10 or len(pts) >= 12:
                    break
                # try to add from a fresh random candidate
                x = rng.randint(-200, 200)
                y = rng.randint(-200, 200)
                cand = (Fraction(x), Fraction(y))
                if any(cand == p for p in pts):
                    continue
                if _valid_add(cand, pts):
                    pts.append(cand)
                    break
                # else try a swap: drop a random point, add cand if it fits
                if len(pts) >= 3:
                    drop = rng.randrange(len(pts))
                    trial = pts[:drop] + pts[drop + 1:]
                    if _valid_add(cand, trial, D):
                        pts = trial + [cand]
                        break

        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts[:]
            if best_len >= 12:
                return best_pts

    if best_pts is None or best_len < 3:
        best_pts = [(Fraction(0), Fraction(0)), (Fraction(5), Fraction(0)),
                    (Fraction(0), Fraction(5))]
    return best_pts


def _finalize(pts, D=1):
    return {'D': D, 'points': [[_fmt(p[0].numerator, p[0].denominator),
                                _fmt(p[1].numerator, p[1].denominator)] for p in pts]}


def _verify(res):
    D = res['D']
    pts_raw = res['points']
    pts = []
    for xraw, yraw in pts_raw:
        x = Fraction(xraw[0], xraw[1])
        y = Fraction(yraw[0], yraw[1])
        if abs(x.numerator) > 10**40 or x.denominator > 10**40:
            return False
        if abs(y.numerator) > 10**40 or y.denominator > 10**40:
            return False
        pts.append((x, y))
    if len(pts) < 3 or len(pts) > 12:
        return False
    if len(set(pts)) != len(pts):
        return False
    if D > 10**12 or D < 1:
        return False
    n = D
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % (p * p) == 0:
            return False
        while n % p == 0:
            n //= p
    n_pts = len(pts)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            if not _distance_rational(pts[i], pts[j], D):
                return False
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            for k in range(j + 1, n_pts):
                if _collinear(pts[i], pts[j], pts[k]):
                    return False
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            for k in range(j + 1, n_pts):
                for l in range(k + 1, n_pts):
                    if _concyclic4(pts[i], pts[j], pts[k], pts[l], D):
                        return False
    return True


def _run():
    best = None
    best_score = -1
    seed = 1
    while _time_left() > 12:
        try:
            pts = _try_build(seed)
        except Exception:
            seed += 1
            continue
        res = _finalize(pts, D=1)
        if _verify(res):
            sc = len(pts)
            if sc > best_score:
                best_score = sc
                best = res
                if len(pts) >= 12:
                    return best
        seed += 1

    if best is None:
        points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        best = {'D': 1, 'points': [[_fmt(x, 1), _fmt(y, 1)] for x, y in points]}
    return best


def construct():
    return _run()
