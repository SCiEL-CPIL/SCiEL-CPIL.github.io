"""Combined program: integer/rational integral point set search with exact verification."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

# ---------- exact arithmetic helpers ----------

def _fmt(num, den):
    num = int(num); den = int(den)
    if den <= 0:
        num, den = -num, -den
    if den == 1:
        return [num, 1]
    return [num, den]

def _fmt_frac(f):
    return _fmt(f.numerator, f.denominator)

def _is_square_int(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _dist_sq(a, b, D):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + D * dy * dy

def _distance_rational(a, b, D):
    v = _dist_sq(a, b, D)
    # distance rational iff v is a rational square
    if v < 0:
        return False
    # v = num/den ; need num*den to be a square (in Q)
    num = v.numerator
    den = v.denominator
    # reduce
    if num == 0:
        return True
    # v is square in Q iff num*den is a square integer (after canceling)
    prod = num * den
    return _is_square_int(prod)

def _collinear(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]) == 0

def _concyclic4(p, q, r, s):
    x1, y1 = p; x2, y2 = q; x3, y3 = r; x4, y4 = s
    a1 = x1*x1 + y1*y1
    a2 = x2*x2 + y2*y2
    a3 = x3*x3 + y3*y3
    a4 = x4*x4 + y4*y4
    b1 = a2 - a1; c1 = x2 - x1; d1 = y2 - y1
    b2 = a3 - a1; c2 = x3 - x1; d2 = y3 - y1
    b3 = a4 - a1; c3 = x4 - x1; d3 = y4 - y1
    det = (b1 * (c2 * d3 - c3 * d2)
           - c1 * (b2 * d3 - b3 * d2)
           + d1 * (b2 * c3 - b3 * c2))
    return det == 0

def _valid_add(pt, pts, D):
    for b in pts:
        if not _distance_rational(pt, b, D):
            return False
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if _collinear(pt, pts[i], pts[j]):
                return False
            for k in range(j + 1, n):
                if _concyclic4(pt, pts[i], pts[j], pts[k]):
                    return False
    return True

def _verify(pts, D):
    if len(pts) < 3 or len(pts) > 12:
        return False
    if len(set(pts)) != len(pts):
        return False
    if D < 1 or D > 10**12:
        return False
    for p in pts:
        for c in (p[0], p[1]):
            if abs(c.numerator) > 10**40 or c.denominator > 10**40:
                return False
    n = len(pts)
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
                    if _concyclic4(pts[i], pts[j], pts[k], pts[l]):
                        return False
    return True

def _is_squarefree(n):
    if n < 2:
        return n == 1
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            m //= p
            if m % p == 0:
                return False
        p += 1
    return True

# ---------- search ----------

_KNOWN5 = [(Fraction(0), Fraction(0)),
           (Fraction(-56), Fraction(33)),
           (Fraction(-16), Fraction(63)),
           (Fraction(0), Fraction(33)),
           (Fraction(16), Fraction(63))]

def _try_build(D, rng, max_coord, max_pool, max_iters):
    pts = list(_KNOWN5)
    pool = set()
    for _ in range(max_pool):
        x = rng.randint(-max_coord, max_coord)
        y = rng.randint(-max_coord, max_coord)
        pool.add((Fraction(x), Fraction(y)))
    iters = 0
    while iters < max_iters and _time_left() > 8 and len(pts) < 12:
        iters += 1
        cand = rng.choice(tuple(pool))
        if any(cand == p for p in pts):
            continue
        if _valid_add(cand, pts, D):
            pts.append(cand)
    # local improvement: expand pool, keep adding
    N = max_coord
    stuck = True
    while not stuck and _time_left() > 8 and len(pts) < 12:
        stuck = True
        sample = rng.sample(tuple(pool), min(len(pool), 80))
        for cand in sample:
            if any(cand == p for p in pts):
                continue
            if _valid_add(cand, pts, D):
                pts.append(cand)
                stuck = False
                break
        if stuck:
            N += 15
            for _ in range(800):
                x = rng.randint(-N, N)
                y = rng.randint(-N, N)
                pool.add((Fraction(x), Fraction(y)))
    return pts

def _try_build_rational(D, rng, max_coord, max_pool, max_iters):
    """Try small-denominator rational coordinates to escape integer lattice limits."""
    pts = list(_KNOWN5)
    dens = [1, 2, 3, 4, 5, 6]
    pool = set()
    for _ in range(max_pool):
        x = Fraction(rng.randint(-max_coord, max_coord), rng.choice(dens))
        y = Fraction(rng.randint(-max_coord, max_coord), rng.choice(dens))
        pool.add((x, y))
    iters = 0
    while iters < max_iters and _time_left() > 8 and len(pts) < 12:
        iters += 1
        cand = rng.choice(tuple(pool))
        if any(cand == p for p in pts):
            continue
        if _valid_add(cand, pts, D):
            pts.append(cand)
    return pts

def construct():
    best = list(_KNOWN5)
    best_D = 1
    best_n = len(best)

    Ds = [1, 2, 3, 5, 6, 10, 13]
    Ds = [d for d in Ds if _is_squarefree(d) and d <= 10**12]

    seed = 1
    while _time_left() > 12:
        rng = random.Random(seed)
        # integer build
        try:
            for D in Ds[:4]:
                if _time_left() < 12:
                    break
                pts = _try_build(D, rng, max_coord=120, max_pool=2500, max_iters=150000)
                if _verify(pts, D) and len(pts) > best_n:
                    best, best_D, best_n = pts, D, len(pts)
        except Exception:
            pass
        # rational build (less frequent)
        if seed % 2 == 0 and _time_left() > 12:
            try:
                for D in Ds[:2]:
                    if _time_left() < 12:
                        break
                    pts = _try_build_rational(D, rng, max_coord=60, max_pool=2000, max_iters=80000)
                    if _verify(pts, D) and len(pts) > best_n:
                        best, best_D, best_n = pts, D, len(pts)
            except Exception:
                pass
        seed += 1

    # final safety: ensure best is valid, else fall back
    if not _verify(best, best_D):
        best, best_D = _KNOWN5, 1

    return {'D': best_D,
            'points': [[_fmt_frac(p[0]), _fmt_frac(p[1])] for p in best]}
# EVOLVE-BLOCK-END
