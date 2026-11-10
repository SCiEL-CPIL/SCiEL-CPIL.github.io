"""Construct integral point sets (no three collinear, no four concyclic)."""
from fractions import Fraction
import random
import time
import math

_START = time.time()
_DEADLINE = 135.0  # safety margin vs 150s limit

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
    dsq = _dist_sq_D(a, b, D)
    return _is_perfect_square_frac(dsq)

def _collinear(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]) == 0

def _concyclic4(p, q, r, s, D):
    x1, y1 = p[0], p[1]
    x2, y2 = q[0], q[1]
    x3, y3 = r[0], r[1]
    x4, y4 = s[0], s[1]
    
    w1 = x1 * x1 + D * y1 * y1
    w2 = x2 * x2 + D * y2 * y2
    w3 = x3 * x3 + D * y3 * y3
    w4 = x4 * x4 + D * y4 * y4
    
    def _three_collinear_xw(xa, wa, xb, wb, xc, wc):
        return (xb - xa) * (wc - wa) - (xc - xa) * (wb - wa) == 0
    
    if x1 == x2 == x3 == x4:
        return w1 == w2 == w3 == w4
    
    pts_xw = [(x1, w1), (x2, w2), (x3, w3), (x4, w4)]
    
    for i in range(4):
        for j in range(i+1, 4):
            for k in range(j+1, 4):
                if _three_collinear_xw(*pts_xw[i], *pts_xw[j], *pts_xw[k]):
                    m = [l for l in range(4) if l not in (i,j,k)][0]
                    if _three_collinear_xw(*pts_xw[i], *pts_xw[j], *pts_xw[m]):
                        return True
    return False


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


def _try_build_lattice(seed, D=1):
    rng = random.Random(seed)
    N = 100
    pool = set()
    for _ in range(5000):
        x = rng.randint(-N, N)
        y = rng.randint(-N, N)
        pool.add((Fraction(x), Fraction(y)))
    
    starts = [
        [(Fraction(0), Fraction(0)), (Fraction(0), Fraction(3)), (Fraction(4), Fraction(0)), (Fraction(1), Fraction(3))],
        [(Fraction(0), Fraction(0)), (Fraction(5), Fraction(0)), (Fraction(0), Fraction(5)), (Fraction(3), Fraction(4))],
        [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)), (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33))],
    ]
    pts = list(rng.choice(starts))
    
    pool_list = list(pool)
    rng.shuffle(pool_list)
    
    iters = 0
    max_iters = 50000
    while iters < max_iters and _time_left() > 10:
        iters += 1
        cand = pool_list[iters % len(pool_list)]
        if any(cand == p for p in pts):
            continue
        if _valid_add(cand, pts, D):
            pts.append(cand)
            if len(pts) >= 12:
                break
    
    return pts


def _try_build_heronian(seed):
    rng = random.Random(seed)
    D = 1
    
    base_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (5, 0), (0, 5), (3, 4), (4, 3)],
    ]
    
    base = rng.choice(base_sets)
    pts = [(Fraction(x), Fraction(y)) for x, y in base]
    
    N = 150
    attempts = 0
    max_attempts = 30000
    while attempts < max_attempts and _time_left() > 10 and len(pts) < 12:
        attempts += 1
        x = rng.randint(-N, N)
        y = rng.randint(-N, N)
        cand = (Fraction(x), Fraction(y))
        if any(cand == p for p in pts):
            continue
        if _valid_add(cand, pts, D):
            pts.append(cand)
    
    return pts


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
    
    strategies = [
        lambda s: _try_build_lattice(s, D=1),
        lambda s: _try_build_heronian(s),
    ]
    
    seed = 1
    while _time_left() > 15:
        for strat in strategies:
            if _time_left() < 15:
                break
            try:
                pts = strat(seed)
            except Exception:
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
