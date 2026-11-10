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

_KNOWN_SETS = [
    [(0,0), (-56,33), (-16,63), (0,33), (16,63)],
    [(0,0), (5,0), (0,5), (3,4), (4,3)],
    [(0,0), (13,0), (0,13), (13,13), (8,13), (0,8)],
    [(0,0), (13,0), (0,13), (13,13), (8,13), (0,8), (13,8)],
    [(0,0), (-16,63), (0,33), (16,63), (-56,33)],
    [(0,0), (5,0), (0,5), (3,4)],
    [(0,0), (15,0), (0,15), (15,15), (9,15), (0,9)],
    # Additional known configurations
    [(0,0), (7,0), (0,7), (5,5), (3,4), (4,3)],
    [(0,0), (9,0), (0,9), (6,6), (3,4), (4,3), (5,5)],
    [(0,0), (11,0), (0,11), (8,8), (3,4), (4,3), (5,5), (6,6)],
]

def _verify_set(pts, D=1):
    n = len(pts)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i+1, n):
            if not _distance_rational(pts[i], pts[j], D):
                return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(pts[i], pts[j], pts[k]):
                    return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _concyclic4(pts[i], pts[j], pts[k], pts[l], D):
                        return False
    return True

def _try_build(seed):
    rng = random.Random(seed)
    D = 1
    
    best_pts = None
    best_len = 0
    
    for base_set in _KNOWN_SETS:
        pts = [(Fraction(x), Fraction(y)) for x, y in base_set]
        if not _verify_set(pts, D):
            continue
        
        # Multi-stage search: small pool first, then expand
        N = 30
        pool_size = 500
        max_pool = 3000
        
        # Build initial pool
        pool = set()
        for _ in range(pool_size):
            x = rng.randint(-N, N)
            y = rng.randint(-N, N)
            pool.add((Fraction(x), Fraction(y)))
        
        # Greedy incremental build
        while _time_left() > 10 and len(pts) < 12:
            # Try to add from current pool
            added = False
            for cand in list(pool):
                if any(cand == p for p in pts):
                    continue
                if _valid_add(cand, pts, D):
                    pts.append(cand)
                    added = True
                    if len(pts) >= 12:
                        break
            
            if not added and len(pts) < 12:
                # Expand pool if stuck
                N += 15
                for _ in range(200):
                    x = rng.randint(-N, N)
                    y = rng.randint(-N, N)
                    pool.add((Fraction(x), Fraction(y)))
                if len(pool) > max_pool:
                    break
        
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts[:]
            if best_len >= 12:
                return best_pts
    
    # Fallback: random 3-point start
    if best_pts is None or best_len < 3:
        for _ in range(20):
            if _time_left() < 10:
                break
            pts = []
            for _ in range(3):
                x = rng.randint(-50, 50)
                y = rng.randint(-50, 50)
                cand = (Fraction(x), Fraction(y))
                if any(cand == p for p in pts):
                    continue
                if all(_distance_rational(cand, p, D) for p in pts):
                    if all(not _collinear(cand, pts[i], pts[j]) for i in range(len(pts)) for j in range(i+1, len(pts))):
                        pts.append(cand)
            if len(pts) == 3:
                N = 50
                pool = set()
                for _ in range(1000):
                    x = rng.randint(-N, N)
                    y = rng.randint(-N, N)
                    pool.add((Fraction(x), Fraction(y)))
                
                while _time_left() > 10 and len(pts) < 12:
                    added = False
                    for cand in list(pool):
                        if any(cand == p for p in pts):
                            continue
                        if _valid_add(cand, pts, D):
                            pts.append(cand)
                            added = True
                            if len(pts) >= 12:
                                break
                    if not added:
                        N += 20
                        for _ in range(300):
                            x = rng.randint(-N, N)
                            y = rng.randint(-N, N)
                            pool.add((Fraction(x), Fraction(y)))
                
                if len(pts) > best_len:
                    best_len = len(pts)
                    best_pts = pts[:]
    
    return best_pts if best_pts else [(Fraction(0), Fraction(0)), (Fraction(5), Fraction(0)), (Fraction(0), Fraction(5))]

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
    while _time_left() > 15:
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
