"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import math
from itertools import combinations

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140

def _elapsed():
    return time.time() - _START_TIME

def _deadline_check():
    return _elapsed() > _DEADLINE

def _sq(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _dist2_rational(p1, p2, D):
    x1, xd1, y1, yd1 = p1
    x2, xd2, y2, yd2 = p2
    dx_num = x1 * xd2 - x2 * xd1
    dx_den = xd1 * xd2
    dy_num = y1 * yd2 - y2 * yd1
    dy_den = yd1 * yd2
    num = dx_num * dx_num * dy_den * dy_den + D * dy_num * dy_num * dx_den * dx_den
    den = dx_den * dx_den * dy_den * dy_den
    if den == 0 or num < 0:
        return False
    return _sq(num * den)

def _collinear(p1, p2, p3):
    x1, xd1, y1, yd1 = p1
    x2, xd2, y2, yd2 = p2
    x3, xd3, y3, yd3 = p3
    lhs = (x2 * xd1 - x1 * xd2) * (y3 * yd1 - y1 * yd3) * yd2 * xd3
    rhs = (x3 * xd1 - x1 * xd3) * (y2 * yd1 - y1 * yd2) * yd3 * xd2
    return lhs == rhs

def _concyclic(p1, p2, p3, p4, D):
    X1, Y1 = Fraction(p1[0], p1[1]), Fraction(p1[2], p1[3])
    X2, Y2 = Fraction(p2[0], p2[1]), Fraction(p2[2], p2[3])
    X3, Y3 = Fraction(p3[0], p3[1]), Fraction(p3[2], p3[3])
    X4, Y4 = Fraction(p4[0], p4[1]), Fraction(p4[2], p4[3])
    S1 = X1*X1 + D*Y1*Y1
    S2 = X2*X2 + D*Y2*Y2
    S3 = X3*X3 + D*Y3*Y3
    S4 = X4*X4 + D*Y4*Y4
    dA = X2 - X1
    dB = Y2 - Y1
    dC = X3 - X1
    dD = Y3 - Y1
    rhs1 = -(S2 - S1)
    rhs2 = -(S3 - S1)
    det = dA * dD - dC * dB
    if det == 0:
        return False
    A_val = (rhs1 * dD - rhs2 * dB) / det
    B_val = (dA * rhs2 - dC * rhs1) / det
    C_val = -S1 - A_val * X1 - B_val * Y1
    return S4 + A_val * X4 + B_val * Y4 + C_val == 0

def _validate(points, D):
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i+1, n):
            if not _dist2_rational(points[i], points[j], D):
                return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k]):
                    return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _concyclic(points[i], points[j], points[k], points[l], D):
                        return False
    return True

def _known_5():
    pts = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    return [(x, 1, y, 1) for x, y in pts]

def _try_add_point(current, D, candidates):
    """Try to add one point from candidates to current set."""
    for cand in candidates:
        if _deadline_check():
            return None
        if any(cand[0] == c[0] and cand[1] == c[1] and cand[2] == c[2] and cand[3] == c[3] for c in current):
            continue
        # Check distance to all existing points
        if not all(_dist2_rational(cand, p, D) for p in current):
            continue
        # Check collinearity
        if any(_collinear(cand, a, b) for a, b in combinations(current, 2)):
            continue
        # Check concyclicity
        if any(_concyclic(cand, a, b, c, D) for a, b, c in combinations(current, 3)):
            continue
        return cand
    return None

def _search_6(D, max_coord=30):
    """Search for 6 points with given D using a greedy approach."""
    candidates = [(x, 1, y, 1) for x in range(-max_coord, max_coord+1) for y in range(-max_coord, max_coord+1)]
    
    best = None
    best_size = 0
    
    # Try different starting triples
    import random
    rng = random.Random(12345)
    for trial in range(200):
        if _deadline_check():
            break
        idxs = rng.sample(range(len(candidates)), 3)
        current = [candidates[i] for i in idxs]
        if not all(_dist2_rational(current[i], current[j], D) for i in range(3) for j in range(i+1, 3)):
            continue
        if any(_collinear(current[i], current[j], current[k]) for i in range(3) for j in range(i+1,3) for k in range(j+1,3)):
            continue
        
        # Try to extend
        for _ in range(3):  # try to add 3 more points
            if _deadline_check():
                break
            added = _try_add_point(current, D, candidates)
            if added is None:
                break
            current.append(added)
        
        if len(current) > best_size:
            best_size = len(current)
            best = current[:]
            if best_size >= 6:
                break
    
    if best and len(best) >= 6:
        if _validate(best, D):
            return best
    return None

def construct():
    # Fallback: known 5-point solution
    pts5 = _known_5()
    D_best = 1
    pts_best = pts5
    
    # Try to find 6 points with D=1
    if not _deadline_check():
        result = _search_6(1, max_coord=20)
        if result:
            pts_best = result
            D_best = 1
    
    # Try D=2
    if not _deadline_check() and len(pts_best) < 6:
        result = _search_6(2, max_coord=15)
        if result:
            pts_best = result
            D_best = 2
    
    # Try D=3
    if not _deadline_check() and len(pts_best) < 6:
        result = _search_6(3, max_coord=12)
        if result:
            pts_best = result
            D_best = 3
    
    return {'D': D_best, 'points': [[[x, xd], [y, yd]] for x, xd, y, yd in pts_best]}
# EVOLVE-BLOCK-END
