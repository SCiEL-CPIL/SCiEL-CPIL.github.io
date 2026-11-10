"""Construct integral point sets (no three collinear, no four concyclic)."""
from fractions import Fraction
import random
import time
import math

_START = time.time()
_DEADLINE = 140.0  # safety margin vs 150s limit

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _fmt(num, den):
    if den <= 0:
        num, den = -num, -den
    if den == 1:
        return [int(num), 1]
    return [int(num), int(den)]

def _point_from_rational(x, y):
    x = Fraction(x)
    y = Fraction(y)
    if abs(x.numerator) > 10**40 or x.denominator > 10**40:
        return None
    if abs(y.numerator) > 10**40 or y.denominator > 10**40:
        return None
    return (x, y)

def _dist_sq(a, b):
    """Squared distance between two points (x, y*sqrt(D)) represented as (x_frac, y_frac).
    Actual squared distance = dx^2 + D*dy^2 where dx = a[0]-b[0], dy = a[1]-b[1]."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx  # We'll handle D separately

def _dist_sq_D(a, b, D):
    """Squared distance in the actual metric: dx^2 + D*dy^2"""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + D * dy * dy

def _is_perfect_square_frac(f):
    """Check if a Fraction is a perfect square of a rational."""
    if f < 0:
        return False
    n, d = f.numerator, f.denominator
    if n == 0:
        return True
    # Check if n and d are perfect squares
    sn = math.isqrt(n)
    sd = math.isqrt(d)
    return sn * sn == n and sd * sd == d

def _distance_rational(a, b, D):
    """Check if distance between a and b is rational (i.e., squared distance is a perfect square rational)."""
    dsq = _dist_sq_D(a, b, D)
    return _is_perfect_square_frac(dsq)

def _collinear(a, b, c):
    """Check if three points are collinear (in the affine plane, ignoring D)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]) == 0

def _concyclic4(p, q, r, s, D):
    """Check if four points are concyclic.
    
    Four points are concyclic iff the cross ratio of their distances is real,
    or equivalently, the circle through any three has the fourth on it.
    
    For points in the plane with metric dx^2 + D*dy^2, we need to be careful.
    Actually, the "circle" concept here is in the standard Euclidean plane where
    point i is at (x_i, y_i * sqrt(D)). So the actual coordinates are (x_i, y_i*sqrt(D)).
    
    Let me re-derive: the actual point is (x_i, y_i * sqrt(D)).
    So actual squared distance between i and j is (x_i-x_j)^2 + D*(y_i-y_j)^2.
    
    For concyclicity, we work in the actual Euclidean plane with coordinates (x_i, y_i*sqrt(D)).
    The determinant condition for four points (X1,Y1),...,(X4,Y4) to be concyclic:
    
    |X1^2+Y1^2  X1  Y1  1|
    |X2^2+Y2^2  X2  Y2  1| = 0
    |X3^2+Y3^2  X3  Y3  1|
    |X4^2+Y4^2  X4  Y4  1|
    
    where Xi = x_i, Yi = y_i * sqrt(D).
    
    But Xi^2 + Yi^2 = x_i^2 + D*y_i^2, which involves sqrt(D) in the Y terms.
    Actually no: X_i^2 + Y_i^2 = x_i^2 + D*y_i^2, which is rational if x_i, y_i are rational.
    
    The determinant has entries:
    Row i: [x_i^2 + D*y_i^2, x_i, y_i*sqrt(D), 1]
    
    This determinant will involve sqrt(D) terms. For it to be zero, both the rational and irrational parts must be zero.
    
    Let me expand: the determinant is a polynomial in sqrt(D). Write it as A + B*sqrt(D) = 0.
    Then we need A = 0 and B = 0.
    
    Alternatively, we can use the condition that four points are concyclic iff the Ptolemy-like
    condition holds, or use the power of a point.
    
    A simpler approach: use the fact that four points are concyclic iff the circle through three
    contains the fourth. Compute the circle through p, q, r (in actual coords) and check if s is on it.
    
    Circle through (x1,y1), (x2,y2), (x3,y3) in actual coords where yi = yi_frac * sqrt(D):
    x^2 + y^2 + Ax + By + C = 0
    
    For point i: x_i^2 + D*y_i^2 + A*x_i + B*y_i*sqrt(D) + C = 0
    
    This gives: (x_i^2 + D*y_i^2 + C) + A*x_i + B*y_i*sqrt(D) = 0
    
    For this to hold for three points, we need:
    x_i^2 + D*y_i^2 + C + A*x_i = 0  (rational part)
    B*y_i = 0  (irrational part, for all i)
    
    Wait, that's not right. The equation is:
    x_i^2 + D*y_i^2 + A*x_i + B*y_i*sqrt(D) + C = 0
    
    Grouping: (x_i^2 + D*y_i^2 + C + A*x_i) + B*y_i*sqrt(D) = 0
    
    For this to be zero for rational x_i, y_i and irrational sqrt(D), we need:
    B*y_i = 0 for all i (irrational part)
    x_i^2 + D*y_i^2 + C + A*x_i = 0 for all i (rational part)
    
    If B != 0, then y_i = 0 for all points on the circle, which is a degenerate case.
    
    Hmm, this is getting complicated. Let me just use the determinant approach with exact arithmetic.
    
    Actually, the cleanest way: the four points (x_i, y_i*sqrt(D)) for i=1..4 are concyclic iff
    the determinant below is zero:
    
    det = |x1^2 + D*y1^2   x1   y1*sqrt(D)   1|
          |x2^2 + D*y2^2   x2   y2*sqrt(D)   1|
          |x3^2 + D*y3^2   x3   y3*sqrt(D)   1|
          |x4^2 + D*y4^2   x4   y4*sqrt(D)   1|
    
    Expand this determinant as a polynomial in sqrt(D). Since sqrt(D)^2 = D, it's of the form:
    det = P + Q*sqrt(D)
    
    where P and Q are rational expressions in x_i, y_i, D.
    
    Four points are concyclic iff P = 0 AND Q = 0.
    
    Let me compute P and Q.
    
    Actually, let me just compute the determinant symbolically. Let t = sqrt(D).
    
    Row i: [x_i^2 + D*y_i^2, x_i, y_i*t, 1]
    
    The determinant is a multilinear function. Let me expand along the last column.
    
    det = sum_{i=1}^4 (-1)^{i+4} * 1 * M_{i4}
    
    where M_{i4} is the minor (remove row i, column 4).
    
    This is getting messy. Let me use a different approach.
    
    Alternative: Four points are concyclic iff the oriented angle conditions hold, or equivalently,
    the cross ratio (z1-z3)(z2-z4)/((z1-z4)(z2-z3)) is real, where z_i = x_i + i*y_i*sqrt(D).
    
    But that's also complex.
    
    Simplest practical approach: use the fact that for the specific case where we're working with
    rational x, y and squarefree D, we can check concyclicity by computing the circle through three
    points and checking if the fourth satisfies it, using exact arithmetic.
    
    Let me just implement it by computing the determinant directly with Fraction arithmetic,
    treating sqrt(D) as a symbol and checking if both the rational and sqrt(D) coefficients vanish.
    """
    # Use the determinant condition. Points in actual coords: (x_i, y_i * sqrt(D))
    # Let t = sqrt(D). Point i has coords (x_i, y_i * t).
    # Circle: X^2 + Y^2 + aX + bY + c = 0
    # For point i: x_i^2 + D*y_i^2 + a*x_i + b*y_i*t + c = 0
    # => (x_i^2 + D*y_i^2 + c + a*x_i) + b*y_i*t = 0
    # For this to hold: b*y_i = 0 and x_i^2 + D*y_i^2 + c + a*x_i = 0
    #
    # If all y_i = 0, degenerate. Otherwise, b = 0 and we have:
    # x_i^2 + D*y_i^2 + c + a*x_i = 0 for i = 1,2,3,4
    # This is a system in a, c. For 4 points to be on such a "circle" (which is actually
    # a conic x^2 + a*x + D*y^2 + c = 0, i.e., an ellipse/line in the (x,y) plane with the
    # metric scaled), we need the 4x2 system to be consistent.
    #
    # Wait, I think I'm overcomplicating. In the ACTUAL Euclidean plane, the points are
    # at (x_i, y_i*sqrt(D)). A standard circle in this plane has equation:
    # X^2 + Y^2 + aX + bY + c = 0
    # Substituting X = x_i, Y = y_i*sqrt(D):
    # x_i^2 + D*y_i^2 + a*x_i + b*y_i*sqrt(D) + c = 0
    #
    # For rational x_i, y_i and squarefree D > 1, sqrt(D) is irrational.
    # So we need: b*y_i = 0 for all i, and x_i^2 + D*y_i^2 + a*x_i + c = 0 for all i.
    #
    # If D > 1 and not all y_i are zero, then b = 0.
    # Then we need: x_i^2 + D*y_i^2 + a*x_i + c = 0 for i=1,2,3,4.
    # This is a linear system in (a, c) with 4 equations. For consistency,
    # the 4 points must lie on a conic of the form x^2 + D*y^2 + ax + c = 0.
    #
    # So four points are concyclic (in the actual plane) iff:
    # 1. b = 0 (which happens automatically if D > 1 and not all y_i = 0)
    # 2. The four points satisfy x_i^2 + D*y_i^2 + a*x_i + c = 0 for some a, c.
    #
    # This means the four values (x_i^2 + D*y_i^2) must be affine in x_i:
    # x_i^2 + D*y_i^2 = -a*x_i - c
    # So the four points (x_i, x_i^2 + D*y_i^2) must be collinear in the (x, w) plane
    # where w = x^2 + D*y^2.
    #
    # Equivalently, for any four points, they are concyclic iff the four points
    # (x_i, x_i^2 + D*y_i^2) are collinear (or three of them are, and the fourth agrees).
    
    # Check if four points (x_i, w_i) are collinear where w_i = x_i^2 + D*y_i^2
    x1, y1 = p[0], p[1]
    x2, y2 = q[0], q[1]
    x3, y3 = r[0], r[1]
    x4, y4 = s[0], s[1]
    
    w1 = x1 * x1 + D * y1 * y1
    w2 = x2 * x2 + D * y2 * y2
    w3 = x3 * x3 + D * y3 * y3
    w4 = x4 * x4 + D * y4 * y4
    
    # Check collinearity of (x1,w1), (x2,w2), (x3,w3), (x4,w4)
    # Three points (x1,w1), (x2,w2), (x3,w3) are collinear iff:
    # (x2-x1)*(w3-w1) - (x3-x1)*(w2-w1) = 0
    # And the fourth is on the same line iff (x2-x1)*(w4-w1) - (x4-x1)*(w2-w1) = 0
    
    # Handle case where x coordinates might be equal
    # Use the determinant form
    def _three_collinear_xw(xa, wa, xb, wb, xc, wc):
        return (xb - xa) * (wc - wa) - (xc - xa) * (wb - wa) == 0
    
    # If all x are the same, check if all w are the same (degenerate)
    if x1 == x2 == x3 == x4:
        return w1 == w2 == w3 == w4
    
    # Check if the four (x, w) points are collinear
    # Pick three non-degenerate points and check the fourth
    # Try all combinations of 3 to find a line, then check the 4th
    pts_xw = [(x1, w1), (x2, w2), (x3, w3), (x4, w4)]
    
    for i in range(4):
        for j in range(i+1, 4):
            for k in range(j+1, 4):
                if _three_collinear_xw(*pts_xw[i], *pts_xw[j], *pts_xw[k]):
                    # Check if the 4th point is on this line
                    m = [l for l in range(4) if l not in (i,j,k)][0]
                    if _three_collinear_xw(*pts_xw[i], *pts_xw[j], *pts_xw[m]):
                        return True
    return False


def _valid_add(pt, pts, D):
    """Check if pt can be added to pts maintaining all constraints."""
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
    """Build a set using lattice points (y=0 or small) with D=1."""
    rng = random.Random(seed)
    N = 100
    pool = set()
    for _ in range(5000):
        x = rng.randint(-N, N)
        y = rng.randint(-N, N)
        pool.add((Fraction(x), Fraction(y)))
    
    # Start with a known good 4-point configuration
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
    """Try to build from Heronian triangle configurations."""
    rng = random.Random(seed)
    D = 1
    
    # Known 5-point integral point set
    base_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (5, 0), (0, 5), (3, 4), (4, 3)],
    ]
    
    base = rng.choice(base_sets)
    pts = [(Fraction(x), Fraction(y)) for x, y in base]
    
    # Try to extend by adding points
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
    """Independent exact verification."""
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
    # Check D is squarefree
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
    
    # Try multiple strategies
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
