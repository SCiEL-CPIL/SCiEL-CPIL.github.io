"""Construct integral point sets (no three collinear, no four concyclic)."""
from fractions import Fraction
import random
import time
import math
from itertools import combinations

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


def _is_perfect_square_frac(f):
    if f < 0:
        return False
    n, d = f.numerator, f.denominator
    if n == 0:
        return True
    sn = math.isqrt(n)
    sd = math.isqrt(d)
    return sn * sn == n and sd * sd == d


def _dist_sq_D(a, b, D):
    """Compute squared distance in the field Q(sqrt(D)).
    Points are (x, y) meaning (x, y*sqrt(D)) in the plane.
    Distance squared = (x1-x2)^2 + D*(y1-y2)^2
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + D * dy * dy


def _distance_rational(a, b, D):
    dsq = _dist_sq_D(a, b, D)
    return _is_perfect_square_frac(dsq)


def _collinear(a, b, c):
    """Check if three points (x, y) are collinear using cross product."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]) == 0


def _concyclic4(p, q, r, s, D):
    """Check if four points are concyclic in the metric with D.
    
    In the field Q(sqrt(D)), a circle equation is:
    x^2 + D*y^2 + Ax + By + C = 0
    
    We transform each point (x, y) to (x, w) where w = x^2 + D*y^2.
    Four points are concyclic iff their (x, w) projections are collinear
    in a specific sense - actually we need the circle equation to be consistent.
    
    A point (x_i, y_i) lies on circle x^2 + Dy^2 + Ax + By + C = 0 iff:
    x_i^2 + D*y_i^2 + A*x_i + B*y_i + C = 0
    
    So we have a system: for 3 points, solve for A, B, C. Then check 4th.
    Using exact arithmetic.
    """
    x1, y1 = p[0], p[1]
    x2, y2 = q[0], q[1]
    x3, y3 = r[0], r[1]
    x4, y4 = s[0], s[1]
    
    # w_i = x_i^2 + D*y_i^2
    w1 = x1 * x1 + D * y1 * y1
    w2 = x2 * x2 + D * y2 * y2
    w3 = x3 * x3 + D * y3 * y3
    w4 = x4 * x4 + D * y4 * y4
    
    # Circle: x^2 + Dy^2 + Ax + By + C = 0
    # => A*x_i + B*y_i + C = -(x_i^2 + D*y_i^2) = -w_i
    # So: A*x_i + B*y_i + C + w_i = 0
    
    # We need to check if there exist A, B, C such that:
    # A*x_i + B*y_i + C = -w_i for i=1,2,3,4
    
    # This is a 4x3 system. The points are concyclic iff the system is consistent.
    # Using determinants: the 4 points are concyclic iff the 4x4 determinant
    # |x_i  y_i  1  w_i| = 0 (where w_i = x_i^2 + D*y_i^2)
    
    # det = |x1 y1 1 w1|
    #       |x2 y2 1 w2|
    #       |x3 y3 1 w3|
    #       |x4 y4 1 w4|
    
    # Expand along first row:
    # x1*(y2*(1*w4 - w3*1) - 1*(y2*w4 - w2*y4) + w1*(y2*w3 - w2*y3))
    # - y1*(x2*(1*w4 - w3*1) - 1*(x2*w4 - w2*x4) + w1*(x2*w3 - w2*x3))
    # + 1*(x2*(y2*w4 - w2*y4) - y2*(x2*w4 - w2*x4) + w2*(x2*y4 - y2*x4))  -- wait let me be more careful
    
    # Let me compute the 4x4 determinant properly.
    # M = [[x1,y1,1,w1],[x2,y2,1,w2],[x3,y3,1,w3],[x4,y4,1,w4]]
    
    # Using cofactor expansion along row 1:
    # det = x1*M11 - y1*M12 + 1*M13 - w1*M14
    # where M1j is the (1,j) minor with sign (-1)^(1+j)
    
    # M11 = det[[y2,1,w2],[y3,1,w3],[y4,1,w4]]
    #      = y2*(1*w4 - w3*1) - 1*(y2*w4 - w2*y4) + w2*(y2*1 - 1*y4)
    #      = y2*(w4 - w3) - (y2*w4 - w2*y4) + w2*(y2 - y4)
    #      = y2*w4 - y2*w3 - y2*w4 + w2*y4 + w2*y2 - w2*y4
    #      = -y2*w3 + w2*y2
    #      = y2*(w2 - w3)  -- wait that's not right, let me redo
    
    # M11 = y2*(1*w4 - w3*1) - 1*(y2*w4 - w2*y4) + w2*(y2*1 - 1*y4)
    #      = y2*(w4 - w3) - (y2*w4 - w2*y4) + w2*(y2 - y4)
    #      = y2*w4 - y2*w3 - y2*w4 + w2*y4 + w2*y2 - w2*y4
    #      = -y2*w3 + w2*y2
    #      = y2*(w2 - w3)  -- hmm, that seems too simple. Let me verify with numbers.
    
    # Actually let me just compute it directly with the formula:
    # 3x3 det = a(ei-fh) - b(di-fg) + c(dh-eg)
    # For [[a,b,c],[d,e,f],[g,h,i]]:
    
    def det3(a,b,c,d,e,f,g,h,i):
        return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    
    # M11 (remove row 1, col 1): [[y2,1,w2],[y3,1,w3],[y4,1,w4]]
    M11 = det3(y2, 1, w2, y3, 1, w3, y4, 1, w4)
    # M12 (remove row 1, col 2): [[x2,1,w2],[x3,1,w3],[x4,1,w4]]
    M12 = det3(x2, 1, w2, x3, 1, w3, x4, 1, w4)
    # M13 (remove row 1, col 3): [[x2,y2,w2],[x3,y3,w3],[x4,y4,w4]]
    M13 = det3(x2, y2, w2, x3, y3, w3, x4, y4, w4)
    # M14 (remove row 1, col 4): [[x2,y2,1],[x3,y3,1],[x4,y4,1]]
    M14 = det3(x2, y2, 1, x3, y3, 1, x4, y4, 1)
    
    det = x1*M11 - y1*M12 + 1*M13 - w1*M14
    
    return det == 0


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


def _generate_pythagorean_points(max_coord):
    """Generate points with rational coordinates that form Pythagorean triples
    with the origin, giving us a good candidate pool."""
    points = set()
    # Add origin
    points.add((Fraction(0), Fraction(0)))
    
    # Generate from Pythagorean triples (a, b, c) where a^2 + b^2 = c^2
    # Point (a, b) has distance c from origin
    # Also (a, -b), (-a, b), (-a, -b)
    for m in range(1, 200):
        for n in range(1, m):
            if (m - n) % 2 == 0 or math.gcd(m, n) != 1:
                continue
            a = m*m - n*n
            b = 2*m*n
            c = m*m + n*n
            if a > max_coord or b > max_coord:
                continue
            for sa in [1, -1]:
                for sb in [1, -1]:
                    points.add((Fraction(sa*a), Fraction(sb*b)))
                    # Also scaled versions
                    for scale in range(1, 5):
                        if sa*a*scale <= max_coord and sb*b*scale <= max_coord:
                            points.add((Fraction(sa*a*scale), Fraction(sb*b*scale)))
    
    return list(points)


def _generate_rational_circle_points(D, max_coord):
    """Generate points on circles centered at origin in Q(sqrt(D)).
    Points (x, y) where x^2 + D*y^2 is a perfect square."""
    points = set()
    points.add((Fraction(0), Fraction(0)))
    
    for x_num in range(-max_coord, max_coord + 1):
        for y_num in range(-max_coord, max_coord + 1):
            x = Fraction(x_num)
            y = Fraction(y_num)
            dsq = x*x + D*y*y
            if _is_perfect_square_frac(dsq):
                points.add((x, y))
    
    return list(points)


def _try_build_backtracking(seed, D, pool, max_points=12):
    """Backtracking search to build a valid set."""
    rng = random.Random(seed)
    
    # Start with a few points that are known to work
    # Try different starting configurations
    starts = []
    
    # Try to find good starting quadruples from the pool
    pool_sample = pool[:min(len(pool), 200)]
    rng.shuffle(pool_sample)
    
    best_pts = None
    best_len = 0
    
    # Try multiple starting points
    for start_idx in range(min(50, len(pool_sample))):
        if _time_left() < 5:
            break
        start_pt = pool_sample[start_idx]
        pts = [start_pt]
        
        # Greedily add points
        for cand in pool_sample:
            if _time_left() < 5:
                break
            if cand == start_pt:
                continue
            if any(cand == p for p in pts):
                continue
            if _valid_add(cand, pts, D):
                pts.append(cand)
                if len(pts) >= max_points:
                    break
        
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = list(pts)
    
    return best_pts if best_pts else []


def _try_build_lattice(seed, D=1):
    """Build using lattice points with rational distances."""
    rng = random.Random(seed)
    
    # Generate pool of points with rational distances from each other
    # Use Pythagorean triple based points
    max_coord = 200
    pool = _generate_pythagorean_points(max_coord)
    
    # Also add some random lattice points that happen to have rational distances
    N = 150
    for _ in range(3000):
        x = rng.randint(-N, N)
        y = rng.randint(-N, N)
        pt = (Fraction(x), Fraction(y))
        pool.append(pt)
    
    # Deduplicate
    pool = list(set(pool))
    rng.shuffle(pool)
    
    return _try_build_backtracking(seed, D, pool, max_points=12)


def _try_build_heronian(seed):
    """Build using known Heronian configurations."""
    rng = random.Random(seed)
    D = 1
    
    # Known good starting sets
    base_sets = [
        [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)), (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)), (Fraction(16), Fraction(63))],
        [(Fraction(0), Fraction(0)), (Fraction(5), Fraction(0)), (Fraction(0), Fraction(5)), (Fraction(3), Fraction(4)), (Fraction(4), Fraction(3))],
        [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)), (Fraction(4), Fraction(3)), (Fraction(5), Fraction(0)), (Fraction(0), Fraction(5))],
    ]
    
    best_pts = None
    best_len = 0
    
    for base in base_sets:
        if _time_left() < 5:
            break
        pts = list(base)
        
        # Generate candidate pool
        N = 200
        pool = []
        for _ in range(5000):
            x = rng.randint(-N, N)
            y = rng.randint(-N, N)
            pool.append((Fraction(x), Fraction(y)))
        
        rng.shuffle(pool)
        
        # Try to extend
        for cand in pool:
            if _time_left() < 5:
                break
            if any(cand == p for p in pts):
                continue
            if _valid_add(cand, pts, D):
                pts.append(cand)
                if len(pts) >= 12:
                    break
        
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = list(pts)
    
    return best_pts if best_pts else []


def _finalize(pts, D=1):
    """Convert internal representation to output format."""
    return {'D': D, 'points': [[_fmt(p[0].numerator, p[0].denominator),
                                _fmt(p[1].numerator, p[1].denominator)] for p in pts]}


def _verify(res):
    """Full verification of the result."""
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
    # Check squarefree
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
            if pts is None or len(pts) < 3:
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
        # Fallback: known valid 5-point set
        points = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)), 
                  (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)), 
                  (Fraction(16), Fraction(63))]
        best = _finalize(points, D=1)
    return best


def construct():
    return _run()
