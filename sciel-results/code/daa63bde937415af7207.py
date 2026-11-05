"""Search-based construction of integral point sets in general position."""
import time
import random
import math
from math import isqrt
from fractions import Fraction

# EVOLVE-BLOCK-START

def _is_squarefree(n):
    if n <= 1:
        return True
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True

def _g2(p1, p2, D=1):
    """Return squared distance as Fraction between two points (x1,y1), (x2,y2) in Q(sqrt(D))."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + D * dy * dy

def _is_rational_dist(p1, p2, D=1):
    """Check if distance between p1 and p2 is rational (exact)."""
    d2 = _g2(p1, p2, D)
    if d2 < 0:
        return False
    num = d2.numerator
    den = d2.denominator
    if num < 0:
        return False
    # Quick check: if numerator or denominator is 0, distance is 0 (invalid for distinct points)
    if num == 0 or den == 0:
        return False
    rn = isqrt(num)
    rd = isqrt(den)
    return rn * rn == num and rd * rd == den

def _no_three_collinear(pts):
    """Check no three points are collinear using exact rational arithmetic."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(pts):
    """Check no four points are concyclic using exact rational arithmetic."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = pts[i], pts[j], pts[k], pts[l]
                    rows = []
                    for p in (p1, p2, p3, p4):
                        x, y = p
                        rows.append([x, y, x * x + y * y, Fraction(1)])
                    a, b, c, d = rows[0]
                    e, f, g, h = rows[1]
                    i_, j_, k_, l_ = rows[2]
                    m, n_, o, p_ = rows[3]
                    det = (a * (f * (k_ * p_ - l_ * o) - g * (j_ * p_ - l_ * n_) + h * (j_ * o - k_ * n_))
                           - b * (e * (k_ * p_ - l_ * o) - g * (i_ * p_ - l_ * m) + h * (i_ * o - k_ * m))
                           + c * (e * (j_ * p_ - l_ * n_) - f * (i_ * p_ - l_ * m) + h * (i_ * n_ - j_ * m))
                           - d * (e * (j_ * o - k_ * n_) - f * (i_ * o - k_ * m) + g * (i_ * n_ - j_ * m)))
                    if det == 0:
                        return False
    return True

def _validate(points_xy, D=1):
    """Validate a set of points: rational distances, no 3 collinear, no 4 concyclic."""
    if len(points_xy) < 3:
        return False
    # Check distinctness
    for i in range(len(points_xy)):
        for j in range(i + 1, len(points_xy)):
            if points_xy[i][0] == points_xy[j][0] and points_xy[i][1] == points_xy[j][1]:
                return False
    # Check all pairwise distances are rational
    for i in range(len(points_xy)):
        for j in range(i + 1, len(points_xy)):
            if not _is_rational_dist(points_xy[i], points_xy[j], D):
                return False
    # Check no three collinear
    if not _no_three_collinear(points_xy):
        return False
    # Check no four concyclic
    if not _no_four_concyclic(points_xy):
        return False
    return True

def _validate_partial(new_pt, existing_pts, D=1):
    """Check if new_pt can be added to existing_pts without violating constraints."""
    # Check distinctness
    for p in existing_pts:
        if new_pt[0] == p[0] and new_pt[1] == p[1]:
            return False
    # Check rational distance to all existing points
    for p in existing_pts:
        if not _is_rational_dist(new_pt, p, D):
            return False
    # Check no three collinear with any pair of existing points
    n = len(existing_pts)
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = existing_pts[i], existing_pts[j]
            cross = (p2[0] - p1[0]) * (new_pt[1] - p1[1]) - (p2[1] - p1[1]) * (new_pt[0] - p1[0])
            if cross == 0:
                return False
    # Check no four concyclic with any triple of existing points
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = existing_pts[i], existing_pts[j], existing_pts[k]
                rows = [
                    [p1[0], p1[1], p1[0] * p1[0] + p1[1] * p1[1], Fraction(1)],
                    [p2[0], p2[1], p2[0] * p2[0] + p2[1] * p2[1], Fraction(1)],
                    [p3[0], p3[1], p3[0] * p3[0] + p3[1] * p3[1], Fraction(1)],
                    [new_pt[0], new_pt[1], new_pt[0] * new_pt[0] + new_pt[1] * new_pt[1], Fraction(1)]
                ]
                a, b, c, d = rows[0]
                e, f, g, h = rows[1]
                i_, j_, k_, l_ = rows[2]
                m, n_, o, p_ = rows[3]
                det = (a * (f * (k_ * p_ - l_ * o) - g * (j_ * p_ - l_ * n_) + h * (j_ * o - k_ * n_))
                       - b * (e * (k_ * p_ - l_ * o) - g * (i_ * p_ - l_ * m) + h * (i_ * o - k_ * m))
                       + c * (e * (j_ * p_ - l_ * n_) - f * (i_ * p_ - l_ * m) + h * (i_ * n_ - j_ * m))
                       - d * (e * (j_ * o - k_ * n_) - f * (i_ * o - k_ * m) + g * (i_ * n_ - j_ * m)))
                if det == 0:
                    return False
    return True

def _generate_rational_point(rng, max_den=100):
    """Generate a random rational point with small numerator/denominator."""
    x_num = rng.randint(-max_den, max_den)
    x_den = rng.randint(1, max_den)
    y_num = rng.randint(-max_den, max_den)
    y_den = rng.randint(1, max_den)
    x = Fraction(x_num, x_den)
    y = Fraction(y_num, y_den)
    return (x, y)

def _pythagorean_triples(max_val=1000):
    """Generate Pythagorean triples (a, b, c) with a^2 + b^2 = c^2 up to max_val."""
    triples = set()
    for m in range(2, int(math.sqrt(max_val)) + 2):
        for n in range(1, m):
            if (m - n) % 2 == 1 and math.gcd(m, n) == 1:
                a = m * m - n * n
                b = 2 * m * n
                c = m * m + n * n
                if a <= max_val and b <= max_val and c <= max_val:
                    triples.add((a, b, c))
                    triples.add((b, a, c))
    return list(triples)

def _try_extend(pts, D, deadline, max_attempts=50000, rng=None):
    """Try to extend pts with one more valid point. Return extended list or None."""
    if rng is None:
        rng = random
    
    # Precompute distance constraints from existing points
    # For a new point (x, y), distance to point (xi, yi) squared is:
    # (x-xi)^2 + D*(y-yi)^2 = rational square
    # This is a quadratic constraint
    
    for attempt in range(max_attempts):
        if time.time() >= deadline:
            break
        
        # Generate candidate point using multiple strategies
        strategy = attempt % 5
        
        if strategy == 0:
            # Small integer coordinates
            x = Fraction(rng.randint(-50, 50), 1)
            y = Fraction(rng.randint(-50, 50), 1)
        elif strategy == 1:
            # Small rational coordinates
            x = Fraction(rng.randint(-50, 50), rng.randint(1, 20))
            y = Fraction(rng.randint(-50, 50), rng.randint(1, 20))
        elif strategy == 2:
            # Wider range
            x = Fraction(rng.randint(-200, 200), rng.randint(1, 50))
            y = Fraction(rng.randint(-200, 200), rng.randint(1, 50))
        elif strategy == 3:
            # Use Pythagorean triple-inspired coordinates
            # Try to make distance to first point a Pythagorean triple
            if len(pts) > 0:
                x0, y0 = pts[0]
                # Generate a Pythagorean triple
                m = rng.randint(2, 20)
                n = rng.randint(1, m - 1)
                if (m - n) % 2 == 1 and math.gcd(m, n) == 1:
                    a = m * m - n * n
                    b = 2 * m * n
                    # Try different assignments
                    if rng.random() < 0.5:
                        x = x0 + Fraction(a, 1)
                        y = y0 + Fraction(b, 1)
                    else:
                        x = x0 + Fraction(b, 1)
                        y = y0 + Fraction(a, 1)
                    if rng.random() < 0.5:
                        x = -x
                        y = -y
                else:
                    x = Fraction(rng.randint(-100, 100), 1)
                    y = Fraction(rng.randint(-100, 100), 1)
            else:
                x = Fraction(rng.randint(-100, 100), 1)
                y = Fraction(rng.randint(-100, 100), 1)
        else:
            # Rational with small denominators, biased toward integers
            x_den = rng.choice([1, 1, 1, 2, 3, 4, 5])
            y_den = rng.choice([1, 1, 1, 2, 3, 4, 5])
            x = Fraction(rng.randint(-100, 100), x_den)
            y = Fraction(rng.randint(-100, 100), y_den)
        
        new_pt = (x, y)
        
        # Quick pre-checks before full validation
        # Check distinctness
        is_distinct = True
        for p in pts:
            if new_pt[0] == p[0] and new_pt[1] == p[1]:
                is_distinct = False
                break
        if not is_distinct:
            continue
        
        # Check rational distance to all existing points (fail fast)
        all_rational = True
        for p in pts:
            if not _is_rational_dist(new_pt, p, D):
                all_rational = False
                break
        if not all_rational:
            continue
        
        # Check no three collinear
        n = len(pts)
        collinear = False
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = pts[i], pts[j]
                cross = (p2[0] - p1[0]) * (new_pt[1] - p1[1]) - (p2[1] - p1[1]) * (new_pt[0] - p1[0])
                if cross == 0:
                    collinear = True
                    break
            if collinear:
                break
        if collinear:
            continue
        
        # Check no four concyclic with any triple
        concyclic = False
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    rows = [
                        [p1[0], p1[1], p1[0] * p1[0] + p1[1] * p1[1], Fraction(1)],
                        [p2[0], p2[1], p2[0] * p2[0] + p2[1] * p2[1], Fraction(1)],
                        [p3[0], p3[1], p3[0] * p3[0] + p3[1] * p3[1], Fraction(1)],
                        [new_pt[0], new_pt[1], new_pt[0] * new_pt[0] + new_pt[1] * new_pt[1], Fraction(1)]
                    ]
                    a, b, c, d = rows[0]
                    e, f, g, h = rows[1]
                    i_, j_, k_, l_ = rows[2]
                    m, n_, o, p_ = rows[3]
                    det = (a * (f * (k_ * p_ - l_ * o) - g * (j_ * p_ - l_ * n_) + h * (j_ * o - k_ * n_))
                           - b * (e * (k_ * p_ - l_ * o) - g * (i_ * p_ - l_ * m) + h * (i_ * o - k_ * m))
                           + c * (e * (j_ * p_ - l_ * n_) - f * (i_ * p_ - l_ * m) + h * (i_ * n_ - j_ * m))
                           - d * (e * (j_ * o - k_ * n_) - f * (i_ * o - k_ * m) + g * (i_ * n_ - j_ * m)))
                    if det == 0:
                        concyclic = True
                        break
                if concyclic:
                    break
            if concyclic:
                break
        if concyclic:
            continue
        
        # Full validation as final check
        candidate = pts + [new_pt]
        if _validate(candidate, D):
            return candidate
    
    return None

def construct():
    """Construct a valid integral point set, maximizing the number of points."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_points = None
    best_D = 1
    best_count = 0
    
    random.seed(42)
    
    # Known valid 5-point solution (D=1)
    known_points = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
                    (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
                    (Fraction(16), Fraction(63))]
    if _validate(known_points, 1):
        best_points = known_points
        best_D = 1
        best_count = len(known_points)
    
    # Strategy 1: Try to extend the known 5-point set with more attempts
    if time.time() < deadline and best_count < 6:
        extended = _try_extend(known_points, 1, deadline, max_attempts=200000)
        if extended is not None:
            best_points = extended
            best_D = 1
            best_count = len(extended)
            # Try to extend further
            if time.time() < deadline and best_count < 7:
                extended2 = _try_extend(extended, 1, deadline, max_attempts=100000)
                if extended2 is not None:
                    best_points = extended2
                    best_count = len(extended2)
    
    # Strategy 2: Systematic search for 6-point sets with D=1 using Pythagorean-inspired configs
    if time.time() < deadline and best_count < 6:
        # Try different base configurations with Pythagorean triple distances
        base_configs = [
            # (3,4,5) triangle based
            [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(0), Fraction(4)),
             (Fraction(-3), Fraction(0)), (Fraction(0), Fraction(-4))],
            # (5,12,13) triangle based
            [(Fraction(0), Fraction(0)), (Fraction(5), Fraction(0)), (Fraction(0), Fraction(12)),
             (Fraction(-5), Fraction(0)), (Fraction(0), Fraction(-12))],
            # (7,24,25) triangle based
            [(Fraction(0), Fraction(0)), (Fraction(7), Fraction(0)), (Fraction(0), Fraction(24)),
             (Fraction(-7), Fraction(0)), (Fraction(0), Fraction(-24))],
            # (8,15,17) triangle based
            [(Fraction(0), Fraction(0)), (Fraction(8), Fraction(0)), (Fraction(0), Fraction(15)),
             (Fraction(-8), Fraction(0)), (Fraction(0), Fraction(-15))],
            # (9,40,41) triangle based
            [(Fraction(0), Fraction(0)), (Fraction(9), Fraction(0)), (Fraction(0), Fraction(40)),
             (Fraction(-9), Fraction(0)), (Fraction(0), Fraction(-40))],
            # (11,60,61) triangle based
            [(Fraction(0), Fraction(0)), (Fraction(11), Fraction(0)), (Fraction(0), Fraction(60)),
             (Fraction(-11), Fraction(0)), (Fraction(0), Fraction(-60))],
            # Asymmetric configurations
            [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(4), Fraction(0)),
             (Fraction(0), Fraction(5)), (Fraction(0), Fraction(-5))],
            [(Fraction(0), Fraction(0)), (Fraction(5), Fraction(0)), (Fraction(12), Fraction(0)),
             (Fraction(0), Fraction(13)), (Fraction(0), Fraction(-13))],
        ]
        
        for base in base_configs:
            if time.time() >= deadline:
                break
            if _validate(base, 1):
                extended = _try_extend(base, 1, deadline, max_attempts=30000)
                if extended is not None:
                    best_points = extended
                    best_D = 1
                    best_count = len(extended)
                    if best_count >= 6:
                        # Try to extend to 7
                        if time.time() < deadline and best_count < 7:
                            extended2 = _try_extend(extended, 1, deadline, max_attempts=50000)
                            if extended2 is not None:
                                best_points = extended2
                                best_count = len(extended2)
                        break
    
    # Strategy 3: Random search with D=1 for 6+ points, using incremental construction
    if time.time() < deadline and best_count < 6:
        for attempt in range(100000):
            if time.time() >= deadline:
                break
            points = []
            valid = True
            for _ in range(6):
                # Use Pythagorean-inspired random points
                if len(points) == 0:
                    x = Fraction(random.randint(-20, 20), 1)
                    y = Fraction(random.randint(-20, 20), 1)
                else:
                    # Try to maintain rational distances to existing points
                    strategy = random.randint(0, 2)
                    if strategy == 0:
                        x = Fraction(random.randint(-30, 30), 1)
                        y = Fraction(random.randint(-30, 30), 1)
                    elif strategy == 1:
                        x = Fraction(random.randint(-30, 30), random.randint(1, 10))
                        y = Fraction(random.randint(-30, 30), random.randint(1, 10))
                    else:
                        # Use Pythagorean triple offset from first point
                        if points:
                            x0, y0 = points[0]
                            m = random.randint(2, 15)
                            n = random.randint(1, m - 1)
                            if (m - n) % 2 == 1 and math.gcd(m, n) == 1:
                                a = m * m - n * n
                                b = 2 * m * n
                                if random.random() < 0.5:
                                    x = x0 + Fraction(a, 1)
                                    y = y0 + Fraction(b, 1)
                                else:
                                    x = x0 + Fraction(b, 1)
                                    y = y0 + Fraction(a, 1)
                            else:
                                x = Fraction(random.randint(-30, 30), 1)
                                y = Fraction(random.randint(-30, 30), 1)
                        else:
                            x = Fraction(random.randint(-30, 30), 1)
                            y = Fraction(random.randint(-30, 30), 1)
                new_pt = (x, y)
                if not _validate_partial(new_pt, points, 1):
                    valid = False
                    break
                points.append(new_pt)
            if valid and len(points) == 6:
                if _validate(points, 1):
                    best_points = points
                    best_D = 1
                    best_count = len(points)
                    # Try to extend to 7
                    if time.time() < deadline:
                        extended = _try_extend(points, 1, deadline, max_attempts=20000)
                        if extended is not None:
                            best_points = extended
                            best_count = len(extended)
                    break
    
    # Strategy 4: Try other D values for 6+ points
    if time.time() < deadline and best_count < 6:
        D_candidates = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33]
        for D in D_candidates:
            if time.time() >= deadline:
                break
            for attempt in range(10000):
                if time.time() >= deadline:
                    break
                points = []
                valid = True
                for _ in range(6):
                    x = Fraction(random.randint(-20, 20), 1)
                    y = Fraction(random.randint(-20, 20), 1)
                    new_pt = (x, y)
                    if not _validate_partial(new_pt, points, D):
                        valid = False
                        break
                    points.append(new_pt)
                if valid and len(points) == 6:
                    if _validate(points, D):
                        best_points = points
                        best_D = D
                        best_count = len(points)
                        break
    
    # Strategy 5: If still at 5 points, try more aggressive random search
    if time.time() < deadline and best_count < 6:
        for attempt in range(200000):
            if time.time() >= deadline:
                break
            points = []
            valid = True
            for _ in range(6):
                x = Fraction(random.randint(-100, 100), random.randint(1, 20))
                y = Fraction(random.randint(-100, 100), random.randint(1, 20))
                new_pt = (x, y)
                if not _validate_partial(new_pt, points, 1):
                    valid = False
                    break
                points.append(new_pt)
            if valid and len(points) == 6:
                if _validate(points, 1):
                    best_points = points
                    best_D = 1
                    best_count = len(points)
                    break
    
    # Final validation
    if best_points is None:
        # Fallback to known valid set
        best_points = known_points
        best_D = 1
    
    # Format output
    result_points = []
    for x, y in best_points:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': best_D, 'points': result_points}

# EVOLVE-BLOCK-END
