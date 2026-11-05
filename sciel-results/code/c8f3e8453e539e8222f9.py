import math
from fractions import Fraction
import time
import itertools
import random

# EVOLVE-BLOCK-START

def _is_square_frac(f):
    """Check if Fraction f is a perfect square of a rational."""
    if f < 0:
        return False
    n, d = f.numerator, f.denominator
    if n < 0:
        return False
    rn = math.isqrt(n)
    rd = math.isqrt(d)
    return rn * rn == n and rd * rd == d

def _sqrt_frac(f):
    """Compute sqrt of a Fraction that is known to be a perfect square."""
    n, d = f.numerator, f.denominator
    rn = math.isqrt(n)
    rd = math.isqrt(d)
    return Fraction(rn, rd)

def _dist_sq(p1, p2, D):
    """Squared distance between two points (x,y) in the (x, y*sqrt(D)) metric."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + Fraction(D) * dy * dy

def _is_rational_dist(p1, p2, D):
    d2 = _dist_sq(p1, p2, D)
    return _is_square_frac(d2)

def _no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if cross == 0:
                    return False
    return True

def _det4(M):
    """Compute 4x4 determinant."""
    a = M[0]; b = M[1]; c = M[2]; d = M[3]
    return (a[0]*(b[1]*(c[2]*d[3]-c[3]*d[2]) - b[2]*(c[1]*d[3]-c[3]*d[1]) + b[3]*(c[1]*d[2]-c[2]*d[1]))
         - a[1]*(b[0]*(c[2]*d[3]-c[3]*d[2]) - b[2]*(c[0]*d[3]-c[3]*d[0]) + b[3]*(c[0]*d[2]-c[2]*d[0]))
         + a[2]*(b[0]*(c[1]*d[3]-c[3]*d[1]) - b[1]*(c[0]*d[3]-c[3]*d[0]) + b[3]*(c[0]*d[1]-c[1]*d[0]))
         - a[3]*(b[0]*(c[1]*d[2]-c[2]*d[1]) - b[1]*(c[0]*d[2]-c[2]*d[0]) + b[2]*(c[0]*d[1]-c[1]*d[0])))

def _no_four_concyclic(points):
    n = len(points)
    one = Fraction(1)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    M = [
                        [p1[0], p1[1], p1[0]**2 + p1[1]**2, one],
                        [p2[0], p2[1], p2[0]**2 + p2[1]**2, one],
                        [p3[0], p3[1], p3[0]**2 + p3[1]**2, one],
                        [p4[0], p4[1], p4[0]**2 + p4[1]**2, one],
                    ]
                    if _det4(M) == 0:
                        return False
    return True

def _no_four_concyclic_with_new(P, Q):
    """Check that no four points among P + [Q] are concyclic, assuming P has no 4 concyclic."""
    n = len(P)
    q0, q1 = Q[0], Q[1]
    q2 = q0 * q0 + q1 * q1
    one = Fraction(1)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = P[i], P[j], P[k]
                M = [
                    [p1[0], p1[1], p1[0]**2 + p1[1]**2, one],
                    [p2[0], p2[1], p2[0]**2 + p2[1]**2, one],
                    [p3[0], p3[1], p3[0]**2 + p3[1]**2, one],
                    [q0, q1, q2, one],
                ]
                if _det4(M) == 0:
                    return False
    return True

def _no_three_collinear_with_new(P, Q):
    """Check that no three points among P + [Q] are collinear, assuming P has no 3 collinear."""
    n = len(P)
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2, p3 = P[i], P[j], Q
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            if cross == 0:
                return False
    return True

def _validate(points, D):
    if len(points) < 3:
        return False
    if len(set(points)) != len(points):
        return False
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if not _is_rational_dist(points[i], points[j], D):
                return False
    if not _no_three_collinear(points):
        return False
    if not _no_four_concyclic(points):
        return False
    return True

def _circle_intersection(p1, p2, r1_sq, r2_sq):
    """
    Find intersection points of two circles centered at p1, p2 with squared radii r1_sq, r2_sq.
    Returns list of intersection points (x, y) as Fractions, or empty list.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    d2 = dx * dx + dy * dy
    
    if d2 == 0:
        return []
    
    if not _is_square_frac(d2):
        return []
    d = _sqrt_frac(d2)
    
    # Check if circles intersect
    r1_sq_f = Fraction(r1_sq) if not isinstance(r1_sq, Fraction) else r1_sq
    r2_sq_f = Fraction(r2_sq) if not isinstance(r2_sq, Fraction) else r2_sq
    
    # Check triangle inequalities
    r1 = _sqrt_frac(r1_sq_f) if _is_square_frac(r1_sq_f) else None
    r2 = _sqrt_frac(r2_sq_f) if _is_square_frac(r2_sq_f) else None
    if r1 is None or r2 is None:
        return []
    
    if r1 + r2 < d or r1 + d < r2 or r2 + d < r1:
        return []
    if r1 + r2 == d or r1 + d == r2 or r2 + d == r1:
        return []
    
    # Distance from p1 to the line connecting intersection points
    x_offset = (r1_sq_f - r2_sq_f + d2) / (2 * d)
    x = x1 + x_offset
    
    y_sq = r1_sq_f - x_offset * x_offset
    if y_sq < 0:
        return []
    if not _is_square_frac(y_sq):
        return []
    y_offset = _sqrt_frac(y_sq)
    
    candidates = [(x, y1 + y_offset), (x, y1 - y_offset)]
    
    # Remove duplicates
    seen = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    
    return result

def _try_extend(points, D, deadline, max_iterations=500000):
    """Try to add one more point to the given set using multiple strategies."""
    start = time.time()
    P = list(points)
    n = len(P)
    
    # Precompute all pairwise distances (squared) for reference
    pair_dists = {}
    pair_dists_sqrt = {}
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _dist_sq(P[i], P[j], D)
            pair_dists[(i, j)] = d2
            if _is_square_frac(d2):
                pair_dists_sqrt[(i, j)] = _sqrt_frac(d2)
    
    # Strategy 1: Circle-circle intersection with rational radii
    # Generate candidate squared radii
    radius_candidates = []
    for num in range(1, 100):
        for den in range(1, 20):
            if math.gcd(num, den) == 1:
                r = Fraction(num, den)
                r2 = r * r
                radius_candidates.append(r2)
    
    # Sort and limit
    radius_candidates.sort(key=lambda x: float(x))
    if len(radius_candidates) > 500:
        step = len(radius_candidates) // 500
        radius_candidates = radius_candidates[::step]
    
    ref_pairs = list(itertools.combinations(range(n), 2))
    random.shuffle(ref_pairs)
    
    for (i, j) in ref_pairs:
        if time.time() > deadline - 15:
            break
        
        d2_ij = pair_dists[(i, j)]
        if not _is_square_frac(d2_ij):
            continue
        
        for r1_sq in radius_candidates:
            if time.time() > deadline - 15:
                break
            for r2_sq in radius_candidates:
                if time.time() > deadline - 15:
                    break
                
                intersections = _circle_intersection(P[i], P[j], r1_sq, r2_sq)
                
                for Q in intersections:
                    if Q in P:
                        continue
                    
                    # Check rational distance to all existing points
                    valid = True
                    for pk in P:
                        if not _is_rational_dist(Q, pk, D):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Check no three collinear with new point
                    if not _no_three_collinear_with_new(P, Q):
                        continue
                    
                    # Check no four concyclic with new point
                    if not _no_four_concyclic_with_new(P, Q):
                        continue
                    
                    return P + [Q]
    
    # Strategy 2: Try points on lines through pairs of existing points
    if time.time() < deadline - 15:
        for i in range(n):
            if time.time() > deadline - 15:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 15:
                    break
                p1, p2 = P[i], P[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                # Parametric: P(t) = p1 + t * (dx, dy) for rational t
                for t_num in range(-20, 21):
                    if time.time() > deadline - 15:
                        break
                    for t_den in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
                        if time.time() > deadline - 15:
                            break
                        if t_num == 0 and t_den != 1:
                            continue
                        if math.gcd(abs(t_num), t_den) != 1:
                            continue
                        t = Fraction(t_num, t_den)
                        qx = p1[0] + t * dx
                        qy = p1[1] + t * dy
                        Q = (qx, qy)
                        
                        if Q in P:
                            continue
                        
                        # Check rational distance to all existing points
                        valid = True
                        for pk in P:
                            if not _is_rational_dist(Q, pk, D):
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Check no three collinear with new point
                        if not _no_three_collinear_with_new(P, Q):
                            continue
                        
                        # Check no four concyclic with new point
                        if not _no_four_concyclic_with_new(P, Q):
                            continue
                        
                        return P + [Q]
    
    # Strategy 3: Grid search around existing points
    if time.time() < deadline - 10:
        for base_pt in P:
            if time.time() > deadline - 10:
                break
            for dx_num in range(-30, 31):
                if time.time() > deadline - 10:
                    break
                for dx_den in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    if time.time() > deadline - 10:
                        break
                    if dx_num == 0 and dx_den != 1:
                        continue
                    if math.gcd(abs(dx_num), dx_den) != 1:
                        continue
                    for dy_num in range(-30, 31):
                        if time.time() > deadline - 10:
                            break
                        for dy_den in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                            if time.time() > deadline - 10:
                                break
                            if dy_num == 0 and dy_den != 1:
                                continue
                            if math.gcd(abs(dy_num), dy_den) != 1:
                                continue
                            if dx_num == 0 and dy_num == 0:
                                continue
                            
                            qx = base_pt[0] + Fraction(dx_num, dx_den)
                            qy = base_pt[1] + Fraction(dy_num, dy_den)
                            Q = (qx, qy)
                            
                            if Q in P:
                                continue
                            
                            # Check rational distance to all existing points
                            valid = True
                            for pk in P:
                                if not _is_rational_dist(Q, pk, D):
                                    valid = False
                                    break
                            if not valid:
                                continue
                            
                            # Check no three collinear with new point
                            if not _no_three_collinear_with_new(P, Q):
                                continue
                            
                            # Check no four concyclic with new point
                            if not _no_four_concyclic_with_new(P, Q):
                                continue
                            
                            return P + [Q]
    
    return None

def _find_3point_base(D, deadline):
    """Find a 3-point base set for given D."""
    base = [(Fraction(0), Fraction(0))]
    
    for x1_num in range(1, 60):
        if time.time() > deadline - 15:
            break
        for y2_num in range(1, 60):
            if time.time() > deadline - 15:
                break
            p1 = (Fraction(x1_num), Fraction(0))
            p2 = (Fraction(0), Fraction(y2_num))
            
            d2 = Fraction(x1_num)**2 + Fraction(D) * Fraction(y2_num)**2
            if not _is_square_frac(d2):
                continue
            
            candidate_3 = [base[0], p1, p2]
            if _validate(candidate_3, D):
                return candidate_3
    
    return None

def _try_random_extension(points, D, deadline, max_attempts=10000):
    """Try random integer/rational perturbations to find a valid extension."""
    P = list(points)
    n = len(P)
    
    # Compute bounding box of existing points
    xs = [p[0] for p in P]
    ys = [p[1] for p in P]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    
    # Expand the search region
    x_range = max(30, int(abs((x_max - x_min).numerator) / max(1, x_max.denominator)) + 30)
    y_range = max(30, int(abs((y_max - y_min).numerator) / max(1, y_max.denominator)) + 30)
    
    for _ in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        # Random rational point in the expanded region
        qx_num = random.randint(-x_range, x_range)
        qx_den = random.choice([1, 1, 1, 1, 2, 3, 4, 5, 6])
        qy_num = random.randint(-y_range, y_range)
        qy_den = random.choice([1, 1, 1, 1, 2, 3, 4, 5, 6])
        
        Q = (Fraction(qx_num, qx_den), Fraction(qy_num, qy_den))
        
        if Q in P:
            continue
        
        # Check rational distance to all existing points
        valid = True
        for pk in P:
            if not _is_rational_dist(Q, pk, D):
                valid = False
                break
        if not valid:
            continue
        
        # Check no three collinear with new point
        if not _no_three_collinear_with_new(P, Q):
            continue
        
        # Check no four concyclic with new point
        if not _no_four_concyclic_with_new(P, Q):
            continue
        
        return P + [Q]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    best_points = None
    best_D = None
    
    # Known 5-point set with D=1
    known_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    points = [(Fraction(x), Fraction(y)) for x, y in known_5]
    
    if _validate(points, D):
        best_points = points
        best_D = D
    
    # Try to extend to 6, 7, ... points with the known 5-point set
    if best_points is not None and time.time() < deadline - 40:
        current = list(best_points)
        for _ in range(7):  # Try to add up to 7 more points
            if time.time() > deadline - 40:
                break
            extended = _try_extend(current, best_D, deadline)
            if extended:
                current = extended
            else:
                break
        
        if len(current) > len(best_points):
            best_points = current
    
        # Try random extension as well
        if time.time() < deadline - 25:
            random_extended = _try_random_extension(best_points, best_D, deadline, max_attempts=5000)
            if random_extended and len(random_extended) > len(best_points):
                best_points = random_extended
    
    # Try other D values with systematic search
    D_values = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 90, 91, 93, 94, 95, 97]
    
    for D_test in D_values:
        if time.time() > deadline - 20:
            break
        
        found_3 = _find_3point_base(D_test, deadline)
        if found_3 is None:
            continue
        
        current = found_3
        for _ in range(5):  # Try to add up to 5 more points
            if time.time() > deadline - 20:
                break
            extended = _try_extend(current, D_test, deadline)
            if extended:
                current = extended
            else:
                break
        
        if len(current) > (len(best_points) if best_points else 0):
            best_points = current
            best_D = D_test
    
    # If we still only have 5 points, try harder with D=1
    if best_points is not None and len(best_points) <= 5 and time.time() < deadline - 10:
        # Try more aggressive random search
        random_extended = _try_random_extension(best_points, best_D, deadline, max_attempts=20000)
        if random_extended and len(random_extended) > len(best_points):
            best_points = random_extended
    
    # Format output
    if best_points is None:
        best_points = [(Fraction(x), Fraction(y)) for x, y in known_5]
        best_D = 1
    
    result_points = []
    for x, y in best_points:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': best_D, 'points': result_points}

# EVOLVE-BLOCK-END
