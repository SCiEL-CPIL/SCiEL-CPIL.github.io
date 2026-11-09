"""Search for integral point sets in general position."""
import time
import random
from fractions import Fraction
from math import isqrt, gcd
from itertools import combinations

# Deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 145.0


def _elapsed():
    return time.time() - _T0


def _deadline_hit():
    return _elapsed() > _DEADLINE


def _is_squarefree(n):
    if n <= 0:
        return False
    for p in range(2, isqrt(n) + 1):
        if n % (p * p) == 0:
            return False
    return True


def _is_rational_sqrt(d2):
    """Check if a Fraction d2 is a perfect square rational.
    Return (rn, rd) such that sqrt(d2) = rn/rd, or None."""
    if d2 < 0:
        return None
    num = d2.numerator
    den = d2.denominator
    g = gcd(num, den)
    num //= g
    den //= g
    rn = isqrt(num)
    rd = isqrt(den)
    if rn * rn == num and rd * rd == den:
        return (rn, rd)
    return None


def _collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)


def _four_concyclic(p1, p2, p3, p4, D):
    """Check if 4 points (x, y*sqrt(D)) are concyclic using the determinant."""
    M = []
    for (x, y) in (p1, p2, p3, p4):
        M.append([x * x + D * y * y, x, y, Fraction(1)])
    n = 4
    A = [row[:] for row in M]
    for c in range(n):
        pivot = None
        for r in range(c, n):
            if A[r][c] != 0:
                pivot = r
                break
        if pivot is None:
            return True
        if pivot != c:
            A[c], A[pivot] = A[pivot], A[c]
        for r in range(c + 1, n):
            factor = A[r][c] / A[c][c]
            for cc in range(c, n):
                A[r][cc] -= factor * A[c][cc]
    det = Fraction(1)
    for i in range(n):
        det *= A[i][i]
    return det == 0


def _check_general_position(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _four_concyclic(points[i], points[j], points[k], points[l], D):
                        return False
    return True


def _build_result(points, D):
    out = []
    for (x, y) in points:
        out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    return {'D': D, 'points': out}


def _score(points, D):
    return len(points)


def _valid_distance(x, y, px, py, D):
    """Check if distance squared between (x,y) and (px,py) is a perfect square rational."""
    dx = Fraction(x) - px
    dy = Fraction(y) - py
    d2 = dx * dx + D * dy * dy
    if d2 < 0:
        return False
    return _is_rational_sqrt(d2) is not None


def _try_add_point(current, new_pt, D):
    """Try to add new_pt to current set, checking all constraints."""
    n = len(current)
    # Check distances
    for (px, py) in current:
        if not _valid_distance(new_pt[0], new_pt[1], px, py, D):
            return False
    # Check collinearity
    for i in range(n):
        for j in range(i + 1, n):
            if _collinear(current[i], current[j], new_pt):
                return False
    # Check concyclicity
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _four_concyclic(current[i], current[j], current[k], new_pt, D):
                    return False
    return True


def _hilbert90_points(D, r, t_range):
    """
    Generate points on a circle of radius r in Q(sqrt(D)) using parameterization.
    Points: (r*(1-t^2)/(1+t^2), r*2t/(1+t^2)) in (x, y*sqrt(D)) coords
    where t is rational.
    """
    pts = []
    for t_num in range(-t_range, t_range + 1):
        for t_den in range(1, t_range + 1):
            if gcd(t_num, t_den) != 1:
                continue
            t = Fraction(t_num, t_den)
            denom = 1 + t * t
            x = r * (1 - t * t) / denom
            y = r * 2 * t / denom
            pts.append((x, y))
    return pts


def construct():
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = _T0 + 145.0

    best = None
    best_score = -1

    # Known small solutions to seed from
    # 5 points with D=1 (integer points, integer distances)
    known_5 = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
               (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
               (Fraction(16), Fraction(63))]

    # Validate the known 5
    ok = True
    for i in range(5):
        for j in range(i + 1, 5):
            if not _valid_distance(known_5[i][0], known_5[i][1], known_5[j][0], known_5[j][1], 1):
                ok = False
                break
        if not ok:
            break
    if ok and _check_general_position(known_5, 1):
        best = known_5
        best_score = 5

    # Try to find 6+ points using Hilbert 90 parameterization on circles
    # For D=1, points on unit circle: (cos θ, sin θ) with rational params
    # Use Pythagorean parameterization: t -> (1-t^2)/(1+t^2), 2t/(1+t^2)
    
    if not _deadline_hit():
        # Strategy 1: Try to find 6 points on/near a circle with D=1
        # Using rational points on the unit circle
        # Then perturb to break concyclicity while maintaining rational distances
        
        # First, try direct search with D=1 using greedy + random restarts
        random.seed(42)
        
        # Generate candidate points using Pythagorean parameterization
        # Points at distance r from origin: (r*(1-t^2)/(1+t^2), r*2t/(1+t^2))
        candidate_pool = []
        for r in [1, 2, 3, 4, 5]:
            for t_num in range(-20, 21):
                for t_den in range(1, 21):
                    if gcd(t_num, t_den) != 1:
                        continue
                    t = Fraction(t_num, t_den)
                    denom = 1 + t * t
                    x = r * (1 - t * t) / denom
                    y = r * 2 * t / denom
                    candidate_pool.append((x, y))
        
        # Also add integer lattice points
        for x in range(-100, 101):
            for y in range(-100, 101):
                candidate_pool.append((Fraction(x), Fraction(y)))
        
        # Greedy construction from multiple starting points
        for start_idx in range(min(50, len(candidate_pool))):
            if _deadline_hit():
                break
            
            current = [candidate_pool[start_idx]]
            # Try to grow the set
            for _ in range(20):
                if _deadline_hit():
                    break
                added = False
                # Try candidates in random order
                indices = list(range(len(candidate_pool)))
                random.shuffle(indices)
                for idx in indices:
                    if _deadline_hit():
                        break
                    p = candidate_pool[idx]
                    if p in current:
                        continue
                    if _try_add_point(current, p, 1):
                        current.append(p)
                        added = True
                        if len(current) > best_score:
                            best_score = len(current)
                            best = list(current)
                        break
                if not added:
                    break
        
        # Strategy 2: Random search with D=1
        if not _deadline_hit():
            max_coord = 200
            for _attempt in range(50):
                if _deadline_hit():
                    break
                pts = []
                tries = 0
                while len(pts) < 3 and tries < 2000:
                    tries += 1
                    x = random.randint(-max_coord, max_coord)
                    y = random.randint(-max_coord, max_coord)
                    p = (Fraction(x), Fraction(y))
                    if p in pts:
                        continue
                    valid = True
                    for (px, py) in pts:
                        if not _valid_distance(x, y, px, py, 1):
                            valid = False
                            break
                    if not valid:
                        continue
                    collinear_ok = True
                    for i in range(len(pts)):
                        for j in range(i + 1, len(pts)):
                            if _collinear(pts[i], pts[j], p):
                                collinear_ok = False
                                break
                        if not collinear_ok:
                            break
                    if not collinear_ok:
                        continue
                    pts.append(p)
                if len(pts) < 3:
                    continue
                current = list(pts)
                for _grow in range(15):
                    if _deadline_hit():
                        break
                    added = False
                    for _tr in range(300):
                        if _deadline_hit():
                            break
                        x = random.randint(-max_coord, max_coord)
                        y = random.randint(-max_coord, max_coord)
                        p = (Fraction(x), Fraction(y))
                        if p in current:
                            continue
                        if _try_add_point(current, p, 1):
                            current.append(p)
                            added = True
                            if len(current) > best_score:
                                best_score = len(current)
                                best = list(current)
                            break
                    if not added:
                        break
        
        # Strategy 3: Try D=3 with hexagon-like structures
        if not _deadline_hit():
            D_hex = 3
            # Regular hexagon vertices in (x, y*sqrt(3)) coordinates
            hexagon_6 = [
                (Fraction(1), Fraction(0)),
                (Fraction(1, 2), Fraction(1, 2)),
                (Fraction(-1, 2), Fraction(1, 2)),
                (Fraction(-1), Fraction(0)),
                (Fraction(-1, 2), Fraction(-1, 2)),
                (Fraction(1, 2), Fraction(-1, 2))
            ]
            
            # Check if hexagon has rational distances with D=3
            hex_ok = True
            for i in range(6):
                for j in range(i + 1, 6):
                    if not _valid_distance(hexagon_6[i][0], hexagon_6[i][1], 
                                           hexagon_6[j][0], hexagon_6[j][1], D_hex):
                        hex_ok = False
                        break
                if not hex_ok:
                    break
            
            if hex_ok:
                base = list(hexagon_6)
                # Try small rational perturbations to one point
                for perturb_idx in range(6):
                    if _deadline_hit():
                        break
                    for dx_n in range(-4, 5):
                        if _deadline_hit():
                            break
                        for dy_n in range(-4, 5):
                            if _deadline_hit():
                                break
                            if dx_n == 0 and dy_n == 0:
                                continue
                            for den in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                                if _deadline_hit():
                                    break
                                new_pt = (
                                    base[perturb_idx][0] + Fraction(dx_n, den),
                                    base[perturb_idx][1] + Fraction(dy_n, den)
                                )
                                candidate = list(base)
                                candidate[perturb_idx] = new_pt
                                
                                # Check all pairwise distances
                                valid = True
                                for i in range(6):
                                    for j in range(i + 1, 6):
                                        if not _valid_distance(candidate[i][0], candidate[i][1],
                                                               candidate[j][0], candidate[j][1], D_hex):
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if not valid:
                                    continue
                                
                                # Check general position
                                if _check_general_position(candidate, D_hex):
                                    if len(candidate) > best_score:
                                        best_score = len(candidate)
                                        best = list(candidate)
        
        # Strategy 4: Try to extend the known 5-point set
        if not _deadline_hit():
            current = list(known_5)
            max_coord = 300
            improved = True
            while improved and not _deadline_hit():
                improved = False
                for x in range(-max_coord, max_coord + 1):
                    if _deadline_hit():
                        break
                    for y in range(-max_coord, max_coord + 1):
                        if _deadline_hit():
                            break
                        new_pt = (Fraction(x), Fraction(y))
                        if new_pt in current:
                            continue
                        if _try_add_point(current, new_pt, 1):
                            current.append(new_pt)
                            improved = True
                            if len(current) > best_score:
                                best_score = len(current)
                                best = list(current)
                            break
                    if improved:
                        break

    # Determine D for the best set
    if best is not None and best_score >= 5:
        # Check if best is from hexagon (D=3)
        if best_score >= 6:
            # Verify with D=3 first
            D_test = 3
            valid = True
            for i in range(len(best)):
                for j in range(i + 1, len(best)):
                    if not _valid_distance(best[i][0], best[i][1], best[j][0], best[j][1], D_test):
                        valid = False
                        break
                if not valid:
                    break
            if valid and _check_general_position(best, D_test):
                return _build_result(best, D_test)
            # Try D=1
            D_test = 1
            valid = True
            for i in range(len(best)):
                for j in range(i + 1, len(best)):
                    if not _valid_distance(best[i][0], best[i][1], best[j][0], best[j][1], D_test):
                        valid = False
                        break
                if not valid:
                    break
            if valid and _check_general_position(best, D_test):
                return _build_result(best, D_test)
        return _build_result(best, 1)

    return _build_result(known_5, 1)
