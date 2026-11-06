"""Integral point sets in general position."""
import math
import time
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    def _gcd(a, b):
        a, b = abs(a), abs(b)
        while b:
            a, b = b, a % b
        return a if a != 0 else 1

    def _is_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    def _is_squarefree(n):
        if n == 1:
            return True
        if n % 4 == 0:
            return False
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    def _check_dist_rational(p1, p2, D):
        """Check if distance between two points (with y scaled by sqrt(D)) is rational."""
        x1, y1 = p1
        x2, y2 = p2
        dx = x1 - x2
        dy = y1 - y2
        d2 = dx * dx + D * dy * dy
        # d2 is a Fraction; check if it's a perfect square of a rational
        if d2 < 0:
            return False
        num, den = d2.numerator, d2.denominator
        if num == 0:
            return False  # same point
        return _is_square(num) and _is_square(den)

    def _no_three_collinear(pts):
        n = len(pts)
        for i, j, k in combinations(range(n), 3):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            x3, y3 = pts[k]
            # Cross product: (x2-x1)(y3-y1) - (x3-x1)(y2-y1)
            if (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1):
                return False
        return True

    def _no_four_concyclic(pts, D):
        """Check no four points are concyclic in the (x, y*sqrt(D)) plane."""
        n = len(pts)
        for i, j, k, l in combinations(range(n), 4):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            x3, y3 = pts[k]
            x4, y4 = pts[l]
            # In the plane with coordinates (X, Y) = (x, y*sqrt(D)):
            # Circle: X^2 + Y^2 + aX + bY + c = 0
            # => x^2 + D*y^2 + a*x + b*sqrt(D)*y + c = 0
            # For 4 points to be concyclic, the determinant must be 0:
            # | x_i^2 + D*y_i^2, x_i, y_i, 1 | (with appropriate scaling)
            # Actually, using (X,Y) = (x, y*sqrt(D)):
            # | X_i^2 + Y_i^2, X_i, Y_i, 1 |
            # But Y_i = y_i * sqrt(D), so let's use the form:
            # | x_i^2 + D*y_i^2, x_i, y_i*sqrt(D), 1 |
            # To avoid sqrt(D), multiply the Y column by sqrt(D):
            # | x_i^2 + D*y_i^2, x_i, y_i, 1 | but with the metric adjusted
            # Actually the cleanest: 4 points (x_i, y_i*sqrt(D)) are concyclic iff
            # det of [[x_i^2 + D*y_i^2, x_i, y_i, 1]] is 0 when we account for the metric.
            # Let me use: circle in (X,Y) plane: X^2+Y^2+aX+bY+c=0
            # X=x, Y=y*sqrt(D)
            # x^2 + D*y^2 + a*x + b*sqrt(D)*y + c = 0
            # For rational x,y to satisfy this for 4 points, we need the system to be consistent.
            # Equivalently, det of the 4x4 matrix with rows [x_i^2+D*y_i^2, x_i, y_i, 1] 
            # is NOT the right test because of the sqrt(D) factor on y.
            # Correct test: det of [[x_i^2 + D*y_i^2, x_i, y_i*sqrt(D), 1]] = 0
            # To avoid irrational, note that multiplying column 3 by sqrt(D) gives:
            # det of [[x_i^2 + D*y_i^2, x_i, y_i, 1]] * sqrt(D) ... no that's wrong.
            # Let me just use the standard 4-point concyclicity test with actual coordinates.
            # Point i in plane: (X_i, Y_i) = (x_i, y_i * sqrt(D))
            # Concyclic iff det [[X_i^2+Y_i^2, X_i, Y_i, 1]] = 0
            # = det [[x_i^2 + D*y_i^2, x_i, y_i*sqrt(D), 1]] = 0
            # This involves sqrt(D). But we can check: if D is a perfect square, use integer coords.
            # Otherwise, the determinant being 0 means a specific algebraic condition.
            # Simpler: 4 points are concyclic iff the circle through 3 passes through the 4th.
            # For D=1, all integer, easy. For general D, let's handle D=1 separately.
            if D == 1:
                # Standard integer test
                A = [[x1*x1+y1*y1, x1, y1, 1],
                     [x2*x2+y2*y2, x2, y2, 1],
                     [x3*x3+y3*y3, x3, y3, 1],
                     [x4*x4+y4*y4, x4, y4, 1]]
                # 4x4 determinant
                det = (A[0][0]*(A[1][1]*(A[2][2]*A[3][3]-A[2][3]*A[3][2]) - A[1][2]*(A[2][1]*A[3][3]-A[2][3]*A[3][1]) + A[1][3]*(A[2][1]*A[3][2]-A[2][2]*A[3][1]))
                     - A[0][1]*(A[1][0]*(A[2][2]*A[3][3]-A[2][3]*A[3][2]) - A[1][2]*(A[2][0]*A[3][3]-A[2][3]*A[3][0]) + A[1][3]*(A[2][0]*A[3][2]-A[2][2]*A[3][0]))
                     + A[0][2]*(A[1][0]*(A[2][1]*A[3][3]-A[2][3]*A[3][1]) - A[1][1]*(A[2][0]*A[3][3]-A[2][3]*A[3][0]) + A[1][3]*(A[2][0]*A[3][1]-A[2][1]*A[3][0]))
                     - A[0][3]*(A[1][0]*(A[2][1]*A[3][2]-A[2][2]*A[3][1]) - A[1][1]*(A[2][0]*A[3][2]-A[2][2]*A[3][0]) + A[1][2]*(A[2][0]*A[3][1]-A[2][1]*A[3][0])))
                if det == 0:
                    return False
            else:
                # For non-square D, use the fact that 4 points (x_i, y_i*sqrt(D)) are concyclic
                # iff there exist a, b, c such that x_i^2 + D*y_i^2 + a*x_i + b*y_i*sqrt(D) + c = 0
                # for all 4. This is a system in a, b*sqrt(D), c.
                # Let u_i = x_i^2 + D*y_i^2, v_i = x_i, w_i = y_i
                # We need: u_i + a*v_i + (b*sqrt(D))*w_i + c = 0
                # This means the vectors (u_i, v_i, w_i, 1) are linearly dependent over Q(sqrt(D)).
                # Equivalently, the 4x4 matrix [[u_i, v_i, w_i, 1]] has determinant 0 in Q(sqrt(D)).
                # Since u_i, v_i, w_i are rational, the determinant is rational.
                # So we can just check if the rational determinant is 0.
                u1 = x1*x1 + D*y1*y1
                u2 = x2*x2 + D*y2*y2
                u3 = x3*x3 + D*y3*y3
                u4 = x4*x4 + D*y4*y4
                A = [[u1, x1, y1, 1],
                     [u2, x2, y2, 1],
                     [u3, x3, y3, 1],
                     [u4, x4, y4, 1]]
                det = (A[0][0]*(A[1][1]*(A[2][2]*A[3][3]-A[2][3]*A[3][2]) - A[1][2]*(A[2][1]*A[3][3]-A[2][3]*A[3][1]) + A[1][3]*(A[2][1]*A[3][2]-A[2][2]*A[3][1]))
                     - A[0][1]*(A[1][0]*(A[2][2]*A[3][3]-A[2][3]*A[3][2]) - A[1][2]*(A[2][0]*A[3][3]-A[2][3]*A[3][0]) + A[1][3]*(A[2][0]*A[3][2]-A[2][2]*A[3][0]))
                     + A[0][2]*(A[1][0]*(A[2][1]*A[3][3]-A[2][3]*A[3][1]) - A[1][1]*(A[2][0]*A[3][3]-A[2][3]*A[3][0]) + A[1][3]*(A[2][0]*A[3][1]-A[2][1]*A[3][0]))
                     - A[0][3]*(A[1][0]*(A[2][1]*A[3][2]-A[2][2]*A[3][1]) - A[1][1]*(A[2][0]*A[3][2]-A[2][2]*A[3][0]) + A[1][2]*(A[2][0]*A[3][1]-A[2][1]*A[3][0])))
                if det == 0:
                    return False
        return True

    def _valid_set(pts, D):
        n = len(pts)
        if n < 3:
            return False
        # Check all pairwise distances are rational
        for i, j in combinations(range(n), 2):
            if not _check_dist_rational(pts[i], pts[j], D):
                return False
        # Check no three collinear
        if not _no_three_collinear(pts):
            return False
        # Check no four concyclic
        if n >= 4 and not _no_four_concyclic(pts, D):
            return False
        return True

    def _find_candidates_from_base(base_pts, D, max_coord=50):
        """Find all rational points at rational distance from all base points."""
        candidates = set()
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                pt = (Fraction(x), Fraction(y))
                if pt in base_pts:
                    continue
                ok = True
                for bp in base_pts:
                    if not _check_dist_rational(pt, bp, D):
                        ok = False
                        break
                if ok:
                    candidates.add(pt)
        return list(candidates)

    def _collinearity_penalty(pt, existing):
        """Count how many pairs in existing are collinear with pt."""
        penalty = 0
        n = len(existing)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = existing[i]
                x2, y2 = existing[j]
                x3, y3 = pt
                if (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1):
                    penalty += 1
        return penalty

    # Start with D=1 (integer lattice) for simplicity and speed
    D = 1

    # Phase 1: Try known good configurations
    known_configs = [
        [(0,0), (4,0), (1,3)],
        [(0,0), (3,0), (0,4)],
        [(0,0), (5,0), (3,4)],
        [(0,0), (4,0), (0,3)],
        [(0,0), (6,0), (2,3)],
        [(0,0), (9,0), (0,12)],
        [(0,0), (8,0), (3,4)],
        [(0,0), (7,0), (4,3)],
        [(0,0), (10,0), (6,8)],
        [(0,0), (12,0), (5,12)],
        [(0,0), (15,0), (8,12)],
        [(0,0), (5,0), (0,12)],
        [(0,0), (9,0), (0,12)],
        [(0,0), (20,0), (15,12)],
    ]

    # Convert to Fractions
    known_configs_frac = [[(Fraction(x), Fraction(y)) for x, y in cfg] for cfg in known_configs]

    best_set = None
    best_size = 0

    # Try to build up from each base triangle
    for base in known_configs_frac:
        if time.time() > deadline - 15:
            break
        if not _valid_set(base, D):
            continue

        current = list(base)
        
        # Iteratively try to add points
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 10:
                break
            
            # Find candidates at rational distance from all current points
            cands = _find_candidates_from_base(current, D, max_coord=40)
            
            # Sort by collinearity penalty (prefer points that break collinearity)
            cands.sort(key=lambda p: _collinearity_penalty(p, current))
            
            for pt in cands:
                if time.time() > deadline - 10:
                    break
                new_set = current + [pt]
                if _valid_set(new_set, D):
                    current = new_set
                    improved = True
                    break
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)
        
        if best_size >= 6:
            break

    # If we didn't find 6 points, try a more exhaustive search
    if best_size < 6 and time.time() < deadline - 20:
        # Try all small triangles and build up
        all_triangles = []
        for a in range(1, 15):
            for b in range(1, 15):
                for c in range(1, 15):
                    # Check if (a,b,c) forms a valid triangle with integer sides
                    # Place at (0,0), (a,0), and find third point
                    # Third point at (x,y) with x^2+y^2 = b^2 and (x-a)^2+y^2 = c^2
                    # x^2+y^2 = b^2, x^2-2ax+a^2+y^2 = c^2
                    # b^2 - 2ax + a^2 = c^2
                    # x = (b^2 + a^2 - c^2) / (2a)
                    num = b*b + a*a - c*c
                    den = 2*a
                    if num % den == 0:
                        x = num // den
                        y2 = b*b - x*x
                        if y2 >= 0 and _is_square(y2):
                            y = int(math.isqrt(y2))
                            if y > 0:
                                tri = [(Fraction(0), Fraction(0)), (Fraction(a), Fraction(0)), (Fraction(x), Fraction(y))]
                                if _valid_set(tri, D):
                                    all_triangles.append(tri)
        
        # Also try with different orientations
        for a in range(1, 12):
            for b in range(1, 12):
                for c in range(1, 12):
                    num = b*b + a*a - c*c
                    den = 2*a
                    if num % den == 0:
                        x = num // den
                        y2 = b*b - x*x
                        if y2 >= 0 and _is_square(y2):
                            y = int(math.isqrt(y2))
                            if y > 0:
                                # Try mirrored version
                                tri = [(Fraction(0), Fraction(0)), (Fraction(a), Fraction(0)), (Fraction(x), Fraction(-y))]
                                if _valid_set(tri, D) and tri not in all_triangles:
                                    all_triangles.append(tri)
        
        for tri in all_triangles:
            if time.time() > deadline - 10:
                break
            current = list(tri)
            improved = True
            while improved:
                improved = False
                if time.time() > deadline - 8:
                    break
                cands = _find_candidates_from_base(current, D, max_coord=35)
                cands.sort(key=lambda p: _collinearity_penalty(p, current))
                for pt in cands:
                    if time.time() > deadline - 8:
                        break
                    new_set = current + [pt]
                    if _valid_set(new_set, D):
                        current = new_set
                        improved = True
                        break
            
            if len(current) > best_size:
                best_size = len(current)
                best_set = list(current)
            if best_size >= 6:
                break

    # Fallback: use the known 5-point set
    if best_set is None or len(best_set) < 3:
        fallback = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)), 
                    (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)), 
                    (Fraction(16), Fraction(63))]
        if _valid_set(fallback, 1):
            best_set = fallback
        else:
            best_set = [(Fraction(0), Fraction(0)), (Fraction(4), Fraction(0)), (Fraction(1), Fraction(3))]

    # Try to extend the best set further
    if len(best_set) < 6 and time.time() < deadline - 5:
        current = list(best_set)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 3:
                break
            cands = _find_candidates_from_base(current, D, max_coord=50)
            cands.sort(key=lambda p: _collinearity_penalty(p, current))
            for pt in cands:
                if time.time() > deadline - 3:
                    break
                new_set = current + [pt]
                if _valid_set(new_set, D):
                    current = new_set
                    improved = True
                    break
        if len(current) > len(best_set):
            best_set = current

    # Convert to output format
    points_out = []
    for x, y in best_set:
        x_num, x_den = x.numerator, x.denominator
        y_num, y_den = y.numerator, y.denominator
        points_out.append([[x_num, x_den], [y_num, y_den]])

    return {'D': D, 'points': points_out}
# EVOLVE-BLOCK-END
