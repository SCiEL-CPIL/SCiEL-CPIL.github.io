import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill

    # Helper: exact integer square check
    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    # Helper: check if a rational number (as Fraction) is a perfect square
    def is_square_frac(f):
        if f < 0:
            return False
        num = f.numerator
        den = f.denominator
        if not (is_square(num) and is_square(den)):
            return False
        return True

    # Helper: rational distance squared between two points (x, y) where coords are Fractions
    def dist_sq_frac(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    # Helper: check three points are not collinear (exact rational)
    def not_collinear(p1, p2, p3):
        # Cross product: (p2-p1) x (p3-p1)
        val = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        return val != 0

    # Helper: check four points are not concyclic (exact rational)
    def not_concyclic(p1, p2, p3, p4):
        # Determinant of:
        # [x^2+y^2, x, y, 1]
        # For 4 points, if det != 0, not concyclic
        def det4(m):
            return (
                m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
                - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
                + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
            )
        M = [[p[0]**2 + p[1]**2, p[0], p[1], Fraction(1)] for p in [p1, p2, p3, p4]]
        return det4(M) != 0

    def check_validity(points):
        n = len(points)
        # All pairwise distances must be rational (squared distance is a perfect square of a rational)
        for i in range(n):
            for j in range(i + 1, n):
                d2 = dist_sq_frac(points[i], points[j])
                if not is_square_frac(d2):
                    return False
        # No three collinear
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if not not_collinear(points[i], points[j], points[k]):
                        return False
        # No four concyclic
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if not not_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    # Strategy: Use the known 5-point configuration but search more effectively for a 6th.
    # The 5 base points from the seed:
    # P1=(0,0), P2=(-56,33), P3=(-16,63), P4=(0,33), P5=(16,63)
    # These are integer points with integer pairwise distances.
    
    base_points = [
        (Fraction(0), Fraction(0)),
        (Fraction(-56), Fraction(33)),
        (Fraction(-16), Fraction(63)),
        (Fraction(0), Fraction(33)),
        (Fraction(16), Fraction(63)),
    ]
    
    # Verify base set is valid
    if not check_validity(base_points):
        # Fallback: return 3 points
        pts3 = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)), (Fraction(0), Fraction(12))]
        return {'D': 1, 'points': [[[p[0].numerator, p[0].denominator], [p[1].numerator, p[1].denominator]] for p in pts3]}

    # Search for 6th point using parametric approach.
    # For a point (x,y) to have rational distance to P1=(0,0): x^2+y^2 = r1^2
    # To have rational distance to P4=(0,33): x^2+(y-33)^2 = r4^2
    # Subtract: 66y - 1089 = r1^2 - r4^2 => y = (r1^2 - r4^2 + 1089)/66
    # Then x^2 = r1^2 - y^2 must be a rational square.
    
    # We search over rational r1, r4 parameterized by Pythagorean parameters.
    # r1 = a1^2 + b1^2 (scaled), actually let's search integer r1, r4 first since
    # the base points are integer, rational distances from integer points to integer points
    # are integers.
    
    found_point = None
    
    # Search range: distances up to ~200
    max_r = 250
    
    for r1 in range(1, max_r + 1):
        if time.time() > deadline:
            break
        r1_sq = r1 * r1
        for r4 in range(1, max_r + 1):
            r4_sq = r4 * r4
            num_y = r1_sq - r4_sq + 1089
            if num_y % 66 != 0:
                continue
            y = num_y // 66
            x2 = r1_sq - y * y
            if x2 < 0:
                continue
            if not is_square(x2):
                continue
            x = math.isqrt(x2)
            for cx in [x, -x]:
                if cx == 0 and x == 0:
                    # only one candidate
                    candidates = [(Fraction(0), Fraction(y))]
                else:
                    candidates = [(Fraction(cx), Fraction(y))]
                for (px, py) in candidates:
                    P6 = (px, py)
                    # Check distinctness
                    if P6 in base_points:
                        continue
                    # Check integer distances to P2, P3, P5
                    ok = True
                    for bp in [base_points[1], base_points[2], base_points[4]]:
                        d2 = dist_sq_frac(P6, bp)
                        if not is_square_frac(d2):
                            ok = False
                            break
                    if not ok:
                        continue
                    # Check no three collinear with P6
                    ok = True
                    for i in range(5):
                        for j in range(i + 1, 5):
                            if not not_collinear(P6, base_points[i], base_points[j]):
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    # Check no four concyclic with P6
                    ok = True
                    for i in range(5):
                        for j in range(i + 1, 5):
                            for k in range(j + 1, 5):
                                if not not_concyclic(P6, base_points[i], base_points[j], base_points[k]):
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    # Full validation
                    new_set = base_points + [P6]
                    if check_validity(new_set):
                        found_point = P6
                        break
                if found_point:
                    break
        if found_point:
            break

    if found_point:
        final_points = base_points + [found_point]
    else:
        final_points = base_points

    # Convert to required format: D=1, points as [[x_num, x_den], [y_num, y_den]]
    D = 1
    points_out = []
    for p in final_points:
        x_num, x_den = p[0].numerator, p[0].denominator
        y_num, y_den = p[1].numerator, p[1].denominator
        # Reduce signs
        if x_den < 0:
            x_num, x_den = -x_num, -x_den
        if y_den < 0:
            y_num, y_den = -y_num, -y_den
        points_out.append([[x_num, x_den], [y_num, y_den]])

    return {'D': D, 'points': points_out}
