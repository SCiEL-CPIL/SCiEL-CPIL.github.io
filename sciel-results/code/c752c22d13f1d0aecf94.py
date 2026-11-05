"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 128.0  # safety margin before 150s kill (allow for evaluation overhead)

    def elapsed():
        return time.time() - start_time

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return abs(a)

    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def check_dist_int(p1, p2, D):
        """Check if squared distance between p1=(x1,y1) and p2=(x2,y2) in Q(sqrt(D)) is a perfect square.
        Points are (x, y*sqrt(D)) where x,y are Fractions.
        dist^2 = (x1-x2)^2 + D*(y1-y2)^2"""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        # dist^2 = dx^2 + D*dy^2
        val = dx * dx + D * dy * dy
        # val is a Fraction
        if val.denominator == 1:
            return is_square(val.numerator)
        # dist^2 = val.numerator / val.denominator
        # For dist to be rational, val must be a perfect square of a rational
        # i.e., val = (a/b)^2 where a,b integers
        # Check if val.numerator and val.denominator are both perfect squares
        # (since val > 0 for distinct points)
        n = val.numerator
        d = val.denominator
        return is_square(n) and is_square(d)

    def check_collinear(p1, p2, p3, D):
        """Check if 3 points are collinear. Points are (x, y*sqrt(D)).
        Collinear iff (x1-x2)(y3*sqrt(D)-y1*sqrt(D)) = (y2*sqrt(D)-y1*sqrt(D))(x3-x1)
        which simplifies to (x1-x2)(y3-y1) = (y2-y1)(x3-x1) since sqrt(D) cancels."""
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]
        return (x1 - x2) * (y3 - y1) == (y2 - y1) * (x3 - x1)

    def check_concyclic(p1, p2, p3, p4, D):
        """Check if 4 points are concyclic.
        Use the determinant of the 4x4 matrix:
        | x  y*sqrt(D)  x^2 + D*y^2  1 |
        
        Since sqrt(D) is irrational and D is not a perfect square, we can separate rational and sqrt(D) parts.
        Actually, for points in Q(sqrt(D)), 4 points are concyclic iff the determinant is 0.
        The determinant will be of the form a + b*sqrt(D). For it to be 0, we need a=0 and b=0.
        
        Let's compute the determinant symbolically or use the circle equation approach.
        Circle equation: x^2 + D*y^2 + Ax + By*sqrt(D) + C = 0
        Wait, the general circle in the (x, y*sqrt(D)) plane:
        (x - h)^2 + (Y - k)^2 = r^2 where Y = y*sqrt(D)
        x^2 - 2hx + h^2 + Y^2 - 2kY + k^2 = r^2
        x^2 + D*y^2 - 2hx - 2k*y*sqrt(D) + (h^2 + k^2 - r^2) = 0
        
        So the circle equation is: x^2 + D*y^2 + A*x + B*y*sqrt(D) + C = 0
        where A, B, C are rational.
        
        For 4 points to be concyclic, there exist rational A, B, C such that all 4 satisfy this.
        This is a linear system in A, B, C. 4 points give 4 equations, 3 unknowns.
        For a solution to exist, the 4x3 matrix must have rank <= 2 (or the augmented system must be consistent).
        
        Actually, let's use the determinant approach directly.
        The 4 points (xi, yi*sqrt(D)) are concyclic iff:
        det [[xi, yi*sqrt(D), xi^2 + D*yi^2, 1] for i=1..4] = 0
        
        Let's compute this determinant. Each entry is of the form rational or rational*sqrt(D).
        The determinant will be of the form P + Q*sqrt(D) where P, Q are rational.
        For this to be 0, we need P = 0 and Q = 0.
        
        Let's compute using exact arithmetic with Fractions, tracking rational and sqrt(D) parts separately.
        """
        # Represent each row as (rational_part, sqrt_D_part)
        # x = rational, y*sqrt(D) = rational * sqrt(D), x^2 + D*y^2 = rational, 1 = rational
        rows = []
        for p in [p1, p2, p3, p4]:
            x, y = p[0], p[1]
            # x is rational, y*sqrt(D) has y rational (so y*sqrt(D) = 0 + y*sqrt(D))
            # x^2 + D*y^2 is rational
            # 1 is rational
            rows.append((x, y, x*x + D*y*y, Fraction(1)))
        
        # The matrix has entries:
        # col0: rational
        # col1: sqrt(D) part (coefficient is y, rational)
        # col2: rational
        # col3: rational
        #
        # det = sum over permutations of sign * prod(entries)
        # Each product has a form: (rational product) * (sqrt(D)^k) where k is the number of col1 entries in the permutation
        # k is either 0 or 1 (since each row contributes exactly one entry, and col1 is one column)
        # Actually, in the determinant expansion, each term picks one entry from each row and column.
        # Col1 has entries y1*sqrt(D), y2*sqrt(D), y3*sqrt(D), y4*sqrt(D)
        # So each term that uses col1 will have exactly one factor of sqrt(D).
        # Terms that don't use col1 have no sqrt(D).
        
        # Let's compute det4 by expanding along the first row.
        # M[i][j] where:
        # j=0: rows[i][0] (rational)
        # j=1: rows[i][1] (coefficient of sqrt(D), so entry is rows[i][1]*sqrt(D))
        # j=2: rows[i][2] (rational)
        # j=3: rows[i][3] (rational)
        
        def det3_rational(m):
            """Determinant of 3x3 where m[i][j] is (rat, sqrt_coeff) but for minors from col expansion,
            we'll have pure rational or pure sqrt coefficients."""
            # Actually, let's just compute the determinant symbolically.
            # For a 3x3 matrix where each entry is (a, b) meaning a + b*sqrt(D):
            # det = (a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31))
            # But we need to track (rational, sqrt) parts.
            # Let entry(i,j) = A[i][j] + B[i][j]*sqrt(D)
            # Product of 3 entries: (A1+A1's*B1*sqrt(D))*(A2+A2's*B2*sqrt(D))*(A3+A3's*B3*sqrt(D))
            # This gets complex. Let's use a different approach.
            
            pass
        
        # Simpler: compute the determinant as P + Q*sqrt(D) directly.
        # Expand along row 0:
        # det = M00*C00 - M01*C01 + M02*C02 - M03*C03
        # where Cij is the cofactor.
        
        def compute_det():
            # Each row: (x, y, x2dy2, one)
            # Entry (i,0) = rows[i][0]  (rational)
            # Entry (i,1) = rows[i][1]  (this is the coefficient of sqrt(D), so entry = rows[i][1]*sqrt(D))
            # Entry (i,2) = rows[i][2]  (rational)
            # Entry (i,3) = rows[i][3]  (rational)
            
            # Expand along row 0:
            # det = e00*det(M without row0,col0) - e01*det(M without row0,col1) + e02*det(M without row0,col2) - e03*det(M without row0,col3)
            
            # e00 = rows[0][0] (rational)
            # Minor 00: rows 1,2,3 and cols 0,2,3 (all rational entries)
            m00 = [
                [rows[1][0], rows[1][2], rows[1][3]],
                [rows[2][0], rows[2][2], rows[2][3]],
                [rows[3][0], rows[3][2], rows[3][3]]
            ]
            det00 = (m00[0][0]*(m00[1][1]*m00[2][2] - m00[1][2]*m00[2][1])
                    - m00[0][1]*(m00[1][0]*m00[2][2] - m00[1][2]*m00[2][0])
                    + m00[0][2]*(m00[1][0]*m00[2][1] - m00[1][1]*m00[2][0]))
            
            # e01 = rows[0][1]*sqrt(D)
            # Minor 01: rows 1,2,3 and cols 0,2,3 (same as above, but sign is -)
            # Actually the minor for (0,1) uses cols 0,2,3 which is the same set as for (0,0)
            # Wait, col 1 is removed, so remaining cols are 0,2,3. Yes same.
            det01 = det00  # Same minor
            
            # e02 = rows[0][2] (rational)
            # Minor 02: rows 1,2,3 and cols 0,1,3
            # Col 1 has sqrt(D) entries
            m02 = [
                [rows[1][0], rows[1][1], rows[1][3]],
                [rows[2][0], rows[2][1], rows[2][3]],
                [rows[3][0], rows[3][1], rows[3][3]]
            ]
            # Each entry in col 1 is y*sqrt(D), so entries are (0, y_i) in (rat, sqrt_coeff) form
            # det of this 3x3: expand along col 1 (the sqrt(D) column)
            # = -m[0][1]*(m[1][0]*m[2][2] - m[2][0]*m[1][2]) + m[1][1]*(m[0][0]*m[2][2] - m[2][0]*m[0][2]) - m[2][1]*(m[0][0]*m[1][2] - m[1][0]*m[0][2])
            # The coefficients of sqrt(D):
            # m[0][1] = rows[1][1], m[1][1] = rows[2][1], m[2][1] = rows[3][1]
            # Other entries are rational.
            sqrt_coeff02 = (-m02[0][1]*(m02[1][0]*m02[2][2] - m02[2][0]*m02[1][2])
                           + m02[1][1]*(m02[0][0]*m02[2][2] - m02[2][0]*m02[0][2])
                           - m02[2][1]*(m02[0][0]*m02[1][2] - m02[1][0]*m02[0][2]))
            # So det02 = 0 + sqrt_coeff02*sqrt(D)
            
            # e03 = rows[0][3] = 1 (rational)
            # Minor 03: rows 1,2,3 and cols 0,1,2
            m03 = [
                [rows[1][0], rows[1][1], rows[1][2]],
                [rows[2][0], rows[2][1], rows[2][2]],
                [rows[3][0], rows[3][1], rows[3][2]]
            ]
            # Col 1 has sqrt(D) entries
            sqrt_coeff03 = (-m03[0][1]*(m03[1][0]*m03[2][2] - m03[2][0]*m03[1][2])
                           + m03[1][1]*(m03[0][0]*m03[2][2] - m03[2][0]*m03[0][2])
                           - m03[2][1]*(m03[0][0]*m03[1][2] - m03[1][0]*m03[0][2]))
            # det03 = sqrt_coeff03*sqrt(D)
            
            # Total det = e00*det00 - e01*det01 + e02*det02 - e03*det03
            # = rows[0][0]*det00 - rows[0][1]*sqrt(D)*det01 + rows[0][2]*sqrt_coeff02*sqrt(D) - rows[0][3]*sqrt_coeff03*sqrt(D)
            # Rational part: rows[0][0]*det00
            # Sqrt(D) part: -rows[0][1]*det01 + rows[0][2]*sqrt_coeff02 - rows[0][3]*sqrt_coeff03
            
            rat_part = rows[0][0] * det00
            sqrt_part = (-rows[0][1] * det01 
                       + rows[0][2] * sqrt_coeff02 
                       - rows[0][3] * sqrt_coeff03)
            
            return rat_part, sqrt_part
        
        rat, sqrt_coeff = compute_det()
        return rat == 0 and sqrt_coeff == 0

    def all_constraints_satisfied(points, D):
        n = len(points)
        # Check all pairwise distances are rational (squared distance is perfect square of rational)
        for i in range(n):
            for j in range(i+1, n):
                if not check_dist_int(points[i], points[j], D):
                    return False
        # Check no 3 collinear
        for combo in combinations(range(n), 3):
            if check_collinear(points[combo[0]], points[combo[1]], points[combo[2]], D):
                return False
        # Check no 4 concyclic
        for combo in combinations(range(n), 4):
            if check_concyclic(points[combo[0]], points[combo[1]], 
                             points[combo[2]], points[combo[3]], D):
                return False
        return True

    # Generate a valid set using symmetric construction
    # Try D=3: embed regular hexagon vertices
    # Regular hexagon with side length s: vertices at (s*cos(k*pi/3), s*sin(k*pi/3)) for k=0..5
    # cos(pi/3)=1/2, sin(pi/3)=sqrt(3)/2
    # So vertices: (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2), (s, 0)
    # In our form (x, y*sqrt(D)) with D=3: y*sqrt(3) = y' where y' = s/2, so y = s/(2*sqrt(3)) = s*sqrt(3)/6
    # That's not rational y. 
    
    # Alternative: use D such that sin and cos are rational multiples of sqrt(D)
    # For a square: D=1, points (0,0), (1,0), (1,1), (0,1) - but these are concyclic? No, they form a square, which IS cyclic.
    # Actually any rectangle is cyclic. So 4 points forming a rectangle are concyclic. Bad.
    
    # Let's use the known 6-point set with D=1
    known_6pt_D1 = [(Fraction(0), Fraction(0)), (Fraction(25), Fraction(0)), (Fraction(14), Fraction(24)), 
                    (Fraction(24), Fraction(7)), (Fraction(0), Fraction(40)), (Fraction(15), Fraction(40))]
    
    if not all_constraints_satisfied(known_6pt_D1, 1):
        # Try another known set
        known_6pt_D1 = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)), (Fraction(-16), Fraction(63)), 
                        (Fraction(0), Fraction(33)), (Fraction(16), Fraction(63)), (Fraction(56), Fraction(33))]
    
    best_points = known_6pt_D1[:]
    best_D = 1
    best_len = len(best_points)
    
    # Try to find a 7-point configuration using search
    # Strategy: for D=1, search for integer lattice points
    def find_7pt_D1(deadline):
        """Search for 7-point integral set in Z^2 (D=1)."""
        # Use the 6-point set as seed and try to add a 7th
        base = best_points[:]
        
        # Generate candidates that are at integer distance from all 6 base points
        candidates = []
        max_coord = 200
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                cand = (Fraction(x), Fraction(y))
                if any(cand == bp for bp in base):
                    continue
                ok = True
                for bp in base:
                    dx = cand[0] - bp[0]
                    dy = cand[1] - bp[1]
                    d2 = dx*dx + dy*dy
                    if not is_square(d2.numerator) or d2.denominator != 1:
                        ok = False
                        break
                if ok:
                    candidates.append(cand)
        
        # Sort by distance from origin
        candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
        
        # Try adding each candidate
        for cand in candidates:
            if elapsed() > deadline:
                break
            new_pts = base + [cand]
            if all_constraints_satisfied(new_pts, 1):
                return new_pts
        return None
    
    if best_len < 7 and elapsed() < DEADLINE - 10:
        result = find_7pt_D1(DEADLINE - 5)
        if result is not None:
            best_points = result
            best_D = 1
            best_len = len(best_points)
    
    # If still 6 points, try other small D values with symmetric constructions
    # Try D=2, D=3, D=5, D=6, D=7, D=10, D=11, D=13, D=14, D=15
    for D in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15]:
        if elapsed() > DEADLINE - 5:
            break
        # Try to construct points using a symmetric pattern
        # Use points on a circle: (r*cos(theta), r*sin(theta)) in the (x, y*sqrt(D)) plane
        # For the distance to be rational, we need specific angles.
        
        # Alternative: use a grid in Q x Q(sqrt(D))
        # Try small integer coordinates
        candidates_D = []
        for x in range(-10, 11):
            for y in range(-10, 11):
                candidates_D.append((Fraction(x), Fraction(y)))
        
        # Filter to those at rational distance from (0,0)
        filtered = []
        for cand in candidates_D:
            dx = cand[0]
            dy = cand[1]
            d2 = dx*dx + D*dy*dy
            if d2.denominator == 1 and is_square(d2.numerator):
                filtered.append(cand)
        
        # Try greedy construction
        pts = [(Fraction(0), Fraction(0))]
        for cand in filtered:
            if elapsed() > DEADLINE - 3:
                break
            if cand == (Fraction(0), Fraction(0)):
                continue
            new_pts = pts + [cand]
            if all_constraints_satisfied(new_pts, D):
                pts = new_pts
                if len(pts) > best_len:
                    best_points = pts
                    best_D = D
                    best_len = len(pts)
                if len(pts) >= 7:
                    break
            if len(pts) >= 7:
                break
    
    # Final fallback
    if best_len < 3:
        best_points = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)), (Fraction(4), Fraction(0))]
        best_D = 1
    
    # Convert to required format
    D = best_D
    points_out = []
    for x, y in best_points:
        # x = x_num / x_den, y = y_num / y_den
        x_num = x.numerator
        x_den = x.denominator
        y_num = y.numerator
        y_den = y.denominator
        points_out.append([[x_num, x_den], [y_num, y_den]])
    
    return {'D': D, 'points': points_out}
# EVOLVE-BLOCK-END
