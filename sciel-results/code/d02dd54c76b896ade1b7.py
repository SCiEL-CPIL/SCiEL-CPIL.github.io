"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction
from functools import reduce
from itertools import combinations

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def is_square(n):
    if n < 0:
        return False
    s = math.isqrt(n)
    return s * s == n

def is_squarefree(n):
    if n <= 0:
        return False
    n = abs(n)
    for p in range(2, math.isqrt(n) + 1):
        if n % (p * p) == 0:
            return False
    return True

def make_squarefree(n):
    """Extract square factors from n, return (squarefree_part, square_factor_squared) such that n = sf * sf_sq^2."""
    if n == 0:
        return 0, 0
    n = abs(n)
    sf = 1
    sf_sq = 1
    p = 2
    while p * p <= n:
        while n % (p * p) == 0:
            n //= (p * p)
            sf_sq *= p
        p += 1
    sf = n
    return sf, sf_sq

# Points are represented as (x, y, D) where actual position is (x, y*sqrt(D)).
# x and y are Fractions. D is int (squarefree).

def dist_sq(p1, p2):
    """Return (num, den, D_effective) such that dist^2 = (num/den) + D_effective * ... wait no.
    dist^2 = (x1-x2)^2 + D*(y1-y2)^2.
    This is of the form A + B*D where A, B are rational.
    """
    x1, y1, D1 = p1
    x2, y2, D2 = p2
    # For points to be in the same field, we should have D1 == D2 in a standard setup.
    # Let's assume we are always working with a single D.
    if D1 != D2:
        # This shouldn't happen in our search strategy
        raise ValueError("Different D in distance calc")
    dx = x1 - x2
    dy = y1 - y2
    # dist_sq = dx^2 + D * dy^2
    # dx^2 is rational, D*dy^2 is D * rational.
    # We need dist_sq to be a perfect square of a rational number.
    # i.e. dist_sq = k^2 for some k in Q.
    val = dx * dx + D1 * dy * dy
    # val is a Fraction.
    return val

def dist_is_rational(p1, p2):
    d2 = dist_sq(p1, p2)
    return is_square(d2.numerator) and is_square(d2.denominator)

def collinear(p1, p2, p3):
    x1, y1, D = p1
    x2, y2, _ = p2
    x3, y3, _ = p3
    # det = (x2-x1)(y3-y1) - (x3-x1)(y2-y1) = 0
    # Note: y is coefficient of sqrt(D), so the geometry is in the plane.
    # Collinearity: (x2-x1)/(y2-y1) = (x3-x1)/(y3-y1) if y diffs non-zero.
    # More robust: (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) == 0
    return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0

def concyclic(p1, p2, p3, p4):
    # Points are concyclic iff the determinant of the circle equation is 0.
    # x^2 + y^2*D + Dx + Ey + F = 0 is not the general form in this metric.
    # Actually, circles in the Euclidean plane defined by these coordinates.
    # Let points be (xi, yi).
    # General circle: (x-h)^2 + (y-k)^2 = r^2  => x^2 - 2hx + h^2 + y^2 - 2ky + k^2 = r^2
    # x^2 + y^2 - 2hx - 2ky + (h^2+k^2-r^2) = 0
    # Here "y^2" means (y*sqrt(D))^2 = y^2 D.
    # So: x^2 + D y^2 - 2h x - 2k y + C = 0
    # We check if there exist h, k, C such that this holds for 4 points.
    x1, y1, _ = p1
    x2, y2, _ = p2
    x3, y3, _ = p3
    x4, y4, _ = p4
    # Equations:
    # x1^2 + D y1^2 - 2h x1 - 2k y1 + C = 0
    # x2^2 + D y2^2 - 2h x2 - 2k y2 + C = 0
    # x3^2 + D y3^2 - 2h x3 - 2k y3 + C = 0
    # x4^2 + D y4^2 - 2h x4 - 2k y4 + C = 0
    
    # This is a linear system for 2h, 2k, C.
    # Let A = 2h, B = 2k, C = C.
    # -x1 A - y1 B + C = -(x1^2 + D y1^2)
    # -x2 A - y2 B + C = -(x2^2 + D y2^2)
    # -x3 A - y3 B + C = -(x3^2 + D y3^2)
    # -x4 A - y4 B + C = -(x4^2 + D y4^2)
    
    # Subtract row 1 from rows 2, 3, 4:
    # -(x2-x1) A - (y2-y1) B = -(x2^2 - x1^2 + D(y2^2 - y1^2))
    # -(x3-x1) A - (y3-y1) B = -(x3^2 - x1^2 + D(y3^2 - y1^2))
    # -(x4-x1) A - (y4-y1) B = -(x4^2 - x1^2 + D(y4^2 - y1^2))
    
    # 3 equations, 2 unknowns (A, B).
    # For a solution to exist, the 3x3 determinant (or rank condition) must hold.
    # Specifically, the vector of RHS must be in the span of the columns of the coefficient matrix.
    
    x1, y1, D = p1
    x2, y2, _ = p2
    x3, y3, _ = p3
    x4, y4, _ = p4
    
    def rhs(p):
        return -(p[0]**2 + D * p[1]**2)
    
    r1 = rhs(p1)
    r2 = rhs(p2)
    r3 = rhs(p3)
    r4 = rhs(p4)
    
    # Coeff matrix M (3x2):
    # [-dx2 -dy2]
    # [-dx3 -dy3]
    # [-dx4 -dy4]
    
    # We can check if the 3x3 matrix formed by adding the RHS as a column has determinant 0.
    # Matrix:
    # [ -dx2  -dy2  r2-r1 ]
    # [ -dx3  -dy3  r3-r1 ]
    # [ -dx4  -dy4  r4-r1 ]
    
    a1 = -(x2 - x1)
    b1 = -(y2 - y1)
    c1 = r2 - r1
    
    a2 = -(x3 - x1)
    b2 = -(y3 - y1)
    c2 = r3 - r1
    
    a3 = -(x4 - x1)
    b3 = -(y4 - y1)
    c3 = r4 - r1
    
    # Det = a1(b2*c3 - c2*b3) - b1(a2*c3 - c2*a3) + c1(a2*b3 - b2*a3)
    det = a1 * (b2 * c3 - c2 * b3) - b1 * (a2 * c3 - c2 * a3) + c1 * (a2 * b3 - b2 * a3)
    return det == 0

def valid_set(points):
    n = len(points)
    # Check distinct
    for i in range(n):
        for j in range(i+1, n):
            if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                return False
    # Check rational distances
    for i in range(n):
        for j in range(i+1, n):
            if not dist_is_rational(points[i], points[j]):
                return False
    # Check no 3 collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if collinear(points[i], points[j], points[k]):
                    return False
    # Check no 4 concyclic
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def find_next_point(points, D, limit=10**4, timeout=5):
    """Search for a new point (x, y, D) such that dist to all existing points is rational."""
    start = time.time()
    # Heuristic: search in a box around the existing points or on a grid.
    # To find rational distances, we can look for points that complete rational triangles.
    
    # A faster search:
    # Iterate over small integer x, y? No, x, y are rational.
    # Let's search for integer coordinates first (scaled by denominator L).
    # Or just search small integers for x, y.
    
    # Better: Use the property that if (0,0) and (a,0) are in set,
    # new point (x,y) must satisfy x^2 + D y^2 = r1^2 and (x-a)^2 + D y^2 = r2^2.
    # => 2ax - a^2 = r1^2 - r2^2 => x = (r1^2 - r2^2 + a^2) / 2a.
    # This requires r1^2 - r2^2 + a^2 to be divisible by 2a (in rationals, always true).
    # So for any pair of rational distances r1, r2 from the pair, we get a candidate x.
    # Then y^2 = (r1^2 - x^2) / D. We need this to be a rational square.
    
    # So we iterate over possible rational distances r1, r2.
    # r1^2 = x1^2 + D y1^2.
    
    # Let's try a simpler grid search for small integers first.
    # If points are small integers, a 6th integer point might be nearby.
    
    # However, rational points are dense.
    # Let's try to add a point by intersecting two circles.
    # Pick two points p1, p2 from the set.
    # We need a point P such that d(P,p1)=r1 and d(P,p2)=r2 with r1,r2 rational.
    # As derived, x is fixed by r1, r2, p1, p2.
    # y^2 is then determined. We need y^2 to be a rational square *and* y rational.
    
    # So we search for rational squares s1, s2 such that:
    # x = (s1 - s2 + d12^2) / (2 * d12_x)  (if d12_x != 0)
    # y^2 = (s1 - x^2) / D is a rational square.
    
    # This is still hard to search directly.
    
    # Alternative: Search small integer grid for D=1.
    # If D=1, we are looking for integer points with integer distances.
    # Known sets:
    # 5 points: (0,0), (100,0), (150,200), (260,150), (50,0) - wait, check collinear.
    # (0,0), (100,0), (50,0) are collinear. Bad.
    
    # Let's use a known 5-point set and try to extend.
    # Seed set: (0,0), (-56, 33), (-16, 63), (0, 33), (16, 63).
    # D=1.
    
    # Try to find a 6th point (x,y) with integer coords in a reasonable range.
    # Range: -200 to 200.
    
    if D != 1:
        # For D != 1, search is much harder. Return None.
        # We can try small integers for x, y and check if y^2 * D + x^2 is square.
        # dist^2 = (x-xi)^2 + D(y-yi)^2.
        # This must be a square of a rational.
        # If x, y are integers, dist^2 is integer. Must be perfect square.
        pass
    
    # Grid search for integer points
    # We will check all integer (x,y) in a box.
    # To be fast, we only check if it's valid against ALL points.
    
    # Box size depends on time.
    # Start with small box, expand if time allows.
    
    # Optimization: Pre-compute constraints.
    # For each candidate (x,y), we need to check N distances.
    
    # Let's just do a brute force grid search for D=1.
    # If D=1, valid points are integer points with integer distances to all others.
    
    # We can also try rational points with small denominators.
    # But integer is a good start.
    
    # Let's try to find a point that is an integer distance from the first point (0,0)
    # and the second point.
    
    # Actually, let's just do a full grid search for integer coords.
    # It's the most robust for D=1.
    
    # If D=1:
    #   Check all (x,y) in [-R, R]^2.
    
    # If D != 1:
    #   Check all (x,y) in [-R, R]^2 such that for all p, (x-px)^2 + D(y-py)^2 is a perfect square (of a rational, but if x,y int, then integer).
    
    R = 50
    # Expand R if we have time?
    # For now, try R=100.
    
    # We can filter candidates early.
    # Must be integer distance to p0.
    # (x-x0)^2 + D(y-y0)^2 = k^2
    
    # Let's iterate over k (distance to p0) and find integer solutions.
    # This is a number theory problem.
    
    # Simpler: just iterate x, y.
    
    p0 = points[0]
    
    for x in range(-R, R + 1):
        if time.time() - start > timeout:
            return None
        for y in range(-R, R + 1):
            # Quick check: distance to p0
            # (x - x0)^2 + D(y - y0)^2 must be a square (of rational)
            # If x,y are integers, this is an integer. Must be perfect square.
            
            # Check if (x,y) is already in points
            # Skip if same x,y
            
            valid_so_far = True
            for p in points:
                d2 = (x - p[0].numerator)**2 + D * (y - p[1].numerator)**2
                # Note: p[0] is Fraction. If p[0] is integer, numerator works.
                # If p[0] is not integer, this grid search fails.
                # For the seed set, points are integers.
                if not is_square(d2):
                    valid_so_far = False
                    break
            
            if valid_so_far:
                # Now check collinear and concyclic with existing points
                # We need to check if adding (x,y,D) creates a violation.
                # We can just add it and check the whole set?
                # No, just check triples with (x,y) and pairs with (x,y).
                
                new_pt = (Fraction(x), Fraction(y), D)
                
                # Check collinearity with any pair of existing points
                coll = False
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        if collinear(points[i], points[j], new_pt):
                            coll = True
                            break
                    if coll:
                        break
                if coll:
                    continue
                
                # Check concyclicity with any triple of existing points
                conc = False
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        for k in range(j+1, len(points)):
                            if concyclic(points[i], points[j], points[k], new_pt):
                                conc = True
                                break
                        if conc:
                            break
                    if conc:
                        break
                if conc:
                    continue
                
                return new_pt
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s
    
    # Start with the seed 5 points
    D = 1
    seed_pts = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    points = [(Fraction(x), Fraction(y), D) for x, y in seed_pts]
    
    best_points = points
    
    # Verify seed is valid
    if not valid_set(points):
        # Fallback to a simple triangle
        points = [(Fraction(0), Fraction(0), D), (Fraction(3), Fraction(0), D), (Fraction(0), Fraction(4), D)]
        best_points = points
        # Try to extend this?
    
    # Try to add a 6th point
    # We can try different search strategies or ranges.
    
    # Strategy 1: Grid search for integer point
    # We already did R=50 in find_next_point? No, R was hardcoded.
    # Let's make find_next_point smarter or just call it with a larger timeout.
    
    new_pt = find_next_point(points, D, timeout=10)
    if new_pt:
        points.append(new_pt)
        if valid_set(points):
            best_points = points
    
    # If we have 6, try to add a 7th?
    # Unlikely to find via simple grid.
    
    # Strategy 2: Try to construct a different set that might be larger.
    # There are known 6-point rational distance sets.
    # One example:
    # (0,0), (1,0), (0,1), (1,1) is 4.
    # Add (1/2, 0)? No, collinear.
    
    # Let's try to search for a 6-point set with D=1 using a more directed search.
    # Start with 4 points that form a "rational rectangle" or similar.
    
    # Actually, let's just return the best found.
    
    # If we have time, try a random search for rational points?
    # Too complex for the remaining time.
    
    # Convert back to output format
    # {'D': D, 'points': [[[x_num, x_den], [y_num, y_den]], ...]}
    
    final_points = []
    for p in best_points:
        x, y, _ = p
        final_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': final_points}

# EVOLVE-BLOCK-END
