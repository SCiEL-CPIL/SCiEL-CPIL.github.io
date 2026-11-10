import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 20s margin
    
    # Seed points (5 points)
    # (0,0), (-56,33), (-16,63), (0,33), (16,63)
    base_points = [
        (Fraction(0), Fraction(0)),
        (Fraction(-56), Fraction(33)),
        (Fraction(-16), Fraction(63)),
        (Fraction(0), Fraction(33)),
        (Fraction(16), Fraction(63))
    ]
    
    best_points = base_points
    best_D = 1
    
    # Helper functions
    def is_square(frac):
        if frac < 0:
            return False
        if frac.denominator == 1:
            n = int(frac.numerator)
            r = int(math.isqrt(n))
            return r*r == n
        # Check if numerator and denominator are perfect squares
        n = frac.numerator
        d = frac.denominator
        rn = int(math.isqrt(n))
        rd = int(math.isqrt(d))
        return rn*rn == n and rd*rd == d

    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def is_rational_dist(p1, p2):
        d2 = dist_sq(p1, p2)
        return is_square(d2)

    def no_three_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    # Cross product (p2-p1) x (p3-p1)
                    dx1 = p2[0] - p1[0]
                    dy1 = p2[1] - p1[1]
                    dx2 = p3[0] - p1[0]
                    dy2 = p3[1] - p1[1]
                    if dx1*dy2 - dy1*dx2 == 0:
                        return False
        return True

    def no_four_concyclic(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                        # Check if 4 points are concyclic
                        # Determinant of the matrix:
                        # | x1 y1 x1^2+y1^2 1 |
                        # | x2 y2 x2^2+y2^2 1 | = 0
                        # | x3 y3 x3^2+y3^2 1 |
                        # | x4 y4 x4^2+y4^2 1 |
                        
                        # To avoid huge numbers, we can use the condition:
                        # (x1-x2)^2 + (y1-y2)^2 ... no, that's distances.
                        # The determinant condition is exact.
                        
                        # Let's compute the determinant.
                        # Since points are rational, we can use Fraction.
                        
                        # Optimization: Check if 3 points define a circle, and if the 4th is on it.
                        # Circle through p1, p2, p3:
                        # x^2 + y^2 + Ax + By + C = 0
                        # A, B, C can be found by solving linear system.
                        
                        # System:
                        # x1^2+y1^2 + Ax1 + By1 + C = 0
                        # x2^2+y2^2 + Ax2 + By2 + C = 0
                        # x3^2+y3^2 + Ax3 + By3 + C = 0
                        
                        # Subtract 1 from 2 and 3:
                        # A(x2-x1) + B(y2-y1) = -(x2^2+y2^2 - x1^2-y1^2)
                        # A(x3-x1) + B(y3-y1) = -(x3^2+y3^2 - x1^2-y1^2)
                        
                        # Solve for A, B.
                        dx21 = p2[0] - p1[0]
                        dy21 = p2[1] - p1[1]
                        dx31 = p3[0] - p1[0]
                        dy31 = p3[1] - p1[1]
                        
                        det = dx21*dy31 - dy21*dx31
                        if det == 0:
                            # Collinear, so no circle (or line). If collinear, they are not concyclic in the strict sense of a circle,
                            # but the problem says "no four concyclic". Collinear points are not concyclic.
                            # However, if 3 are collinear, the set is invalid anyway.
                            # We assume no 3 collinear is checked first.
                            continue

                        rhs2 = -( (p2[0]**2 + p2[1]**2) - (p1[0]**2 + p1[1]**2) )
                        rhs3 = -( (p3[0]**2 + p3[1]**2) - (p1[0]**2 + p1[1]**2) )
                        
                        # Cramer's rule
                        A_num = rhs2*dy31 - rhs3*dy21
                        B_num = dx21*rhs3 - dx31*rhs2
                        A = A_num / det
                        B = B_num / det
                        
                        # C = -(x1^2+y1^2 + Ax1 + By1)
                        C = -(p1[0]**2 + p1[1]**2 + A*p1[0] + B*p1[1])
                        
                        # Check p4
                        val = p4[0]**2 + p4[1]**2 + A*p4[0] + B*p4[1] + C
                        if val == 0:
                            return False
        return True

    def validate(points):
        if len(points) < 3:
            return False
        if not no_three_collinear(points):
            return False
        if not no_four_concyclic(points):
            return False
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if not is_rational_dist(points[i], points[j]):
                    return False
        return True

    # Check seed
    if validate(base_points):
        pass # Already valid
    
    # Try to find a 6th point
    # Use P0, P1, P2 to determine P5
    P0, P1, P2 = base_points[0], base_points[1], base_points[2]
    P3, P4 = base_points[3], base_points[4]
    
    # Linear equations:
    # 2*x*(P1x-P0x) + 2*y*(P1y-P0y) = r0^2 - r1^2 + P1^2 - P0^2
    # 2*x*(P2x-P0x) + 2*y*(P2y-P0y) = r0^2 - r2^2 + P2^2 - P0^2
    
    # Let u = r0^2, v = r1^2, w = r2^2
    # C1 = v - u + P1^2 - P0^2
    # C2 = w - u + P2^2 - P0^2
    
    # We iterate over small integer u, v, w
    # To keep numbers manageable, we search in a box
    # The coordinates of base points are around 50.
    # Distances should be around 50-100.
    # Squared distances 2500 - 10000.
    
    found = False
    # We can also try to vary the base points used for determination
    # But let's stick to 0,1,2 first.
    
    # Optimization: Instead of iterating u,v,w, we can iterate over x,y in a grid?
    # No, x,y must be rational.
    
    # Let's iterate over u, v, w in a range
    # Range: 0 to 50000 (squared distances)
    # This is too many (10^15).
    
    # Better: Iterate over u, and for each u, solve for x,y in terms of v,w?
    # Or iterate over v, w and solve for u?
    
    # Actually, we can parameterize the line segments.
    # If we fix u, the locus of points with distance sqrt(u) from P0 is a circle.
    # The intersection of 3 circles is a point.
    
    # Let's try a smarter search.
    # We know that if we find one 6th point, we can try to find a 7th.
    # But 6 is hard.
    
    # Let's try to find a 6th point by searching for rational x,y in a box
    # such that distances to P0..P4 are rational.
    # This is a system of 5 equations.
    # We can fix x,y to be integers in a small box and check?
    # If we find integer points, they are rational.
    
    # Search for integer points (x,y) in [-100, 100] x [-100, 100]
    # such that dist to all 5 base points are integers.
    
    candidates = []
    for x in range(-100, 101):
        for y in range(-100, 101):
            p_new = (Fraction(x), Fraction(y))
            # Check distinct
            if p_new in base_points:
                continue
            
            ok = True
            for p in base_points:
                d2 = dist_sq(p_new, p)
                if not is_square(d2):
                    ok = False
                    break
            if ok:
                # Check no 3 collinear with existing
                # And no 4 concyclic
                test_points = base_points + [p_new]
                if no_three_collinear(test_points) and no_four_concyclic(test_points):
                    candidates.append(p_new)
                    found = True
                    break
        if found:
            break
        if time.time() > deadline:
            break
    
    if found:
        best_points = base_points + [candidates[0]]
        
    # If we found 6, try to find 7?
    # Given time constraints, 6 is a good improvement.
    
    # Format output
    out_points = []
    for p in best_points:
        out_points.append([[p[0].numerator, p[0].denominator], [p[1].numerator, p[1].denominator]])
        
    return {'D': 1, 'points': out_points}

if __name__ == '__main__':
    construct()
