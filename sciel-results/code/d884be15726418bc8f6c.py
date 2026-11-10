import time
from math import isqrt
from fractions import Fraction
import sys

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin before 120s kill

    def time_ok():
        return time.time() - t0 < DEADLINE

    def is_square(n):
        """Check if non-negative integer n is a perfect square."""
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n

    def check_no_three_collinear(pts):
        """pts: list of (x_num, x_den, y_num, y_den). No three collinear."""
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1 = Fraction(pts[i][0], pts[i][1])
                    y1 = Fraction(pts[i][2], pts[i][3])
                    x2 = Fraction(pts[j][0], pts[j][1])
                    y2 = Fraction(pts[j][2], pts[j][3])
                    x3 = Fraction(pts[k][0], pts[k][1])
                    y3 = Fraction(pts[k][2], pts[k][3])
                    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                    if det == 0:
                        return False
        return True

    def check_no_four_concyclic(pts, D):
        """pts: list of (x_num, x_den, y_num, y_den). No four concyclic."""
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        x1 = Fraction(pts[i][0], pts[i][1])
                        y1 = Fraction(pts[i][2], pts[i][3])
                        x2 = Fraction(pts[j][0], pts[j][1])
                        y2 = Fraction(pts[j][2], pts[j][3])
                        x3 = Fraction(pts[k][0], pts[k][1])
                        y3 = Fraction(pts[k][2], pts[k][3])
                        x4 = Fraction(pts[l][0], pts[l][1])
                        y4 = Fraction(pts[l][2], pts[l][3])
                        
                        # Circle: x^2 + D*y^2 + ux + vy + w = 0
                        # For each point: ux + vy + w = -(x^2 + D*y^2)
                        # Solve 3x3 for u,v,w from first 3, check 4th
                        def rhs(px, py):
                            return -(px * px + D * py * py)
                        
                        # Matrix rows: [x, y, 1]
                        a1, b1, c1 = x1, y1, Fraction(1)
                        a2, b2, c2 = x2, y2, Fraction(1)
                        a3, b3, c3 = x3, y3, Fraction(1)
                        
                        r1, r2, r3 = rhs(x1, y1), rhs(x2, y2), rhs(x3, y3)
                        
                        # Cramer's rule for 3x3
                        det_main = (a1*(b2*c3 - b3*c2) - b1*(a2*c3 - a3*c2) + c1*(a2*b3 - a3*b2))
                        if det_main == 0:
                            continue
                        
                        det_u = (r1*(b2*c3 - b3*c2) - b1*(r2*c3 - r3*c2) + c1*(r2*b3 - r3*b2))
                        det_v = (a1*(r2*c3 - r3*c2) - r1*(a2*c3 - a3*c2) + c1*(a2*r3 - a3*r2))
                        det_w = (a1*(b2*r3 - b3*r2) - b1*(a2*r3 - a3*r2) + r1*(a2*b3 - a3*b2))
                        
                        u = det_u / det_main
                        v = det_v / det_main
                        w = det_w / det_main
                        
                        # Check 4th point
                        if u * x4 + v * y4 + w != rhs(x4, y4):
                            continue
                        else:
                            return False
        return True

    def validate(pts, D):
        """Full validation."""
        n = len(pts)
        if n < 3:
            return False
        # Check distinctness
        seen = set()
        for p in pts:
            key = (Fraction(p[0], p[1]), Fraction(p[2], p[3]))
            if key in seen:
                return False
            seen.add(key)
        # Check rational distances
        for i in range(n):
            for j in range(i + 1, n):
                dx = Fraction(pts[i][0], pts[i][1]) - Fraction(pts[j][0], pts[j][1])
                dy = Fraction(pts[i][2], pts[i][3]) - Fraction(pts[j][2], pts[j][3])
                d2 = dx * dx + D * dy * dy
                if d2 < 0:
                    return False
                if not (is_square(d2.numerator) and is_square(d2.denominator)):
                    return False
        if not check_no_three_collinear(pts):
            return False
        if not check_no_four_concyclic(pts, D):
            return False
        return True

    def to_output(pts, D):
        """Convert internal representation to output format."""
        out = []
        for (xn, xd, yn, yd) in pts:
            out.append([[xn, xd], [yn, yd]])
        return {'D': D, 'points': out}

    # Seed: 5 known points with D=1
    seed_pts = [(0, 1, 0, 1), (-56, 1, 33, 1), (-16, 1, 63, 1), (0, 1, 33, 1), (16, 1, 63, 1)]
    D = 1

    best_pts = seed_pts[:]
    best_D = D
    best_count = len(seed_pts)

    if not validate(best_pts, best_D):
        best_pts = [(0,1,0,1), (3,1,0,1), (0,1,4,1)]
        best_count = 3

    # Try to find a 6th point using algebraic constraints
    if time_ok() and len(best_pts) == 5:
        # The 5 seed points: (0,0), (-56,33), (-16,63), (0,33), (16,63)
        # For a 6th point (x,y):
        # x^2 + y^2 = a^2
        # (x+56)^2 + (y-33)^2 = b^2
        # (x+16)^2 + (y-63)^2 = c^2
        # x^2 + (y-33)^2 = d^2
        # (x-16)^2 + (y-63)^2 = e^2
        
        # Strategy: search over possible values of a, d (distances from origin)
        # a^2 - d^2 = 66y - 1089
        # a^2 - d^2 = (a-d)(a+d)
        
        # More efficient: for each candidate (x,y), check all 5 distances.
        # Expand search range to +/-500 and use early termination.
        
        found = False
        for x in range(-500, 501):
            if not time_ok():
                break
            for y in range(-500, 501):
                # Quick check: distance to (0,0) must be integer
                d0 = x*x + y*y
                if not is_square(d0):
                    continue
                # Distance to (-56, 33)
                d1 = (x+56)**2 + (y-33)**2
                if not is_square(d1):
                    continue
                # Distance to (-16, 63)
                d2 = (x+16)**2 + (y-63)**2
                if not is_square(d2):
                    continue
                # Distance to (0, 33)
                d3 = x*x + (y-33)**2
                if not is_square(d3):
                    continue
                # Distance to (16, 63)
                d4 = (x-16)**2 + (y-63)**2
                if not is_square(d4):
                    continue
                
                # All 5 distances are integers! Now check collinearity and concyclicity
                new_pts = best_pts[:] + [(x, 1, y, 1)]
                if check_no_three_collinear(new_pts) and check_no_four_concyclic(new_pts, 1):
                    best_pts = new_pts
                    best_count = 6
                    found = True
                    break
            if found or not time_ok():
                break
        
        # If we found 6 points, try to find a 7th
        if found and time_ok() and len(best_pts) == 6:
            for x in range(-500, 501):
                if not time_ok():
                    break
                for y in range(-500, 501):
                    valid = True
                    for p in best_pts:
                        dx = x - p[0]
                        dy = y - p[2]
                        d2 = dx*dx + dy*dy
                        if not is_square(d2):
                            valid = False
                            break
                    if not valid:
                        continue
                    new_pts = best_pts[:] + [(x, 1, y, 1)]
                    if check_no_three_collinear(new_pts) and check_no_four_concyclic(new_pts, 1):
                        best_pts = new_pts
                        best_count = 7
                        break
                if best_count >= 7 or not time_ok():
                    break

    # Try some D > 1 configurations
    if time_ok():
        # D=2: (0,0), (1,2): dist 3
        D_test = 2
        base_pts = [(0, 1, 0, 1), (1, 1, 2, 1)]
        
        # Find a 3rd point for D=2
        found3 = False
        for x in range(-100, 101):
            if not time_ok():
                break
            for y in range(-100, 101):
                if x == 0 and y == 0:
                    continue
                if x == 1 and y == 2:
                    continue
                d0 = x*x + D_test*y*y
                if not is_square(d0):
                    continue
                d1 = (x-1)**2 + D_test*(y-2)**2
                if not is_square(d1):
                    continue
                new_pts = base_pts + [(x, 1, y, 1)]
                if check_no_three_collinear(new_pts) and check_no_four_concyclic(new_pts, D_test):
                    if len(new_pts) > best_count:
                        best_pts = new_pts
                        best_D = D_test
                        best_count = len(new_pts)
                    found3 = True
                    break
            if found3 or not time_ok():
                break
        
        # If we found 3 points for D=2, try to add more
        if found3 and time_ok() and best_D == D_test:
            cur = best_pts[:]
            found_more = False
            for x in range(-100, 101):
                if not time_ok():
                    break
                for y in range(-100, 101):
                    valid = True
                    for p in cur:
                        dx = x - p[0]
                        dy = y - p[2]
                        d2 = dx*dx + D_test*dy*dy
                        if not is_square(d2):
                            valid = False
                            break
                    if not valid:
                        continue
                    new_pts = cur + [(x, 1, y, 1)]
                    if check_no_three_collinear(new_pts) and check_no_four_concyclic(new_pts, D_test):
                        if len(new_pts) > best_count:
                            best_pts = new_pts
                            best_D = D_test
                            best_count = len(new_pts)
                        cur = new_pts
                        found_more = True
                        break
                if found_more or not time_ok():
                    break
            if len(cur) > best_count:
                best_pts = cur[:]
                best_D = D_test
                best_count = len(cur)

    return to_output(best_pts, best_D)

# EVOLVE-BLOCK-END
