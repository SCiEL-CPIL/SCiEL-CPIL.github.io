import time
from math import isqrt
from fractions import Fraction

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
                            # Degenerate - points might be collinear, but we check that separately
                            # For concyclicity, if 3 points are collinear, they can't define a circle
                            # unless all 4 are collinear, which is already excluded
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
    # For D=1, integer coordinates, we need (x-xi)^2 + (y-yi)^2 = perfect square for all i
    # Use the seed points to derive constraints
    
    if time_ok() and len(best_pts) == 5:
        # The 5 seed points: (0,0), (-56,33), (-16,63), (0,33), (16,63)
        # Strategy: Iterate over y, find x such that distances to (0,0) and (0,33) are integer,
        # then check the other 3.
        
        found = False
        # y range can be expanded because we are not iterating x blindly
        for y in range(-2000, 2001):
            if not time_ok():
                break
            
            # Check if x^2 + y^2 is a square and x^2 + (y-33)^2 is a square
            # Let x^2 + y^2 = d0^2 and x^2 + (y-33)^2 = d3^2
            # d0^2 - d3^2 = y^2 - (y-33)^2 = 66y - 1089
            # This difference must be representable as difference of two squares.
            # Also x must be integer.
            
            # Efficiently find x:
            # x^2 = d0^2 - y^2.
            # We can iterate d0 from max(1, abs(y)) to some limit.
            # Limit d0 to 5000 (x^2 < 25e6 -> x < 5000)
            max_d0 = 5000
            for d0 in range(max(1, abs(y)), max_d0 + 1):
                d0_sq = d0 * d0
                x_sq = d0_sq - y * y
                if x_sq < 0:
                    break
                if not is_square(x_sq):
                    continue
                x = isqrt(x_sq)
                # Check x^2 + (y-33)^2 is square
                d3_sq = x_sq + (y - 33) * (y - 33)
                if not is_square(d3_sq):
                    continue
                
                # We have a candidate (x, y) with integer distances to (0,0) and (0,33).
                # Check distances to (-56,33), (-16,63), (16,63)
                d1_sq = (x + 56) * (x + 56) + (y - 33) * (y - 33)
                if not is_square(d1_sq):
                    continue
                d2_sq = (x + 16) * (x + 16) + (y - 63) * (y - 63)
                if not is_square(d2_sq):
                    continue
                d4_sq = (x - 16) * (x - 16) + (y - 63) * (y - 63)
                if not is_square(d4_sq):
                    continue
                
                # All 5 distances are integers!
                # Check distinctness
                if (x == 0 and y == 0) or (x == -56 and y == 33) or (x == -16 and y == 63) or (x == 0 and y == 33) or (x == 16 and y == 63):
                    continue
                
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
            # Use similar strategy for 7th point?
            # Iterate y, find x with integer distances to all 6.
            # This is much harder. Maybe just grid search with early termination.
            for x in range(-1000, 1001):
                if not time_ok():
                    break
                for y in range(-1000, 1001):
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
        # D=3: (0,0), (1,1) dist sqrt(1+3)=2, (0,0),(2,0) dist 2
        # (1,1) to (2,0): sqrt(1+3)=2. All good so far.
        # (0,0),(1,1),(2,0): check collinear: slope 0->1 is 1, slope 1->2 is -1. Not collinear.
        # Try to add (3,1): d to (0,0)=sqrt(9+3)=sqrt(12) no
        # Try (0,2): d to (0,0)=sqrt(12) no
        # Try (2,2): d to (0,0)=sqrt(4+12)=4 yes, d to (1,1)=sqrt(1+3)=2 yes, d to (2,0)=sqrt(0+12)=2*sqrt(3) no
        
        # D=5: (0,0), (1,2): sqrt(1+20)=sqrt(21) no
        # (0,0), (2,1): sqrt(4+5)=3 yes
        # (0,0), (2,1), (1,3): d(2,1 to 1,3)=sqrt(1+20)=sqrt(21) no
        # (0,0), (2,1), (4,2): collinear
        # (0,0), (2,1), (3,4): d(2,1 to 3,4)=sqrt(1+45)=sqrt(46) no
        # (0,0), (2,1), (1,2): d(2,1 to 1,2)=sqrt(1+5)=sqrt(6) no
        
        # D=2: (0,0), (1,2): sqrt(1+8)=3 yes
        # (0,0), (1,2), (7,2): d(1,2 to 7,2)=6 yes, d(0,0 to 7,2)=sqrt(49+8)=sqrt(57) no
        # (0,0), (1,2), (3,4): d(1,2 to 3,4)=sqrt(4+8)=sqrt(12) no
        # (0,0), (1,2), (5,4): d(1,2 to 5,4)=sqrt(16+8)=sqrt(24) no
        # (0,0), (1,2), (7,4): d(1,2 to 7,4)=sqrt(36+8)=sqrt(44) no
        # (0,0), (1,2), (9,2): d(1,2 to 9,2)=8 yes, d(0,0 to 9,2)=sqrt(81+8)=sqrt(89) no
        # (0,0), (1,2), (11,2): d(1,2 to 11,2)=10 yes, d(0,0 to 11,2)=sqrt(121+8)=sqrt(129) no
        
        # Try a small search for D=2
        if time_ok():
            D_test = 2
            base_pts = [(0, 1, 0, 1), (1, 1, 2, 1)]  # dist 3
            # Find more points at integer distance from both
            for x in range(-50, 51):
                if not time_ok():
                    break
                for y in range(-50, 51):
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
                        # We have 3 points for D=2. Try to add more.
                        cur = new_pts[:]
                        for x2 in range(-50, 51):
                            if not time_ok():
                                break
                            for y2 in range(-50, 51):
                                valid = True
                                for p in cur:
                                    dx = x2 - p[0]
                                    dy = y2 - p[2]
                                    d2 = dx*dx + D_test*dy*dy
                                    if not is_square(d2):
                                        valid = False
                                        break
                                if not valid:
                                    continue
                                new_pts2 = cur + [(x2, 1, y2, 1)]
                                if check_no_three_collinear(new_pts2) and check_no_four_concyclic(new_pts2, D_test):
                                    if len(new_pts2) > best_count:
                                        best_pts = new_pts2
                                        best_D = D_test
                                        best_count = len(new_pts2)
                                    cur = new_pts2
                                    break
                            if len(cur) > 3 or not time_ok():
                                break
                        if len(cur) > best_count:
                            best_pts = cur[:]
                            best_D = D_test
                            best_count = len(cur)
                        break

    return to_output(best_pts, best_D)

# EVOLVE-BLOCK-END
