import time
from math import isqrt, gcd
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def time_ok():
        return time.time() - t0 < DEADLINE

    def is_square(n):
        """Check if non-negative integer n is a perfect square."""
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n

    def is_rational_square(frac):
        """Check if Fraction is a perfect square of a rational."""
        if frac < 0:
            return False
        # frac = p/q in lowest terms; need p and q to be perfect squares
        return is_square(frac.numerator) and is_square(frac.denominator)

    def check_rational_distances(pts, D):
        """pts: list of (x_num, x_den, y_num, y_den). Check all pairwise distances rational."""
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                dx = Fraction(pts[i][0], pts[i][1]) - Fraction(pts[j][0], pts[j][1])
                dy = Fraction(pts[i][2], pts[i][3]) - Fraction(pts[j][2], pts[j][3])
                d2 = dx * dx + D * dy * dy
                if not is_rational_square(d2):
                    return False
        return True

    def check_no_three_collinear(pts):
        """pts: list of (x_num, x_den, y_num, y_den). No three collinear."""
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    # Cross product of (p_j - p_i) and (p_k - p_i)
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

    def check_no_four_concyclic(pts):
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
                        # Four points concyclic iff the determinant is zero:
                        # |x^2+y^2  x  y  1| for each point
                        def circ_det(px, py):
                            return (px * px + py * py, px, py, 1)
                        rows = [circ_det(x1, y1), circ_det(x2, y2), circ_det(x3, y3), circ_det(x4, y4)]
                        # Compute 4x4 determinant
                        # Use expansion
                        a, b, c, d = rows[0]
                        e, f, g, h = rows[1]
                        i2, j2, k2, l2 = rows[2]
                        m, n2, o, p = rows[3]
                        det = (a*(f*(k2*p - l2*o) - g*(j2*p - l2*n2) + h*(j2*o - k2*n2))
                               - b*(e*(k2*p - l2*o) - g*(i2*p - l2*n2) + h*(i2*o - k2*n2))
                               + c*(e*(j2*p - l2*n2) - f*(i2*p - l2*n2) + h*(i2*n2 - j2*n2))
                               - d*(e*(j2*o - k2*n2) - f*(i2*o - k2*n2) + g*(i2*n2 - j2*n2)))
                        if det == 0:
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
        if not check_rational_distances(pts, D):
            return False
        if not check_no_three_collinear(pts):
            return False
        if not check_no_four_concyclic(pts):
            return False
        return True

    def to_output(pts, D):
        """Convert internal representation to output format."""
        out = []
        for (xn, xd, yn, yd) in pts:
            out.append([[xn, xd], [yn, yd]])
        return {'D': D, 'points': out}

    # Start with seed: 5 points
    seed_pts = [(0, 1, 0, 1), (-56, 1, 33, 1), (-16, 1, 63, 1), (0, 1, 33, 1), (16, 1, 63, 1)]
    D = 1

    best_pts = seed_pts[:]
    best_D = D
    best_count = len(seed_pts)

    if not validate(best_pts, best_D):
        # Should not happen, but fallback
        best_pts = [(0,1,0,1), (3,1,0,1), (0,1,4,1)]
        best_count = 3

    # Try to add points to the seed set
    # Search strategy: try integer lattice points in a growing box
    # For D=1, points are (x, y) with integer coords, distance = sqrt((dx)^2 + (dy)^2)
    
    def try_add_point(current_pts, D, max_coord):
        """Try to add one more point to current_pts. Returns new point or None."""
        n = len(current_pts)
        # Try integer points in a box around the existing points
        # Compute bounding box
        xs = [p[0] // p[1] for p in current_pts]
        ys = [p[2] // p[3] for p in current_pts]
        min_x = min(xs) - max_coord
        max_x = max(xs) + max_coord
        min_y = min(ys) - max_coord
        max_y = max(ys) + max_coord
        
        # To keep search tractable, limit the box
        # For n points, the new point must satisfy rational distance to all n existing points
        # This is a system of n equations
        
        # Try a subset of candidate points
        # For efficiency, only try points that are at rational distance from the first few points
        for x in range(min_x, max_x + 1):
            if not time_ok():
                return None
            for y in range(min_y, max_y + 1):
                # Quick check: distance to first point must be rational
                for i in range(n):
                    dx = x - current_pts[i][0] // current_pts[i][1]
                    dy = y - current_pts[i][2] // current_pts[i][3]
                    d2 = dx * dx + D * dy * dy
                    if not is_square(d2):
                        break
                else:
                    # Passed all distance checks, now check collinearity and concyclicity
                    new_pts = current_pts[:] + [(x, 1, y, 1)]
                    if check_no_three_collinear(new_pts) and check_no_four_concyclic(new_pts):
                        return (x, 1, y, 1)
        return None

    # Iteratively try to add points
    current_pts = best_pts[:]
    current_D = D
    for attempt in range(20):
        if not time_ok():
            break
        # Try increasing search radius
        added = False
        for radius in [10, 20, 30, 50, 80, 120]:
            if not time_ok():
                break
            new_pt = try_add_point(current_pts, current_D, radius)
            if new_pt is not None:
                current_pts.append(new_pt)
                best_pts = current_pts[:]
                best_D = current_D
                best_count = len(current_pts)
                added = True
                break
        if not added:
            break

    # Also try D > 1 with rational coordinates
    # For D > 1, we need (x_i - x_j)^2 + D*(y_i - y_j)^2 to be a rational square
    # Try small D values and rational y coordinates
    
    for D_test in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15]:
        if not time_ok():
            break
        # Start with a small set for D_test
        # Try to build up from scratch
        # For D=2: need dx^2 + 2*dy^2 = rational square
        # (0,0), (1,1): 1+2=3 not square
        # (0,0), (2,1): 4+2=6 not square
        # (0,0), (1,0): 1 square. (0,0),(1,0),(0,1): 1, 2, 1+2=3 no
        # Let me just try a few known configurations
        
        # For D=2: (0,0), (1,0) dist=1. (0,0),(2,0) dist=2.
        # (0,0), (1,0), (2,0) - collinear!
        # (0,0), (1,0), (0,1): d(0,1 to 1,0) = sqrt(1+2)=sqrt(3) no
        # (0,0), (3,0), (0,1): d(3,0 to 0,1)=sqrt(9+2)=sqrt(11) no
        # (0,0), (1,1): sqrt(1+2)=sqrt(3) no
        # (0,0), (2,2): sqrt(4+8)=sqrt(12) no
        # (0,0), (1,2): sqrt(1+8)=3 yes!
        # So (0,0), (1,2): dist 3. (0,0), (2,1): sqrt(4+2)=sqrt(6) no
        # (0,0), (1,2), (3,1): d(1,2 to 3,1)=sqrt(4+2)=sqrt(6) no
        # (0,0), (1,2), (2,4): collinear
        # (0,0), (1,2), (4,1): d(1,2 to 4,1)=sqrt(9+2)=sqrt(11) no
        # (0,0), (1,2), (3,4): d(1,2 to 3,4)=sqrt(4+8)=sqrt(12) no
        # (0,0), (1,2), (5,1): d(1,2 to 5,1)=sqrt(16+2)=sqrt(18) no
        # (0,0), (1,2), (7,2): d(1,2 to 7,2)=6 yes! But (0,0),(1,2),(7,2) - check collinear:
        # slope 1->2 is 2/1=2, slope 7->2 is 2/7, not collinear. Good.
        # d(0,0 to 7,2)=sqrt(49+8)=sqrt(57) no
        
        # This is getting complex. Let me try a more systematic approach for D>1
        # but with limited search due to time constraints.
        
        # For now, focus on D=1 which we can search more efficiently
        # and also try some specific D>1 configurations

    # Try some specific D>1 configurations
    # D=2: (0,0), (1,2), (7,2) - need to verify all distances
    # d(0,0 to 1,2) = sqrt(1+8) = 3 ✓
    # d(0,0 to 7,2) = sqrt(49+8) = sqrt(57) ✗
    
    # D=3: (0,0), (1,1): sqrt(1+3)=2 ✓, (0,0),(2,0): 2 ✓
    # (0,0),(1,1),(2,0): d(1,1 to 2,0)=sqrt(1+3)=2 ✓
    # But (0,0),(1,1),(2,0): collinear? slope 0->1: 1, slope 1->2: -1. Not collinear.
    # Add (0,2): d(0,2 to 0,0)=sqrt(12)=2*sqrt(3) ✗
    # Add (3,1): d(3,1 to 0,0)=sqrt(9+3)=sqrt(12) ✗
    # Add (2,2): d(2,2 to 0,0)=sqrt(4+12)=4 ✓, d(2,2 to 1,1)=sqrt(1+3)=2 ✓, d(2,2 to 2,0)=sqrt(0+12)=2*sqrt(3) ✗
    
    # Let me try D=1 more aggressively with a smarter search
    # Use the fact that for D=1, we need Pythagorean-like conditions
    
    # Try to build larger sets from scratch using a different approach
    # Start with (0,0) and find points at rational distances
    
    if time_ok():
        # Try to find 6+ points with D=1 using a more directed search
        # Known: (0,0), (3,4) dist 5, (0,0),(5,12) dist 13
        # (3,4) to (5,12): sqrt(4+64)=sqrt(68) no
        # (0,0),(3,4),(6,8) collinear
        # (0,0),(3,4),(8,6): d(3,4 to 8,6)=sqrt(25+4)=sqrt(29) no
        # (0,0),(3,4),(15,8): d(3,4 to 15,8)=sqrt(144+16)=sqrt(160) no
        # (0,0),(3,4),(9,12): collinear
        # (0,0),(5,12),(12,5): d(5,12 to 12,5)=sqrt(49+49)=7*sqrt(2) no
        # (0,0),(5,12),(21,28): collinear
        
        # Let me try a computational search with the current best as starting point
        # but with a wider and more structured search
        
        # The current best is 5 points. Try to get 6.
        # Use the existing 5 points and search for a 6th more carefully
        
        if len(current_pts) < 6 and time_ok():
            # More systematic search for the 6th point
            # The 5 seed points are: (0,0), (-56,33), (-16,63), (0,33), (16,63)
            # For a new point (x,y) to have integer distance to all 5:
            # x^2 + y^2 = a^2
            # (x+56)^2 + (y-33)^2 = b^2
            # (x+16)^2 + (y-63)^2 = c^2
            # x^2 + (y-33)^2 = d^2
            # (x-16)^2 + (y-63)^2 = e^2
            
            # From eq1 and eq4: x^2+y^2 - x^2-(y-33)^2 = a^2-d^2
            # y^2 - (y-33)^2 = a^2-d^2
            # 66y - 1089 = a^2 - d^2
            # Similarly from eq2 and eq3:
            # (x+56)^2+(y-33)^2 - (x+16)^2-(y-63)^2 = b^2-c^2
            # (x+56)^2-(x+16)^2 + (y-33)^2-(y-63)^2 = b^2-c^2
            # (x^2+112x+3136)-(x^2+32x+256) + (y^2-66y+1089)-(y^2-126y+3969) = b^2-c^2
            # 80x+2880 + 60y-2880 = b^2-c^2
            # 80x+60y = b^2-c^2
            
            # This is complex. Let me just do a bounded search with early termination.
            # Search in a larger box but with smarter pruning.
            
            for x in range(-200, 201):
                if not time_ok():
                    break
                for y in range(-200, 201):
                    # Check distance to each of the 5 points
                    valid = True
                    for i in range(len(current_pts)):
                        dx = x - current_pts[i][0]
                        dy = y - current_pts[i][2]
                        d2 = dx * dx + dy * dy
                        if not is_square(d2):
                            valid = False
                            break
                    if not valid:
                        continue
                    # Check collinearity with all pairs
                    new_pts = current_pts[:] + [(x, 1, y, 1)]
                    if not check_no_three_collinear(new_pts):
                        continue
                    if not check_no_four_concyclic(new_pts):
                        continue
                    current_pts.append((x, 1, y, 1))
                    best_pts = current_pts[:]
                    best_D = 1
                    best_count = len(current_pts)
                    break
                if len(current_pts) > best_count or best_count >= 6:
                    if len(current_pts) >= 6:
                        break

    # Return the best result found
    return to_output(best_pts, best_D)

# EVOLVE-BLOCK-END
