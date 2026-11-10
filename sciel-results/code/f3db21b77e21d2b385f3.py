import time
from math import isqrt
from fractions import Fraction
import sys
import random

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
        # Precompute Fractions
        xs = [Fraction(pts[i][0], pts[i][1]) for i in range(n)]
        ys = [Fraction(pts[i][2], pts[i][3]) for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    det = (xs[j] - xs[i]) * (ys[k] - ys[i]) - (xs[k] - xs[i]) * (ys[j] - ys[i])
                    if det == 0:
                        return False
        return True

    def check_no_four_concyclic(pts, D):
        """pts: list of (x_num, x_den, y_num, y_den). No four concyclic."""
        n = len(pts)
        # Precompute Fractions
        xs = [Fraction(pts[i][0], pts[i][1]) for i in range(n)]
        ys = [Fraction(pts[i][2], pts[i][3]) for i in range(n)]
        
        # Precompute x^2 + D*y^2
        # Note: D is an integer
        rhs = []
        for i in range(n):
            val = -(xs[i] * xs[i] + D * ys[i] * ys[i])
            rhs.append(val)

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        # Solve for u, v, w from points i, j, k
                        # u*x + v*y + w = rhs
                        a1, b1, c1 = xs[i], ys[i], Fraction(1)
                        a2, b2, c2 = xs[j], ys[j], Fraction(1)
                        a3, b3, c3 = xs[k], ys[k], Fraction(1)
                        
                        r1, r2, r3 = rhs[i], rhs[j], rhs[k]
                        
                        # Cramer's rule
                        det_main = (a1*(b2*c3 - b3*c2) - b1*(a2*c3 - a3*c2) + c1*(a2*b3 - a3*b2))
                        if det_main == 0:
                            continue
                        
                        # We don't need to compute u, v, w explicitly if we just check the 4th point
                        # The condition for point l to be on the circle defined by i,j,k is:
                        # | x_l y_l 1 |
                        # | x_i y_i 1 |  = 0  (with the augmented column being rhs? No, standard determinant)
                        # Actually, the determinant of the 4x4 matrix:
                        # | x^2+Dy^2  x  y  1 |
                        # | ...      ... ... ... |
                        # must be 0.
                        
                        # Let's use the 4x4 determinant directly, it's often faster or comparable and less error-prone with fractions if optimized.
                        # But 3x3 Cramer + check is also good. Let's stick to Cramer but optimize the check.
                        
                        # Check if u*x_l + v*y_l + w == rhs[l]
                        # u = det_u / det_main, etc.
                        # So det_u * x_l + det_v * y_l + det_w == rhs[l] * det_main
                        
                        det_u = (r1*(b2*c3 - b3*c2) - b1*(r2*c3 - r3*c2) + c1*(r2*b3 - r3*b2))
                        det_v = (a1*(r2*c3 - r3*c2) - r1*(a2*c3 - a3*c2) + c1*(a2*r3 - a3*r2))
                        det_w = (a1*(b2*r3 - b3*r2) - b1*(a2*r3 - a3*r2) + r1*(a2*b3 - a3*b2))
                        
                        lhs = det_u * xs[l] + det_v * ys[l] + det_w
                        rhs_check = rhs[l] * det_main
                        
                        if lhs == rhs_check:
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

    # Strategy: 
    # 1. Try to expand the D=1 seed set by searching for integer points that satisfy all constraints.
    #    The seed points are integers. We search for other integer points (x, y) such that 
    #    distances to all existing points are integers.
    # 2. Also try D > 1 with small integer coordinates.

    # Phase 1: Expand D=1 seed
    if time_ok() and len(best_pts) == 5:
        found = False
        # Search range: The seed points are in x: -56..16, y: 0..63.
        # Distances are integers. 
        # Let's try a wider range first, then narrow down if needed.
        # Using a heuristic: points likely to have integer distances to multiple points are often 
        # found in a bounded region relative to the seed.
        
        # Pre-compute distances to first few points for quick rejection?
        # The current loop is O(N^2) per candidate, with N=5. That's 10 checks.
        # Range: -1000 to 1000 is 4 million points. 10 checks * 4M = 40M ops. Might be slow in Python.
        # Let's optimize by checking the most restrictive points first.
        # Or use a smaller range first.
        
        # Let's try range -200 to 200 first.
        for x in range(-200, 201):
            if not time_ok():
                break
            for y in range(-200, 201):
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
            for x in range(-200, 201):
                if not time_ok():
                    break
                for y in range(-200, 201):
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

        # If we found 7 points, try to find an 8th
        if time_ok() and len(best_pts) == 7:
            for x in range(-200, 201):
                if not time_ok():
                    break
                for y in range(-200, 201):
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
                        best_count = 8
                        break
                if best_count >= 8 or not time_ok():
                    break

    # Phase 2: Try other D values
    # If we didn't get 8 points with D=1, or even if we did, we might find a larger set with D>1?
    # Unlikely to beat 8 if we found 8. But if we only found 5 or 6, maybe D>1 helps?
    # The task says "8 points would be new". So 8 is the target.
    
    # Let's also try a greedy construction for D=1 starting from scratch with a larger search space
    # if the seed expansion failed to go beyond 5.
    if time_ok() and best_count < 8:
        # Greedy search for D=1
        # Start with a random point or (0,0)
        # Try to build a set of points
        # This is a heuristic approach.
        
        # Let's try to find a 6th point with a wider range if we only have 5
        if best_count == 5:
             # The previous search was -200..200. Let's try -500..500 but with early exit.
             found = False
             for x in range(-500, 501):
                 if not time_ok():
                     break
                 for y in range(-500, 501):
                     d0 = x*x + y*y
                     if not is_square(d0):
                         continue
                     d1 = (x+56)**2 + (y-33)**2
                     if not is_square(d1):
                         continue
                     d2 = (x+16)**2 + (y-63)**2
                     if not is_square(d2):
                         continue
                     d3 = x*x + (y-33)**2
                     if not is_square(d3):
                         continue
                     d4 = (x-16)**2 + (y-63)**2
                     if not is_square(d4):
                         continue
                     
                     new_pts = best_pts[:] + [(x, 1, y, 1)]
                     if check_no_three_collinear(new_pts) and check_no_four_concyclic(new_pts, 1):
                         best_pts = new_pts
                         best_count = 6
                         found = True
                         break
                 if found or not time_ok():
                     break

    # Try D=2, 3, etc. with small integer grids
    if time_ok() and best_count < 8:
        for D_test in [2, 3, 5, 6, 7, 10, 13, 14, 15, 17]:
            if not time_ok():
                break
            
            # Try to find a set of 3, then 4, etc.
            # Start with (0,0) and (1,0) -> dist 1.
            # Or (0,0) and (1, 1/sqrt(D)) is not integer y.
            # We need integer x, y.
            # Dist^2 = dx^2 + D dy^2 = square.
            
            # Let's just do a brute force search for small D for up to 6 points?
            # This is expensive.
            
            # Better: Find 3 points first.
            # p1=(0,0), p2=(a,0) -> dist a.
            # p3=(x,y) -> x^2 + D y^2 = b^2, (x-a)^2 + D y^2 = c^2.
            
            # This is getting complex. Let's stick to the D=1 expansion which is most promising 
            # given the seed is already 5 points.
            # If we can't find a 6th point in D=1, it's unlikely D>1 will easily give 6+ points 
            # within the time limit without a more sophisticated search.
            
            # However, let's try a quick greedy for D=2 with small range.
            base = [(0,1,0,1)]
            # Find p2
            p2_found = False
            for x in range(1, 50):
                for y in range(0, 50):
                    d2 = x*x + D_test*y*y
                    if is_square(d2):
                        base.append((x,1,y,1))
                        p2_found = True
                        break
                if p2_found:
                    break
            
            if p2_found:
                # Find p3
                p3_found = False
                for x in range(-50, 51):
                    if not time_ok(): break
                    for y in range(-50, 51):
                        if x==0 and y==0: continue
                        if x==base[1][0] and y==base[1][2]: continue
                        valid = True
                        for p in base:
                            dx = x - p[0]
                            dy = y - p[2]
                            d2 = dx*dx + D_test*dy*dy
                            if not is_square(d2):
                                valid = False
                                break
                        if not valid: continue
                        new_pts = base[:] + [(x,1,y,1)]
                        if check_no_three_collinear(new_pts) and check_no_four_concyclic(new_pts, D_test):
                            base = new_pts
                            p3_found = True
                            break
                    if p3_found or not time_ok(): break
                
                if p3_found and len(base) > best_count:
                    best_pts = base[:]
                    best_D = D_test
                    best_count = len(base)
                    
                    # Try to add p4
                    p4_found = False
                    for x in range(-50, 51):
                        if not time_ok(): break
                        for y in range(-50, 51):
                            valid = True
                            for p in base:
                                dx = x - p[0]
                                dy = y - p[2]
                                d2 = dx*dx + D_test*dy*dy
                                if not is_square(d2):
                                    valid = False
                                    break
                            if not valid: continue
                            new_pts = base[:] + [(x,1,y,1)]
                            if check_no_three_collinear(new_pts) and check_no_four_concyclic(new_pts, D_test):
                                base = new_pts
                                p4_found = True
                                break
                        if p4_found or not time_ok(): break
                            
                    if len(base) > best_count:
                        best_pts = base[:]
                        best_D = D_test
                        best_count = len(base)

    return to_output(best_pts, best_D)

# EVOLVE-BLOCK-END
