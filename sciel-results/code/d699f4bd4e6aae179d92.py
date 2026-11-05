"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def check_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def _concyclic_det(p1, p2, p3, p4):
        """Exact 4x4 determinant for concyclic check using integer arithmetic."""
        # |x  y  x^2+y^2  1|
        rows = []
        for p in [p1, p2, p3, p4]:
            x, y = p
            rows.append([x, y, x*x + y*y, 1])
        
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
                   - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
                   + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]))
        
        def det4(m):
            result = 0
            signs = [1, -1, 1, -1]
            for j in range(4):
                minor = [[m[i][k] for k in range(4) if k != j] for i in range(1, 4)]
                result += signs[j] * m[0][j] * det3(minor)
            return result
        
        return det4(rows)

    def all_constraints_satisfied(points):
        """Check all pairwise distances are integer, no 3 collinear, no 4 concyclic."""
        n = len(points)
        # Check all pairwise distances are integers
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                d2 = dx*dx + dy*dy
                if not is_perfect_square(d2):
                    return False
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        # Check no four concyclic
        for combo in combinations(range(n), 4):
            if _concyclic_det(points[combo[0]], points[combo[1]], 
                            points[combo[2]], points[combo[3]]) == 0:
                return False
        return True

    def find_next_point(points, max_radius=100):
        """
        Find a new integer point that is at integer distance from at least 2 existing points
        and satisfies all constraints with the current set.
        """
        n = len(points)
        # Try all pairs of existing points
        for i in range(n):
            for j in range(i+1, n):
                A = points[i]
                B = points[j]
                xA, yA = A
                xB, yB = B
                dAB2 = (xA - xB)**2 + (yA - yB)**2
                dAB = math.isqrt(dAB2)
                if dAB * dAB != dAB2:
                    continue # Should not happen for valid sets
                
                # Iterate possible radii
                # We can limit radii based on the bounding box of existing points + margin
                max_r = max_radius
                for rA in range(1, max_r + 1):
                    for rB in range(1, max_r + 1):
                        # Triangle inequality
                        if abs(rA - rB) >= dAB or rA + rB <= dAB:
                            continue
                        
                        # Calculate intersection of two circles
                        # (x - xA)^2 + (y - yA)^2 = rA^2
                        # (x - xB)^2 + (y - yB)^2 = rB^2
                        # Subtract equations:
                        # 2(xB - xA)x + 2(yB - yA)y = rA^2 - rB^2 + xA^2 - xB^2 + yA^2 - yB^2
                        # Let a = 2(xB - xA), b = 2(yB - yA), c = rA^2 - rB^2 + xA^2 - xB^2 + yA^2 - yB^2
                        a = 2 * (xB - xA)
                        b = 2 * (yB - yA)
                        c = rA*rA - rB*rB + xA*xA - xB*xB + yA*yA - yB*yB
                        
                        # The line is ax + by = c
                        # We need integer solutions for x, y on this line that also satisfy the circle equation.
                        # Alternatively, use the standard intersection formula.
                        # d = dAB
                        # a_val = (rA^2 - rB^2 + d^2) / (2d)
                        # h = sqrt(rA^2 - a_val^2)
                        # Midpoint M = ((xA+xB)/2, (yA+yB)/2)
                        # Vector AB = (xB-xA, yB-yA)
                        # Perpendicular vector = (-(yB-yA), xB-xA)
                        # Unit perp = (-(yB-yA)/d, (xB-xA)/d)
                        # P = M + a_val * UnitAB +/- h * UnitPerp
                        
                        # To keep exact arithmetic, we work with fractions or integers.
                        # Let's use fractions for precision.
                        d = dAB
                        # a_val = (rA^2 - rB^2 + d^2) / (2d)
                        num_a = rA*rA - rB*rB + d*d
                        den_a = 2 * d
                        # a_val is rational.
                        # h^2 = rA^2 - a_val^2
                        # h^2 = rA^2 - (num_a/den_a)^2 = (rA^2 * den_a^2 - num_a^2) / den_a^2
                        num_h2 = rA*rA * den_a*den_a - num_a*num_a
                        den_h2 = den_a*den_a
                        
                        if num_h2 < 0:
                            continue
                        
                        # Check if h is rational (sqrt(num_h2) must be integer if den_h2 is a square, etc.)
                        # Actually, we just need the final coordinates to be integers.
                        # x = (xA + xB)/2 + a_val * (xB - xA) / d +/- h * (-(yB - yA)) / d
                        # This is getting messy with fractions.
                        
                        # Simpler approach:
                        # We know x, y are integers.
                        # From ax + by = c, we can iterate x if gcd(a,b) divides c.
                        # But a,b can be large.
                        
                        # Let's just solve for x in terms of y or vice versa?
                        # Or just check if the intersection points are integers by using the formula with Fractions.
                        
                        # Let's use the property that if A, B, rA, rB are integers,
                        # the intersection points are rational.
                        # We check if they are integers.
                        
                        # Formula:
                        # x = (xA + xB)/2 + (rA^2 - rB^2 + d^2) / (2d) * (xB - xA) / d ... no
                        # Let's use the standard derivation.
                        # Let u = xB - xA, v = yB - yA.
                        # d^2 = u^2 + v^2.
                        # a = (rA^2 - rB^2 + d^2) / (2d)
                        # h^2 = rA^2 - a^2
                        # If h^2 is not a perfect square rational, no rational point?
                        # No, h can be irrational, but the projection might cancel it?
                        # No, for integer coordinates, the area of the triangle must be rational (it is, as coordinates are integer).
                        # The height h must be rational.
                        
                        # Let's just calculate the potential x coordinates.
                        # The x-coordinates of the intersection points are the roots of a quadratic?
                        # Or we can just iterate x in a small range around the midpoint?
                        # Midpoint x_m = (xA+xB)/2.
                        # The distance from midpoint to intersection in x-direction is bounded by d/2.
                        # So x is in [min(xA,xB)-d/2, max(xA,xB)+d/2].
                        # This range is small (<= d).
                        # We can iterate x in this range, solve for y from the line equation, and check if it's an integer and on the circle.
                        
                        x_min = min(xA, xB) - d
                        x_max = max(xA, xB) + d
                        
                        # Line: a*x + b*y = c
                        # If b != 0, y = (c - a*x) / b
                        # If b == 0, a*x = c. If a!=0, x = c/a.
                        
                        if b != 0:
                            for x in range(x_min, x_max + 1):
                                # Check if a*x + b*y = c has integer y
                                rem = c - a*x
                                if rem % b != 0:
                                    continue
                                y = rem // b
                                
                                # Check if (x,y) is on circle A
                                if (x - xA)**2 + (y - yA)**2 == rA*rA:
                                    # Check if point is new
                                    new_pt = (x, y)
                                    if new_pt in points:
                                        continue
                                    
                                    # Check constraints with all other points
                                    new_set = points + [new_pt]
                                    if all_constraints_satisfied(new_set):
                                        return new_pt
                        else:
                            # b == 0
                            if a != 0:
                                if c % a != 0:
                                    continue
                                x = c // a
                                # y is not constrained by the line, so we must check circle equation for y
                                # (x - xA)^2 + (y - yA)^2 = rA^2
                                # (y - yA)^2 = rA^2 - (x - xA)^2
                                dy2 = rA*rA - (x - xA)**2
                                if dy2 < 0:
                                    continue
                                rdy = math.isqrt(dy2)
                                if rdy*rdy != dy2:
                                    continue
                                # y = yA +/- rdy
                                for sign in [1, -1]:
                                    y = yA + sign * rdy
                                    new_pt = (x, y)
                                    if new_pt in points:
                                        continue
                                    new_set = points + [new_pt]
                                    if all_constraints_satisfied(new_set):
                                        return new_pt
                            else:
                                # a=0, b=0 implies A=B, impossible
                                pass
        return None

    def local_search(points, max_iter=200):
        pts = list(points)
        for _ in range(max_iter):
            if elapsed() > DEADLINE:
                break
            new_pt = find_next_point(pts, max_radius=150)
            if new_pt is not None:
                pts.append(new_pt)
                if len(pts) >= 8:
                    break
            else:
                # If we can't add, maybe we should swap?
                # For now, just stop.
                break
        return pts

    # Start with the valid 5-point set
    # (0,0), (-56,33), (-16,63), (0,33), (16,63)
    start_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify start points
    if not all_constraints_satisfied(start_points):
        # Fallback to a simpler 3-point set if needed
        start_points = [(0,0), (3,4), (4,0)]

    best_points = local_search(start_points)
    
    # If we didn't get 7, try a different start or more search
    if len(best_points) < 7 and elapsed() < DEADLINE - 10:
        # Try another known 5-point set or random restarts
        # For now, just return what we have
        pass

    # Convert to required format with D=1
    D = 1
    points_out = [[[x, 1], [y, 1]] for x, y in best_points]

    return {'D': D, 'points': points_out}
# EVOLVE-BLOCK-END
