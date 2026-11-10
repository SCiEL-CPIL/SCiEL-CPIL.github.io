import time
import math
import sys
from fractions import Fraction

def construct():
    t0 = time.time()
    DEADLINE = 118.0  # Safety margin before 120s kill

    def time_ok():
        return time.time() - t0 < DEADLINE

    def is_square(n):
        """Check if non-negative integer n is a perfect square."""
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def check_no_three_collinear_int(pts):
        """
        pts: list of (x, y) integers.
        Returns True if no three are collinear.
        """
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    # Determinant of matrix with rows (x, y, 1)
                    # (x2-x1)(y3-y1) - (x3-x1)(y2-y1) == 0
                    if (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1):
                        return False
        return True

    def check_no_four_concyclic_int(pts):
        """
        pts: list of (x, y) integers.
        Returns True if no four are concyclic.
        Circle equation: x^2 + y^2 + Ax + By + C = 0
        For 4 points to be concyclic, the determinant of the 4x4 matrix must be 0.
        | x^2+y^2  x  y  1 |
        """
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        x1, y1 = pts[i]
                        x2, y2 = pts[j]
                        x3, y3 = pts[k]
                        x4, y4 = pts[l]
                        
                        # Matrix rows:
                        # [x1^2+y1^2, x1, y1, 1]
                        # [x2^2+y2^2, x2, y2, 1]
                        # [x3^2+y3^2, x3, y3, 1]
                        # [x4^2+y4^2, x4, y4, 1]
                        
                        # We can solve for A, B, C using first 3 points and check 4th.
                        # x1^2 + y1^2 + A x1 + B y1 + C = 0
                        # Let R_i = x_i^2 + y_i^2
                        # A x1 + B y1 + C = -R1
                        # A x2 + B y2 + C = -R2
                        # A x3 + B y3 + C = -R3
                        
                        # Subtract eq1 from eq2 and eq3:
                        # A(x2-x1) + B(y2-y1) = -(R2-R1)
                        # A(x3-x1) + B(y3-y1) = -(R3-R1)
                        
                        dx21 = x2 - x1
                        dy21 = y2 - y1
                        dx31 = x3 - x1
                        dy31 = y3 - y1
                        
                        dR21 = R2 - R1
                        dR31 = R3 - R1
                        
                        # System:
                        # dx21 A + dy21 B = -dR21
                        # dx31 A + dy31 B = -dR31
                        
                        det = dx21 * dy31 - dx31 * dy21
                        if det == 0:
                            continue # Degenerate (collinear), handled elsewhere or implies no unique circle
                
                        # Cramer's rule
                        # A = ( (-dR21)*dy31 - (-dR31)*dy21 ) / det
                        # B = ( dx21*(-dR31) - dx31*(-dR21) ) / det
                        
                        # To avoid fractions, we check if the 4th point satisfies the equation derived from A, B, C.
                        # Instead of calculating A, B, C explicitly, we can check the 4x4 determinant directly.
                        # But 4x4 det is slower. Let's stick to the linear check with integer arithmetic.
                        
                        # A_num = -dR21 * dy31 + dR31 * dy21
                        # B_num = dx21 * (-dR31) + dx31 * dR21
                        # C_num = -R1 * det - x1 * A_num - y1 * B_num  (This is getting messy)
                        
                        # Alternative: Check if (x4, y4) lies on the circle defined by 1,2,3.
                        # The circle is defined by:
                        # (x^2+y^2) * det(x2,y2,1; x3,y3,1; x4,y4,1) ... no.
                        
                        # Let's just compute A, B, C as Fractions if needed, but for integer points,
                        # we can check:
                        # (x4-x1)*(y3-y1) - (x3-x1)*(y4-y1) ... no that's collinearity.
                        
                        # Concyclic condition:
                        # | x1^2+y1^2  x1  y1  1 |
                        # | x2^2+y2^2  x2  y2  1 | = 0
                        # | x3^2+y3^2  x3  y3  1 |
                        # | x4^2+y4^2  x4  y4  1 |
                        
                        # We can expand this determinant.
                        # For speed, let's use the property that if 1,2,3 define a unique circle,
                        # the power of point 4 with respect to this circle must be 0.
                        
                        # Let's use the linear system solution for A, B, C.
                        # A = (dR21*dy31 - dR31*dy21) / det  (signs might vary, let's be careful)
                        # Eq:
                        # dx21 A + dy21 B = -dR21
                        # dx31 A + dy31 B = -dR31
                        
                        # Multiply 1st by dy31, 2nd by dy21:
                        # dx21 dy31 A + dy21 dy31 B = -dR21 dy31
                        # dx31 dy21 A + dy31 dy21 B = -dR31 dy21
                        # Subtract:
                        # (dx21 dy31 - dx31 dy21) A = -dR21 dy31 + dR31 dy21
                        # det * A = dR31 dy21 - dR21 dy31
                        # A = (dR31 dy21 - dR21 dy31) / det
                        
                        # Similarly for B:
                        # Multiply 1st by dx31, 2nd by dx21:
                        # dx21 dx31 A + dy21 dx31 B = -dR21 dx31
                        # dx31 dx21 A + dy31 dx21 B = -dR31 dx21
                        # Subtract:
                        # (dy21 dx31 - dy31 dx21) B = -dR21 dx31 + dR31 dx21
                        # -det * B = dR31 dx21 - dR21 dx31
                        # B = (dR21 dx31 - dR31 dx21) / det
                        
                        # Now C:
                        # A x1 + B y1 + C = -R1
                        # C = -R1 - A x1 - B y1
                        # C = (-R1 * det - x1 * (dR31 dy21 - dR21 dy31) - y1 * (dR21 dx31 - dR31 dx21)) / det
                        
                        # Check if 4th point satisfies:
                        # A x4 + B y4 + C = -R4
                        # (A_num * x4 + B_num * y4 + C_num) / det = -R4
                        # A_num * x4 + B_num * y4 + C_num + R4 * det == 0
                        
                        A_num = dR31 * dy21 - dR21 * dy31
                        B_num = dR21 * dx31 - dR31 * dx21
                        C_num = -R1 * det - x1 * A_num - y1 * B_num
                        
                        if A_num * x4 + B_num * y4 + C_num + R4 * det != 0:
                            continue # Not concyclic
                        else:
                            return False # Found 4 concyclic points
        return True

    def validate_int(pts, D):
        """
        pts: list of (x, y) integers.
        D: integer.
        Checks:
        1. Distinctness
        2. Rational distances (dx^2 + D*dy^2 is square)
        3. No 3 collinear
        4. No 4 concyclic
        """
        n = len(pts)
        if n < 3:
            return False
        
        # Distinctness
        seen = set()
        for p in pts:
            if p in seen:
                return False
            seen.add(p)
            
        # Rational distances
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i+1, n):
                x2, y2 = pts[j]
                dx = x1 - x2
                dy = y1 - y2
                d2 = dx*dx + D*dy*dy
                if not is_square(d2):
                    return False
                    
        # Collinearity
        if not check_no_three_collinear_int(pts):
            return False
            
        # Concyclicity
        # Note: The concyclic check above assumes D=1 (Euclidean circle).
        # For D != 1, the "circle" is x^2 + D y^2 + Ax + By + C = 0.
        # The determinant check needs modification for D.
        # For now, we prioritize D=1. If we extend to D>1, we need a generalized check.
        if D == 1:
            if not check_no_four_concyclic_int(pts):
                return False
        else:
            # Generalized concyclic check for x^2 + D y^2 + Ax + By + C = 0
            # Matrix:
            # [x^2 + D y^2, x, y, 1]
            # We can use the same logic: Solve for A, B, C from 3 points, check 4th.
            # R_i = x_i^2 + D y_i^2
            # A x1 + B y1 + C = -R1
            # ...
            # The linear system for A, B, C is the same, but R_i changes.
            # So the logic in check_no_four_concyclic_int works if we pass R_i calculated with D.
            # Let's refactor check_no_four_concyclic_int to accept D.
            if not check_no_four_concyclic_general(pts, D):
                return False
        return True

    def check_no_four_concyclic_general(pts, D):
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        x1, y1 = pts[i]
                        x2, y2 = pts[j]
                        x3, y3 = pts[k]
                        x4, y4 = pts[l]
                        
                        R1 = x1*x1 + D*y1*y1
                        R2 = x2*x2 + D*y2*y2
                        R3 = x3*x3 + D*y3*y3
                        R4 = x4*x4 + D*y4*y4
                        
                        dx21 = x2 - x1
                        dy21 = y2 - y1
                        dx31 = x3 - x1
                        dy31 = y3 - y1
                        
                        dR21 = R2 - R1
                        dR31 = R3 - R1
                        
                        det = dx21 * dy31 - dx31 * dy21
                        if det == 0:
                            continue
                        
                        A_num = dR31 * dy21 - dR21 * dy31
                        B_num = dR21 * dx31 - dR31 * dx21
                        C_num = -R1 * det - x1 * A_num - y1 * B_num
                        
                        if A_num * x4 + B_num * y4 + C_num + R4 * det != 0:
                            continue
                        else:
                            return False
        return True

    # Known 5-point configuration for D=1
    seed_pts_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best_pts = seed_pts_5[:]
    best_D = 1
    best_count = 5

    if validate_int(best_pts, best_D):
        # Confirmed valid
        pass
    else:
        # Fallback to 3 points if seed is invalid (should not happen)
        best_pts = [(0,0), (3,0), (0,4)]
        best_count = 3

    # Search for 6th point for D=1
    if time_ok():
        base_pts = best_pts[:]
        # Search range
        # The points are roughly in [-56, 16] x [0, 63].
        # Try a wider range.
        search_range = 150
        found_6 = False
        
        for x in range(-search_range, search_range + 1):
            if not time_ok():
                break
            for y in range(-search_range, search_range + 1):
                # Quick check: distinct
                if (x, y) in base_pts:
                    continue
                
                # Check distances to all existing points
                valid_dist = True
                for (px, py) in base_pts:
                    dx = x - px
                    dy = y - py
                    d2 = dx*dx + dy*dy
                    if not is_square(d2):
                        valid_dist = False
                        break
                
                if not valid_dist:
                    continue
                
                new_pts = base_pts + [(x, y)]
                
                # Check collinearity
                if not check_no_three_collinear_int(new_pts):
                    continue
                
                # Check concyclicity
                if not check_no_four_concyclic_int(new_pts):
                    continue
                
                # Found a valid 6-point set!
                best_pts = new_pts
                best_count = 6
                found_6 = True
                break
            if found_6 or not time_ok():
                break
        
        # If 6 found, try for 7? (Unlikely to be fast, but let's try a small search)
        if found_6 and time_ok():
            base_pts = best_pts[:]
            found_7 = False
            # Smaller search range for 7th point
            for x in range(-100, 101):
                if not time_ok():
                    break
                for y in range(-100, 101):
                    if (x, y) in base_pts:
                        continue
                    valid_dist = True
                    for (px, py) in base_pts:
                        dx = x - px
                        dy = y - py
                        d2 = dx*dx + dy*dy
                        if not is_square(d2):
                            valid_dist = False
                            break
                    if not valid_dist:
                        continue
                    new_pts = base_pts + [(x, y)]
                    if check_no_three_collinear_int(new_pts) and check_no_four_concyclic_int(new_pts):
                        best_pts = new_pts
                        best_count = 7
                        found_7 = True
                        break
                if found_7 or not time_ok():
                    break

    # If D=1 didn't yield >5, try D=2, 3, 5... with smaller search
    if best_count <= 5 and time_ok():
        for D_test in [2, 3, 5, 6, 7, 10, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 37, 38, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 90, 93, 94, 95, 97]:
            if not time_ok():
                break
            
            # Start with (0,0)
            base = [(0, 0)]
            found_3 = False
            
            # Find 2 more points to make 3
            for x1 in range(-20, 21):
                if not time_ok():
                    break
                for y1 in range(-20, 21):
                    if x1 == 0 and y1 == 0:
                        continue
                    d0 = x1*x1 + D_test*y1*y1
                    if not is_square(d0):
                        continue
                    p1 = (x1, y1)
                    
                    for x2 in range(-20, 21):
                        if not time_ok():
                            break
                        for y2 in range(-20, 21):
                            if x2 == 0 and y2 == 0:
                                continue
                            if x2 == x1 and y2 == y1:
                                continue
                            d1 = x2*x2 + D_test*y2*y2
                            if not is_square(d1):
                                continue
                            dx = x2 - x1
                            dy = y2 - y1
                            d2 = dx*dx + D_test*dy*dy
                            if not is_square(d2):
                                continue
                            p2 = (x2, y2)
                            pts3 = base + [p1, p2]
                            
                            if check_no_three_collinear_int(pts3) and check_no_four_concyclic_general(pts3, D_test):
                                # Found 3 points! Try to add more
                                cur = pts3[:]
                                for _ in range(5):  # try to add up to 5 more
                                    if not time_ok():
                                        break
                                    added = False
                                    for x3 in range(-50, 51):
                                        if not time_ok():
                                            break
                                        for y3 in range(-50, 51):
                                            if (x3, y3) in cur:
                                                continue
                                            valid = True
                                            for p in cur:
                                                dx = x3 - p[0]
                                                dy = y3 - p[1]
                                                d2 = dx*dx + D_test*dy*dy
                                                if not is_square(d2):
                                                    valid = False
                                                    break
                                            if not valid:
                                                continue
                                            new_pts = cur + [(x3, y3)]
                                            if check_no_three_collinear_int(new_pts) and check_no_four_concyclic_general(new_pts, D_test):
                                                cur = new_pts
                                                added = True
                                                break
                                        if added or not time_ok():
                                            break
                                    if not added:
                                        break
                                if len(cur) > best_count:
                                    best_pts = cur[:]
                                    best_D = D_test
                                    best_count = len(cur)
                                found_3 = True
                                break
                        if found_3 or not time_ok():
                            break
                    if found_3 or not time_ok():
                        break
                if found_3 or not time_ok():
                    break
            if found_3:
                continue

    # Format output
    out_points = []
    for (x, y) in best_pts:
        out_points.append([[x, 1], [y, 1]])
        
    return {'D': best_D, 'points': out_points}

if __name__ == "__main__":
    construct()
