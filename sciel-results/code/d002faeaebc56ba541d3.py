import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit to be safe
    
    # Helper to check if a number is a perfect square
    def is_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    # Helper to check collinearity of 3 points
    # Points are (x1,y1), (x2,y2), (x3,y3)
    def is_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0

    # Helper to check concyclicity of 4 points
    # Points are (x1,y1), (x2,y2), (x3,y3), (x4,y4)
    # They are concyclic if the determinant of the following matrix is 0:
    # x^2+y^2  x  y  1
    def is_concyclic(p1, p2, p3, p4):
        pts = [p1, p2, p3, p4]
        # Check if any 3 are collinear, if so, they are trivially "concyclic" in a degenerate sense?
        # No, the problem says "no four concyclic". If 3 are collinear, a circle cannot pass through them unless it's a line (infinite radius), 
        # but standard definition of concyclic requires a finite circle. 
        # However, the constraint "no three collinear" is separate. 
        # If no three are collinear, we can check the circle equation.
        # Equation: x^2 + y^2 + Dx + Ey + F = 0
        # Subtract eq1 from eq2, eq3, eq4 to eliminate x^2+y^2
        # This leads to a linear system for D, E, F.
        # Alternatively, use the determinant.
        M = []
        for x, y in pts:
            M.append([x*x + y*y, x, y, 1])
        
        # Calculate determinant of 4x4 matrix
        # Using expansion or Gaussian elimination for exact integer arithmetic
        # Since numbers can be large, we use Fraction or just integer arithmetic if possible.
        # The values x, y are integers, so x^2+y^2 is integer.
        
        # Simple determinant calculation for 4x4
        def det4(m):
            # m is 4x4 list of lists
            # Using Leibniz formula or recursive
            # For performance and simplicity with small ints, we can use a standard algorithm
            # But let's use a robust method.
            # Actually, for integer coords, we can just check if the system has a unique solution for the circle.
            # 3 non-collinear points define a unique circle. The 4th is on it if it satisfies the equation.
            
            # Find circle through p1, p2, p3
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            
            # 2*det([x1 y1 1; x2 y2 1; x3 y3 1]) etc
            # General circle: x^2+y^2+Dx+Ey+F=0
            # (x2-x1)^2 + (y2-y1)^2 + D(x2-x1) + E(y2-y1) = 0
            # (x3-x1)^2 + (y3-y1)^2 + D(x3-x1) + E(y3-y1) = 0
            
            a1 = x2 - x1
            b1 = y2 - y1
            c1 = -(x2*x2 + y2*y2 - x1*x1 - y1*y1)
            
            a2 = x3 - x1
            b2 = y3 - y1
            c2 = -(x3*x3 + y3*y3 - x1*x1 - y1*y1)
            
            det = a1 * b2 - a2 * b1
            if det == 0:
                return True # Collinear, handled by other check
            
            D = (c1 * b2 - c2 * b1) / det
            E = (a1 * c2 - a2 * c1) / det
            
            # Check if p4 is on the circle
            # x4^2 + y4^2 + D*x4 + E*y4 + F = 0
            # F = -x1^2 - y1^2 - D*x1 - E*y1
            # LHS = x4^2 + y4^2 + D*x4 + E*y4 - x1^2 - y1^2 - D*x1 - E*y1
            # = (x4^2 - x1^2) + (y4^2 - y1^2) + D(x4 - x1) + E(y4 - y4)
            
            # To avoid floating point, check:
            # det * (x4^2 + y4^2 + D_int*x4 + E_int*y4 + F_int) == 0
            # Where D = D_int/det, E = E_int/det
            # D_int = c1*b2 - c2*b1
            # E_int = a1*c2 - a2*c1
            # F_int = -(x1*x1 + y1*y1)*det - D_int*x1 - E_int*y1
            
            D_int = c1 * b2 - c2 * b1
            E_int = a1 * c2 - a2 * c1
            F_int = -(x1*x1 + y1*y1) * det - D_int * x1 - E_int * y1
            
            val = (x4*x4 + y4*y4) * det + D_int * x4 + E_int * y4 + F_int
            return val == 0

    def is_valid_set(points):
        n = len(points)
        # Check distinctness
        if len(set(points)) != n:
            return False
        # Check distances
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + dy*dy
                if not is_square(d2):
                    return False
        # Check collinearity
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        return False
        # Check concyclicity
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if is_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    # Initial seed points
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best_set = seed_points[:]
    
    # Try to extend the set
    current_set = seed_points[:]
    
    # Search space for new points
    # We look for integer points
    # Limit search range to keep it fast
    search_range = 200
    
    # Pre-calculate squares for speed?
    # is_square is fast enough for small numbers.
    
    # Greedy extension
    extended = False
    while time.time() < deadline - 5.0:
        extended = False
        for x in range(-search_range, search_range + 1):
            if time.time() > deadline - 5.0:
                break
            for y in range(-search_range, search_range + 1):
                if time.time() > deadline - 5.0:
                    break
                new_point = (x, y)
                if new_point in current_set:
                    continue
                
                # Check if new point can be added
                # 1. Distances to all existing points
                valid_dist = True
                for p in current_set:
                    dx = x - p[0]
                    dy = y - p[1]
                    d2 = dx*dx + dy*dy
                    if not is_square(d2):
                        valid_dist = False
                        break
                
                if not valid_dist:
                    continue
                
                # 2. Collinearity with any pair
                valid_col = True
                for i in range(len(current_set)):
                    for j in range(i+1, len(current_set)):
                        if is_collinear(current_set[i], current_set[j], new_point):
                            valid_col = False
                            break
                    if not valid_col:
                        break
                
                if not valid_col:
                    continue
                
                # 3. Concyclicity with any triple
                valid_conc = True
                n_cur = len(current_set)
                for i in range(n_cur):
                    for j in range(i+1, n_cur):
                        for k in range(j+1, n_cur):
                            if is_concyclic(current_set[i], current_set[j], current_set[k], new_point):
                                valid_conc = False
                                break
                        if not valid_conc:
                            break
                    if not valid_conc:
                        break
                
                if not valid_conc:
                    continue
                
                # If we reach here, the point is valid
                current_set.append(new_point)
                extended = True
                break # Restart search with new set
                
            if extended:
                break
        
        if not extended:
            # No more points can be added to this specific configuration
            # Try a different strategy? Or accept this set.
            # For now, accept.
            break

    if len(current_set) > len(best_set):
        best_set = current_set[:]

    # Format output
    # D=1
    points_out = []
    for x, y in best_set:
        points_out.append([[[x, 1], [y, 1]]])
        
    return {'D': 1, 'points': points_out}
# EVOLVE-BLOCK-END
