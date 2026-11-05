import math
import time

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    def is_perfect_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    def is_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def is_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Determinant of 4x4 matrix
        # | x^2+y^2  x  y  1 |
        # ...
        # Using Cramer's rule or direct determinant
        # For efficiency, we can use the property:
        # 4 points are concyclic iff the determinant is 0.
        
        # A faster check:
        # Circumcircle of p1, p2, p3.
        # Equation: x^2 + y^2 + Dx + Ey + F = 0
        # Solve for D, E, F.
        # (x1^2+y1^2) + D x1 + E y1 + F = 0
        # (x2^2+y2^2) + D x2 + E y2 + F = 0
        # (x3^2+y3^2) + D x3 + E y3 + F = 0
        
        # Subtract 1 from 2 and 3 to eliminate F:
        # D(x2-x1) + E(y2-y1) = x1^2+y1^2 - x2^2-y2^2
        # D(x3-x1) + E(y3-y1) = x1^2+y1^2 - x3^2-y3^2
        
        # Let A = x2-x1, B = y2-y1, C = x1^2+y1^2 - x2^2-y2^2
        # Let D = x3-x1, E = y3-y1, F = x1^2+y1^2 - x3^2-y3^2
        # A D_x + B E_y = C
        # D D_x + E E_y = F
        
        # Determinant of coeff matrix: A E - B D
        # If 0, p1,p2,p3 collinear (handled by is_collinear)
        
        det = A * E - B * D
        if det == 0:
            return True # If collinear, they are on a circle of infinite radius? 
            # But we check collinear separately. If collinear, they are not concyclic in the finite sense.
            # However, the problem says "no four concyclic". If 3 are collinear, the 4th cannot be on a circle with them 
            # unless the 4th is also on the line? No, a circle intersects a line at max 2 points.
            # So if 3 are collinear, no 4th point is concyclic with them (finite circle).
            # So is_concyclic should return False if 3 are collinear?
            # The determinant of the 4x4 matrix for 3 collinear points and a 4th is non-zero 
            # unless the 4th is also on the line?
            # Let's stick to the determinant for safety.
            
        # Actually, let's just use the determinant of the 4x4 matrix.
        # It's robust.
        
        def det4(m):
            # m is 4x4 list of lists
            return (m[0][0]*(m[1][1]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - 
                             m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] - m[1][3]*m[2][2]*m[3][1]) -
                    m[0][1]*(m[1][0]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - 
                             m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] - m[1][3]*m[2][2]*m[3][0]) +
                    m[0][2]*(m[1][0]*m[2][1]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - 
                             m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] - m[1][3]*m[2][1]*m[3][0]) -
                    m[0][3]*(m[1][0]*m[2][1]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - 
                             m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] - m[1][2]*m[2][1]*m[3][0]))
        
        rows = []
        for p in (p1, p2, p3, p4):
            x, y = p
            rows.append([x*x + y*y, x, y, 1])
        return det4(rows) == 0

    def is_valid_set(points):
        n = len(points)
        if len(set(points)) != n:
            return False
        # Check distances
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + dy*dy
                if not is_perfect_square(d2):
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

    # Known 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify base
    if not is_valid_set(base_points):
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in base_points]}

    best_set = base_points
    
    # Search for 6th point
    # Iterate over possible distances d1 from P1(0,0)
    # (x,y) must satisfy x^2+y^2 = d1^2
    # And (x-xi)^2 + (y-yi)^2 = di^2 for i=2..5
    
    # We can iterate d1.
    # Max d1: Let's try up to 2000.
    max_d1 = 2000
    
    found_6 = False
    for d1 in range(1, max_d1 + 1):
        if time.time() > deadline - 10:
            break
        d1_sq = d1 * d1
        # Find all (x,y) such that x^2+y^2 = d1_sq
        # Iterate x from -d1 to d1
        for x in range(-d1, d1 + 1):
            y_sq = d1_sq - x * x
            if y_sq < 0:
                continue
            y = int(math.isqrt(y_sq))
            if y * y != y_sq:
                continue
            
            # Candidates: (x, y) and (x, -y) if y != 0
            candidates = [(x, y)]
            if y != 0:
                candidates.append((x, -y))
                
            for p in candidates:
                # Check if p is in base_points
                if p in base_points:
                    continue
                
                # Check distances to P2..P5
                valid = True
                for i in range(1, 5):
                    dx = p[0] - base_points[i][0]
                    dy = p[1] - base_points[i][1]
                    d2 = dx*dx + dy*dy
                    if not is_perfect_square(d2):
                        valid = False
                        break
                
                if valid:
                    candidate_set = base_points + [p]
                    if is_valid_set(candidate_set):
                        best_set = candidate_set
                        found_6 = True
                        break
            if found_6:
                break
        if found_6:
            break
    
    # If found 6, try to find 7
    if found_6:
        # The 6th point is best_set[5]
        # We need to find a 7th point at integer distance from all 6
        # This is harder. We can do a similar search relative to best_set[0]
        # But the constraints are tighter.
        # Let's try a random search in a box around the 6 points
        # Or iterate d1 again, but check against all 6.
        
        # For efficiency, let's just do a small random search or a limited box search
        # because the space of solutions is likely empty or very sparse.
        
        # Let's try to find a 7th point by iterating d1 from 1 to 500
        # and checking against all 6 points.
        max_d1_7 = 500
        found_7 = False
        for d1 in range(1, max_d1_7 + 1):
            if time.time() > deadline - 5:
                break
            d1_sq = d1 * d1
            for x in range(-d1, d1 + 1):
                y_sq = d1_sq - x * x
                if y_sq < 0:
                    continue
                y = int(math.isqrt(y_sq))
                if y * y != y_sq:
                    continue
                candidates = [(x, y)]
                if y != 0:
                    candidates.append((x, -y))
                for p in candidates:
                    if p in best_set:
                        continue
                    valid = True
                    for i in range(1, 6): # Check against P2..P6
                        dx = p[0] - best_set[i][0]
                        dy = p[1] - best_set[i][1]
                        d2 = dx*dx + dy*dy
                        if not is_perfect_square(d2):
                            valid = False
                            break
                    if valid:
                        candidate_set = best_set + [p]
                        if is_valid_set(candidate_set):
                            best_set = candidate_set
                            found_7 = True
                            break
                if found_7:
                    break
            if found_7:
                break
        
        # If found 7, try 8
        if found_7:
            max_d1_8 = 200
            found_8 = False
            for d1 in range(1, max_d1_8 + 1):
                if time.time() > deadline - 2:
                    break
                d1_sq = d1 * d1
                for x in range(-d1, d1 + 1):
                    y_sq = d1_sq - x * x
                    if y_sq < 0:
                        continue
                    y = int(math.isqrt(y_sq))
                    if y * y != y_sq:
                        continue
                    candidates = [(x, y)]
                    if y != 0:
                        candidates.append((x, -y))
                    for p in candidates:
                        if p in best_set:
                            continue
                        valid = True
                        for i in range(1, 7):
                            dx = p[0] - best_set[i][0]
                            dy = p[1] - best_set[i][1]
                            d2 = dx*dx + dy*dy
                            if not is_perfect_square(d2):
                                valid = False
                                break
                        if valid:
                            candidate_set = best_set + [p]
                            if is_valid_set(candidate_set):
                                best_set = candidate_set
                                found_8 = True
                                break
                    if found_8:
                        break
                if found_8:
                    break

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
