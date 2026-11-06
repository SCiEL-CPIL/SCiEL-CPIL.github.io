import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 110  # seconds, safe margin from 120s limit (evaluator kills at 150s)
    
    def check_time():
        return time.time() - start_time < deadline
    
    def gcd(a, b):
        a, b = abs(a), abs(b)
        while b:
            a, b = b, a % b
        return a
    
    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n
    
    def dist_sq_rational(x1, y1, x2, y2, D):
        """Check if distance squared is a perfect square (integer coords in Q(sqrt(D)))"""
        dx = x1 - x2
        dy = y1 - y2
        d2 = dx*dx + D*dy*dy
        return is_square(d2)
    
    def collinear(x1, y1, x2, y2, x3, y3):
        """Check if three points (xi, yi*sqrt(D)) are collinear"""
        return (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) == 0
    
    def concyclic(x1, y1, x2, y2, x3, y3, x4, y4, D):
        """Check if four points are concyclic in Q(sqrt(D))"""
        a1 = x2 - x1
        b1 = y2 - y1
        c1 = (x1*x1 + D*y1*y1) - (x2*x2 + D*y2*y2)
        
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = (x1*x1 + D*y1*y1) - (x3*x3 + D*y3*y3)
        
        det = a1*b2 - a2*b1
        if det == 0:
            return True
        
        A_num = c1*b2 - c2*b1
        B_num = a1*c2 - a2*c1
        
        val = (det*(x4*x4 + D*y4*y4) 
               + A_num*x4 
               + B_num*y4 
               - (x1*x1 + D*y1*y1)*det 
               - A_num*x1 
               - B_num*y1)
        
        return val == 0
    
    def validate_set(points, D):
        """points: list of (x, y) integers, representing (x, y*sqrt(D))"""
        n = len(points)
        if len(set(points)) != n:
            return False
        for i in range(n):
            for j in range(i+1, n):
                if not dist_sq_rational(points[i][0], points[i][1], 
                                        points[j][0], points[j][1], D):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if collinear(points[i][0], points[i][1],
                                 points[j][0], points[j][1],
                                 points[k][0], points[k][1]):
                        return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if concyclic(points[i][0], points[i][1],
                                     points[j][0], points[j][1],
                                     points[k][0], points[k][1],
                                     points[l][0], points[l][1], D):
                            return False
        return True
    
    # Try D=1 (integer coordinates) first - simpler and known to have solutions
    # Search for 6-point rational distance sets in Z^2
    
    def search_integer_points(max_coord=30, target_n=6):
        """Search for target_n points with integer coordinates and rational distances"""
        if not check_time():
            return None
        
        # Precompute all valid displacement vectors
        valid_disps = []
        for dy in range(-max_coord, max_coord+1):
            for dx in range(-max_coord, max_coord+1):
                if dx == 0 and dy == 0:
                    continue
                d2 = dx*dx + dy*dy
                if is_square(d2):
                    valid_disps.append((dx, dy))
        
        # Sort by magnitude for heuristic
        valid_disps.sort(key=lambda p: (p[0]*p[0] + p[1]*p[1], p[0], p[1]))
        
        # Try to build a set starting from origin
        # Use backtracking with symmetry breaking
        
        best_found = None
        
        # Try different starting pairs to break symmetry
        for p1_idx in range(min(10, len(valid_disps))):
            if not check_time():
                return best_found
            
            p1 = (0, 0)
            dx1, dy1 = valid_disps[p1_idx]
            p2 = (dx1, dy1)
            
            current = [p1, p2]
            
            # Find all points that work with both p1 and p2
            candidates = []
            for dx, dy in valid_disps:
                if not check_time():
                    break
                p3 = (dx, dy)
                if p3 in current:
                    continue
                if abs(p3[0]) > max_coord or abs(p3[1]) > max_coord:
                    continue
                # Check distance to p2
                dx2 = p3[0] - p2[0]
                dy2 = p3[1] - p2[1]
                d2_23 = dx2*dx2 + dy2*dy2
                if not is_square(d2_23):
                    continue
                # Check not collinear
                if collinear(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]):
                    continue
                candidates.append(p3)
            
            if len(candidates) < target_n - 2:
                continue
            
            # Try to add more points
            for p3 in candidates[:min(20, len(candidates))]:
                if not check_time():
                    break
                
                current3 = [p1, p2, p3]
                
                # Find points that work with all 3
                cand4 = []
                for p4 in valid_disps:
                    if not check_time():
                        break
                    if p4 in current3:
                        continue
                    if abs(p4[0]) > max_coord or abs(p4[1]) > max_coord:
                        continue
                    valid = True
                    for p in current3:
                        dx = p4[0] - p[0]
                        dy = p4[1] - p[1]
                        d2 = dx*dx + dy*dy
                        if not is_square(d2):
                            valid = False
                            break
                    if not valid:
                        continue
                    # Check collinearity
                    for i in range(3):
                        for j in range(i+1, 3):
                            if collinear(current3[i][0], current3[i][1],
                                         current3[j][0], current3[j][1],
                                         p4[0], p4[1]):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    # Check concyclicity with the 3 points
                    if concyclic(current3[0][0], current3[0][1],
                                 current3[1][0], current3[1][1],
                                 current3[2][0], current3[2][1],
                                 p4[0], p4[1], 1):
                        continue
                    cand4.append(p4)
                
                if len(cand4) < target_n - 3:
                    continue
                
                for p4 in cand4[:min(15, len(cand4))]:
                    if not check_time():
                        break
                    
                    current4 = current3 + [p4]
                    
                    # Find points that work with all 4
                    cand5 = []
                    for p5 in valid_disps:
                        if not check_time():
                            break
                        if p5 in current4:
                            continue
                        if abs(p5[0]) > max_coord or abs(p5[1]) > max_coord:
                            continue
                        valid = True
                        for p in current4:
                            dx = p5[0] - p[0]
                            dy = p5[1] - p[1]
                            d2 = dx*dx + dy*dy
                            if not is_square(d2):
                                valid = False
                                break
                        if not valid:
                            continue
                        for i in range(4):
                            for j in range(i+1, 4):
                                if collinear(current4[i][0], current4[i][1],
                                             current4[j][0], current4[j][1],
                                             p5[0], p5[1]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        for i in range(4):
                            for j in range(i+1, 4):
                                for k in range(j+1, 4):
                                    if concyclic(current4[i][0], current4[i][1],
                                                 current4[j][0], current4[j][1],
                                                 current4[k][0], current4[k][1],
                                                 p5[0], p5[1], 1):
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        cand5.append(p5)
                    
                    if len(cand5) < target_n - 4:
                        continue
                    
                    for p5 in cand5[:min(10, len(cand5))]:
                        if not check_time():
                            break
                        
                        current5 = current4 + [p5]
                        
                        # Find points that work with all 5
                        for p6 in valid_disps:
                            if not check_time():
                                break
                            if p6 in current5:
                                continue
                            if abs(p6[0]) > max_coord or abs(p6[1]) > max_coord:
                                continue
                            valid = True
                            for p in current5:
                                dx = p6[0] - p[0]
                                dy = p6[1] - p[1]
                                d2 = dx*dx + dy*dy
                                if not is_square(d2):
                                    valid = False
                                    break
                            if not valid:
                                continue
                            for i in range(5):
                                for j in range(i+1, 5):
                                    if collinear(current5[i][0], current5[i][1],
                                                 current5[j][0], current5[j][1],
                                                 p6[0], p6[1]):
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                continue
                            for i in range(5):
                                for j in range(i+1, 5):
                                    for k in range(j+1, 5):
                                        if concyclic(current5[i][0], current5[i][1],
                                                     current5[j][0], current5[j][1],
                                                     current5[k][0], current5[k][1],
                                                     p6[0], p6[1], 1):
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            if not valid:
                                continue
                            
                            # Found a candidate set of 6
                            full_set = current5 + [p6]
                            if validate_set(full_set, 1):
                                return full_set
                        
                        # If we didn't find p6 in the first batch, try next p5
                        break  # Only try first p5 for now to save time
                    break  # Only try first p4
                break  # Only try first p3
            break  # Only try first p1
        
        return best_found
    
    # Search for 6 points with D=1
    result_6 = search_integer_points(max_coord=25, target_n=6)
    
    if result_6 and validate_set(result_6, 1):
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in result_6]}
    
    # Fallback: try D=3 search
    def search_d3_points(max_coord=20, target_n=6):
        if not check_time():
            return None
        
        valid_disps = []
        for dy in range(-max_coord, max_coord+1):
            for dx in range(-max_coord, max_coord+1):
                if dx == 0 and dy == 0:
                    continue
                d2 = dx*dx + 3*dy*dy
                if is_square(d2):
                    valid_disps.append((dx, dy))
        
        valid_disps.sort(key=lambda p: (p[0]*p[0] + 3*p[1]*p[1], p[0], p[1]))
        
        for p1_idx in range(min(10, len(valid_disps))):
            if not check_time():
                return None
            
            p1 = (0, 0)
            dx1, dy1 = valid_disps[p1_idx]
            p2 = (dx1, dy1)
            
            current = [p1, p2]
            
            candidates = []
            for dx, dy in valid_disps:
                if not check_time():
                    break
                p3 = (dx, dy)
                if p3 in current:
                    continue
                if abs(p3[0]) > max_coord or abs(p3[1]) > max_coord:
                    continue
                dx2 = p3[0] - p2[0]
                dy2 = p3[1] - p2[1]
                d2_23 = dx2*dx2 + 3*dy2*dy2
                if not is_square(d2_23):
                    continue
                if collinear(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]):
                    continue
                candidates.append(p3)
            
            if len(candidates) < target_n - 2:
                continue
            
            for p3 in candidates[:min(20, len(candidates))]:
                if not check_time():
                    break
                
                current3 = [p1, p2, p3]
                
                cand4 = []
                for p4 in valid_disps:
                    if not check_time():
                        break
                    if p4 in current3:
                        continue
                    if abs(p4[0]) > max_coord or abs(p4[1]) > max_coord:
                        continue
                    valid = True
                    for p in current3:
                        dx = p4[0] - p[0]
                        dy = p4[1] - p[1]
                        d2 = dx*dx + 3*dy*dy
                        if not is_square(d2):
                            valid = False
                            break
                    if not valid:
                        continue
                    for i in range(3):
                        for j in range(i+1, 3):
                            if collinear(current3[i][0], current3[i][1],
                                         current3[j][0], current3[j][1],
                                         p4[0], p4[1]):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    if concyclic(current3[0][0], current3[0][1],
                                 current3[1][0], current3[1][1],
                                 current3[2][0], current3[2][1],
                                 p4[0], p4[1], 3):
                        continue
                    cand4.append(p4)
                
                if len(cand4) < target_n - 3:
                    continue
                
                for p4 in cand4[:min(15, len(cand4))]:
                    if not check_time():
                        break
                    
                    current4 = current3 + [p4]
                    
                    cand5 = []
                    for p5 in valid_disps:
                        if not check_time():
                            break
                        if p5 in current4:
                            continue
                        if abs(p5[0]) > max_coord or abs(p5[1]) > max_coord:
                            continue
                        valid = True
                        for p in current4:
                            dx = p5[0] - p[0]
                            dy = p5[1] - p[1]
                            d2 = dx*dx + 3*dy*dy
                            if not is_square(d2):
                                valid = False
                                break
                        if not valid:
                            continue
                        for i in range(4):
                            for j in range(i+1, 4):
                                if collinear(current4[i][0], current4[i][1],
                                             current4[j][0], current4[j][1],
                                             p5[0], p5[1]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        for i in range(4):
                            for j in range(i+1, 4):
                                for k in range(j+1, 4):
                                    if concyclic(current4[i][0], current4[i][1],
                                                 current4[j][0], current4[j][1],
                                                 current4[k][0], current4[k][1],
                                                 p5[0], p5[1], 3):
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        cand5.append(p5)
                    
                    if len(cand5) < target_n - 4:
                        continue
                    
                    for p5 in cand5[:min(10, len(cand5))]:
                        if not check_time():
                            break
                        
                        current5 = current4 + [p5]
                        
                        for p6 in valid_disps:
                            if not check_time():
                                break
                            if p6 in current5:
                                continue
                            if abs(p6[0]) > max_coord or abs(p6[1]) > max_coord:
                                continue
                            valid = True
                            for p in current5:
                                dx = p6[0] - p[0]
                                dy = p6[1] - p[1]
                                d2 = dx*dx + 3*dy*dy
                                if not is_square(d2):
                                    valid = False
                                    break
                            if not valid:
                                continue
                            for i in range(5):
                                for j in range(i+1, 5):
                                    if collinear(current5[i][0], current5[i][1],
                                                 current5[j][0], current5[j][1],
                                                 p6[0], p6[1]):
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                continue
                            for i in range(5):
                                for j in range(i+1, 5):
                                    for k in range(j+1, 5):
                                        if concyclic(current5[i][0], current5[i][1],
                                                     current5[j][0], current5[j][1],
                                                     current5[k][0], current5[k][1],
                                                     p6[0], p6[1], 3):
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            if not valid:
                                continue
                            
                            full_set = current5 + [p6]
                            if validate_set(full_set, 3):
                                return full_set
                        
                        break
                    break
                break
            break
        
        return None
    
    result_6_d3 = search_d3_points(max_coord=15, target_n=6)
    
    if result_6_d3 and validate_set(result_6_d3, 3):
        return {'D': 3, 'points': [[[x, 1], [y, 1]] for x, y in result_6_d3]}
    
    # Fallback: known valid 5-point set with D=1
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if validate_set(seed_points, 1):
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in seed_points]}
    
    # Last resort: return 3 known valid points
    return {'D': 1, 'points': [[[0, 1], [0, 1]], [[3, 1], [4, 1]], [[0, 1], [8, 1]]]}
