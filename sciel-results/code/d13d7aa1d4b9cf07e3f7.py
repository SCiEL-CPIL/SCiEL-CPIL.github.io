import random
import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 145.0

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
        
        def det4(m):
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

    def check_new_point(base, new_pt):
        """Fast check if adding new_pt to base maintains validity."""
        x, y = new_pt
        # Check distances to all base points
        for i, (bx, by) in enumerate(base):
            dx = x - bx
            dy = y - by
            d2 = dx*dx + dy*dy
            if not is_perfect_square(d2):
                return False
        # Check collinearity with pairs from base
        n = len(base)
        for i in range(n):
            for j in range(i+1, n):
                if is_collinear(base[i], base[j], new_pt):
                    return False
        # Check concyclicity with triples from base
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_concyclic(base[i], base[j], base[k], new_pt):
                        return False
        return True

    def generate_lattice_points(max_radius, D=1):
        """Generate integer lattice points within max_radius where all pairwise distances are integers."""
        pts = []
        for x in range(-max_radius, max_radius + 1):
            for y in range(-max_radius, max_radius + 1):
                if x == 0 and y == 0:
                    continue
                pts.append((x, y))
        return pts

    def greedy_build(points, max_points=12):
        """Greedily add points that maintain validity."""
        current = list(points)
        for candidate in points:
            if len(current) >= max_points:
                break
            if candidate not in current:
                if check_new_point(current, candidate):
                    current.append(candidate)
        return current

    # Seed 5-point set
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    bases = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (56, 33), (16, 63), (0, 33), (-16, 63)],
        [(0, 0), (7, 24), (15, 8), (24, 7), (32, 24)],
        [(0, 0), (3, 4), (4, 3), (5, 12), (12, 5)],
        [(0, 0), (8, 15), (15, 8), (9, 12), (12, 9)],
    ]
    
    # Verify bases first
    valid_bases = []
    for base in bases:
        if is_valid_set(base):
            valid_bases.append(base)
            
    if not valid_bases:
        if is_valid_set(seed_points):
            valid_bases.append(seed_points)
        else:
            return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in seed_points]}

    best_set = valid_bases[0]
    best_size = len(valid_bases[0])

    # Phase 1: Try to extend each base to 6+ points via grid search
    search_box = 200
    
    for base in valid_bases:
        if time.time() > deadline - 30.0:
            break
            
        for x in range(-search_box, search_box + 1):
            if time.time() > deadline - 10.0:
                break
            for y in range(-search_box, search_box + 1):
                if (x, y) in base:
                    continue
                # Quick check: distance to first point
                dx0 = x - base[0][0]
                dy0 = y - base[0][1]
                d2_0 = dx0*dx0 + dy0*dy0
                if not is_perfect_square(d2_0):
                    continue
                
                # Check distance to other points
                valid_dist = True
                for i in range(1, len(base)):
                    dx = x - base[i][0]
                    dy = y - base[i][1]
                    d2 = dx*dx + dy*dy
                    if not is_perfect_square(d2):
                        valid_dist = False
                        break
                
                if valid_dist:
                    if check_new_point(base, (x, y)):
                        new_set = base + [(x, y)]
                        if len(new_set) > best_size:
                            best_size = len(new_set)
                            best_set = new_set
                        # Try to extend further
                        extended = greedy_build(new_set, max_points=12)
                        if len(extended) > best_size:
                            best_size = len(extended)
                            best_set = extended
                        break
            if len(best_set) >= 8:
                break
        if len(best_set) >= 8:
            break

    # Phase 2: If we have 6+ points, try to extend further with larger search
    if best_size >= 6:
        extended_search_box = 300
        for x in range(-extended_search_box, extended_search_box + 1):
            if time.time() > deadline - 15.0:
                break
            for y in range(-extended_search_box, extended_search_box + 1):
                if (x, y) in best_set:
                    continue
                dx0 = x - best_set[0][0]
                dy0 = y - best_set[0][1]
                d2_0 = dx0*dx0 + dy0*dy0
                if not is_perfect_square(d2_0):
                    continue
                
                valid_dist = True
                for i in range(1, len(best_set)):
                    dx = x - best_set[i][0]
                    dy = y - best_set[i][1]
                    d2 = dx*dx + dy*dy
                    if not is_perfect_square(d2):
                        valid_dist = False
                        break
                
                if valid_dist:
                    if check_new_point(best_set, (x, y)):
                        new_set = best_set + [(x, y)]
                        if len(new_set) > best_size:
                            best_size = len(new_set)
                            best_set = new_set
                        extended = greedy_build(new_set, max_points=12)
                        if len(extended) > best_size:
                            best_size = len(extended)
                            best_set = extended
                        break
            if len(best_set) >= 10:
                break

    # Phase 3: Random perturbation search if still stuck
    if best_size < 7 and time.time() < deadline - 20.0:
        random.seed(42)
        for attempt in range(1000):
            if time.time() > deadline - 10.0:
                break
            # Random 5-point base
            base = []
            base.append((0, 0))
            for _ in range(4):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in base:
                    base.append((x, y))
            
            if len(base) < 5:
                continue
            if not is_valid_set(base):
                continue
            
            # Try to extend
            for x in range(-150, 151):
                if time.time() > deadline - 5.0:
                    break
                for y in range(-150, 151):
                    if (x, y) in base:
                        continue
                    dx0 = x - base[0][0]
                    dy0 = y - base[0][1]
                    d2_0 = dx0*dx0 + dy0*dy0
                    if not is_perfect_square(d2_0):
                        continue
                    
                    valid_dist = True
                    for i in range(1, len(base)):
                        dx = x - base[i][0]
                        dy = y - base[i][1]
                        d2 = dx*dx + dy*dy
                        if not is_perfect_square(d2):
                            valid_dist = False
                            break
                    
                    if valid_dist:
                        if check_new_point(base, (x, y)):
                            new_set = base + [(x, y)]
                            if len(new_set) > best_size:
                                best_size = len(new_set)
                                best_set = new_set
                            extended = greedy_build(new_set, max_points=12)
                            if len(extended) > best_size:
                                best_size = len(extended)
                                best_set = extended
                            break
                if len(best_set) >= 8:
                    break
            if len(best_set) >= 8:
                break

    pts = best_set
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
