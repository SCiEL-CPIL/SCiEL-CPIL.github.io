import math
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit

    # Known 5-point configuration with integer distances
    # Points: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def is_perfect_square(n):
        if n < 0:
            return False
        root = math.isqrt(n)
        return root * root == n

    def is_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]) == 0

    def is_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        A = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
        if A == 0:
            return True
        
        B1 = -(x2*x2 + y2*y2 - x1*x1 - y1*y1)
        B2 = -(x3*x3 + y3*y3 - x1*x1 - y1*y1)
        
        num_A = B1*(y3-y1) - B2*(y2-y1)
        num_B = B2*(x2-x1) - B1*(x3-x1)
        
        val = A*(x4*x4 + y4*y4 + x1*x1 + y1*y1) + num_A*(x4+x1) + num_B*(y4+y1)
        
        return val == 0

    def check_candidate(candidate, existing_points):
        for p in existing_points:
            dx = candidate[0] - p[0]
            dy = candidate[1] - p[1]
            d2 = dx*dx + dy*dy
            if not is_perfect_square(d2):
                return False
        
        n = len(existing_points)
        for i in range(n):
            for j in range(i+1, n):
                if is_collinear(candidate, existing_points[i], existing_points[j]):
                    return False
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_concyclic(candidate, existing_points[i], existing_points[j], existing_points[k]):
                        return False
                        
        return True

    def generate_candidates(R, max_hyp=None):
        if max_hyp is None:
            max_hyp = R * 2
        
        candidates = set()
        max_u = int(math.isqrt(max_hyp)) + 2
        
        for u in range(2, max_u + 1):
            u2 = u * u
            for v in range(1, u):
                if (u - v) % 2 == 0:
                    continue
                if math.gcd(u, v) != 1:
                    continue
                v2 = v * v
                x_base = u2 - v2
                y_base = 2 * u * v
                k_base = u2 + v2
                
                if k_base > max_hyp:
                    continue
                
                max_k = max_hyp // k_base
                for k in range(1, max_k + 1):
                    x = k * x_base
                    y = k * y_base
                    if abs(x) > R or abs(y) > R:
                        continue
                    candidates.add((x, y))
                    candidates.add((x, -y))
                    candidates.add((-x, y))
                    candidates.add((-x, -y))
                    if x != y:
                        candidates.add((y, x))
                        candidates.add((y, -x))
                        candidates.add((-y, x))
                        candidates.add((-y, -x))
        
        for i in range(1, R + 1):
            candidates.add((i, 0))
            candidates.add((-i, 0))
            candidates.add((0, i))
            candidates.add((0, -i))
        
        return candidates

    best_set = seed_points
    search_radii = [100, 200, 400, 800, 1200]
    
    found_6 = None
    for R in search_radii:
        if time.time() - start_time > deadline - 20:
            break
            
        candidates = generate_candidates(R)
        for p in seed_points:
            candidates.discard(p)
        
        candidates = sorted(candidates, key=lambda p: p[0]*p[0] + p[1]*p[1])
        
        for cand in candidates:
            if time.time() - start_time > deadline - 15:
                break
            if check_candidate(cand, seed_points):
                found_6 = cand
                break
        
        if found_6:
            break
    
    if found_6:
        points_6 = seed_points + [found_6]
        
        found_7 = None
        time_limit_7 = min(time.time() + 15, deadline - 5)
        
        for R in search_radii:
            if time.time() > time_limit_7:
                break
            candidates = generate_candidates(R)
            for p in points_6:
                candidates.discard(p)
            
            candidates = sorted(candidates, key=lambda p: p[0]*p[0] + p[1]*p[1])
            
            for cand in candidates:
                if time.time() > time_limit_7:
                    break
                if check_candidate(cand, points_6):
                    found_7 = cand
                    break
            
            if found_7:
                break
        
        if found_7:
            points_6 = points_6 + [found_7]
            
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points_6]}
        
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
