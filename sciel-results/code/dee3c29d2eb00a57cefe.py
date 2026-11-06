import time
import random

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave margin for evaluator overhead
    
    N = 20
    BOUND = 100
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def is_valid_config(points):
        if len(set(map(tuple, points))) != N:
            return False
        for p in points:
            if abs(p[0]) > BOUND or abs(p[1]) > BOUND:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                count = 0
                p1 = points[i]
                p2 = points[j]
                for k in range(N):
                    if k == i or k == j:
                        continue
                    if are_collinear(p1, p2, points[k]):
                        count += 1
                if count >= 2:
                    return False
        return True

    def check_move_valid(points, idx, new_point):
        for i in range(N):
            if i != idx and points[i] == new_point:
                return False
        if abs(new_point[0]) > BOUND or abs(new_point[1]) > BOUND:
            return False
        for i in range(N):
            if i == idx:
                continue
            count = 0
            for j in range(N):
                if j == idx or j == i:
                    continue
                if are_collinear(new_point, points[i], points[j]):
                    count += 1
            if count >= 2:
                return False
        return True

    def calc_alpha(points):
        triples = []
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        
        if not triples:
            return N
        
        best = 0
        
        def dfs(mask, depth):
            nonlocal best
            cnt = mask.bit_count()
            if cnt <= best:
                return
            
            found_triple = False
            for (i, j, k) in triples:
                if (mask & (1 << i)) and (mask & (1 << j)) and (mask & (1 << k)):
                    found_triple = True
                    if cnt - 1 > best:
                        dfs(mask & ~(1 << i), depth + 1)
                        dfs(mask & ~(1 << j), depth + 1)
                        dfs(mask & ~(1 << k), depth + 1)
                    return
            if not found_triple:
                if cnt > best:
                    best = cnt
                return
        
        init_mask = (1 << N) - 1
        dfs(init_mask, 0)
        return best

    def generate_initial():
        while True:
            pts = []
            used = set()
            success = True
            for _ in range(N):
                while True:
                    x = random.randint(-BOUND, BOUND)
                    y = random.randint(-BOUND, BOUND)
                    p = (x, y)
                    if p in used:
                        continue
                    pts.append(p)
                    if is_valid_config(pts):
                        used.add(p)
                        break
                    else:
                        pts.pop()
                        # Try again
                else:
                    success = False
                    break
            if success and len(pts) == N:
                return pts

    points = generate_initial()
    current_alpha = calc_alpha(points)
    best_alpha = current_alpha
    best_points = [list(p) for p in points]
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        idx = random.randint(0, N - 1)
        new_p = (random.randint(-BOUND, BOUND), random.randint(-BOUND, BOUND))
        
        if check_move_valid(points, idx, new_p):
            old_p = points[idx]
            points[idx] = new_p
            
            new_alpha = calc_alpha(points)
            
            delta = new_alpha - current_alpha
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_points = [list(p) for p in points]
            else:
                points[idx] = old_p
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_points
