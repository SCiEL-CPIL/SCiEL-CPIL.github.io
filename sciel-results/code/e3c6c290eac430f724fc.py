import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial points: Parabola
    points = [[x, x * x] for x in range(-10, 10)]
    
    def get_triples(pts):
        triples = []
        n = len(pts)
        for i, j, k in combinations(range(n), 3):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            x3, y3 = pts[k]
            if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                triples.append((i, j, k))
        return triples

    def is_valid(pts):
        n = len(pts)
        for i, j in combinations(range(n), 2):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            count = 2
            for k in range(n):
                if k == i or k == j:
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        count += 1
            if count >= 4:
                return False
        return True

    def max_3_free_subset(pts):
        triples = get_triples(pts)
        if not triples:
            return len(pts)
        
        n = len(pts)
        deg = [0] * n
        for t in triples:
            for i in t:
                deg[i] += 1
        
        indices = list(range(n))
        indices.sort(key=lambda x: deg[x], reverse=True)
        
        triple_set = set(frozenset(t) for t in triples)
        
        best = 0
        current_set = []
        
        def dfs(pos, size):
            nonlocal best
            if size > best:
                best = size
            if pos == n:
                return
            if size + (n - pos) <= best:
                return
                
            idx = indices[pos]
            
            # Try including idx
            ok = True
            for i in range(len(current_set)):
                for j in range(i+1, len(current_set)):
                    if frozenset([current_set[i], current_set[j], idx]) in triple_set:
                        ok = False
                        break
                if not ok:
                    break
            
            if ok:
                current_set.append(idx)
                dfs(pos + 1, size + 1)
                current_set.pop()
                
            # Try excluding idx
            dfs(pos + 1, size)
            
        dfs(0, 0)
        return best

    best_score = max_3_free_subset(points)
    best_points = [p[:] for p in points]
    
    while time.time() < deadline:
        # Targeted move: Move point i to a point on line jk
        i = random.randint(0, 19)
        j, k = random.sample([x for x in range(20) if x != i], 2)
        
        xj, yj = points[j]
        xk, yk = points[k]
        
        dx = xk - xj
        dy = yk - yj
        
        if dx == 0 and dy == 0:
            continue
            
        g = math.gcd(abs(dx), abs(dy))
        if g == 0:
            continue
            
        step_x = dx // g
        step_y = dy // g
        
        for m in range(-200, 201):
            nx = xj + m * step_x
            ny = yj + m * step_y
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                # Check duplicates
                dup = False
                for idx, p in enumerate(points):
                    if p[0] == nx and p[1] == ny:
                        if idx != i:
                            dup = True
                            break
                if dup:
                    continue
                    
                new_points = [p[:] for p in points]
                new_points[i] = [nx, ny]
                
                if is_valid(new_points):
                    s = max_3_free_subset(new_points)
                    if s < best_score:
                        best_score = s
                        best_points = [p[:] for p in new_points]
                        
        # Random move
        if random.random() < 0.2:
            i = random.randint(0, 19)
            nx = random.randint(-100, 100)
            ny = random.randint(-100, 100)
            
            dup = False
            for idx, p in enumerate(points):
                if p[0] == nx and p[1] == ny:
                    if idx != i:
                        dup = True
                        break
            if not dup:
                new_points = [p[:] for p in points]
                new_points[i] = [nx, ny]
                if is_valid(new_points):
                    s = max_3_free_subset(new_points)
                    if s < best_score:
                        best_score = s
                        best_points = [p[:] for p in new_points]
                        
    return best_points
