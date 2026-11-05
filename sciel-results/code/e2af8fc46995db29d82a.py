import numpy as np
import random
import time

def construct():
    def max_no_3_collinear(points):
        n = len(points)
        triples = []
        triple_masks = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        triples.append((i, j, k))
                        triple_masks.append((1 << i) | (1 << j) | (1 << k))
        
        if not triples:
            return n
            
        best = 0
        
        def popcount(x):
            return bin(x).count('1')
            
        def get_bound(mask):
            count = popcount(mask)
            temp = mask
            disjoint = 0
            for tm in triple_masks:
                if (tm & temp) == tm:
                    disjoint += 1
                    temp &= ~tm
            return count - disjoint

        def solve(mask):
            nonlocal best
            ub = get_bound(mask)
            if ub <= best:
                return ub
            
            branch_idx = -1
            for idx, tm in enumerate(triple_masks):
                if (tm & mask) == tm:
                    branch_idx = idx
                    break
                    
            if branch_idx == -1:
                val = popcount(mask)
                if val > best:
                    best = val
                return val
                
            t = triples[branch_idx]
            u, v, w = t
            
            r1 = solve(mask & ~(1 << u))
            r2 = solve(mask & ~(1 << v))
            r3 = solve(mask & ~(1 << w))
            
            res = max(r1, r2, r3)
            if res > best:
                best = res
            return res

        solve((1 << n) - 1)
        return best

    def is_valid(points):
        if len(points) != 20:
            return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        if len(set(map(tuple, points))) != 20:
            return False
        
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        count += 1
                if count >= 4:
                    return False
        return True

    random.seed(42)
    
    # Start with 5x4 grid
    points = []
    for i in range(5):
        for j in range(4):
            points.append([i * 20, j * 20])
            
    # Repair validity
    valid = False
    attempts = 0
    while not valid and attempts < 5000:
        if is_valid(points):
            valid = True
            break
        
        # Find a 4-collinear set
        n = len(points)
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                line_pts = []
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        line_pts.append(k)
                if len(line_pts) >= 2:
                    idx = random.choice(line_pts)
                    new_x = points[idx][0] + random.randint(-10, 10)
                    new_y = points[idx][1] + random.randint(-10, 10)
                    if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                        collides = False
                        for l, p in enumerate(points):
                            if l != idx and p[0] == new_x and p[1] == new_y:
                                collides = True
                                break
                        if not collides:
                            points[idx] = [new_x, new_y]
                            moved = True
                            break
            if moved:
                break
        if not moved:
            idx = random.randint(0, 19)
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            collides = False
            for l, p in enumerate(points):
                if l != idx and p[0] == new_x and p[1] == new_y:
                    collides = True
                    break
            if not collides:
                points[idx] = [new_x, new_y]
            else:
                # Reset if stuck
                points = []
                for i in range(5):
                    for j in range(4):
                        points.append([i * 20, j * 20])
        attempts += 1
        
    if not valid:
        # Fallback
        return [[x, x * x % 101] for x in range(20)]

    best_points = [p[:] for p in points]
    best_score = max_no_3_collinear(points)
    
    time_limit = time.time() + 115
    
    while time.time() < time_limit:
        idx = random.randint(0, 19)
        old_p = points[idx][:]
        
        improved = False
        for _ in range(20):
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
            collides = False
            for i, p in enumerate(points):
                if i != idx and p[0] == new_x and p[1] == new_y:
                    collides = True
                    break
            if collides:
                continue
                
            points[idx] = [new_x, new_y]
            
            if is_valid(points):
                new_score = max_no_3_collinear(points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in points]
                    improved = True
                    break
                else:
                    points[idx] = old_p
            else:
                points[idx] = old_p
                
        if not improved:
            i1, i2 = random.sample(range(20), 2)
            points[i1], points[i2] = points[i2][:], points[i1][:]
            if is_valid(points):
                new_score = max_no_3_collinear(points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in points]
                else:
                    points[i1], points[i2] = points[i2][:], points[i1][:]
            else:
                points[i1], points[i2] = points[i2][:], points[i1][:]
                
    return best_points
