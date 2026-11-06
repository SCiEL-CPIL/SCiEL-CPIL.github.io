import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def count_ordinary(points):
        n = len(points)
        if n < 3:
            return 0
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                p1 = points[i]
                p2 = points[j]
                # Normalize line representation
                # A line is defined by ax + by + c = 0
                # a = y2 - y1, b = x1 - x2, c = x2*y1 - x1*y2
                a = p2[1] - p1[1]
                b = p1[0] - p2[0]
                c = p2[0]*p1[1] - p1[0]*p2[1]
                # Normalize (a,b,c) by GCD
                from math import gcd
                g = gcd(gcd(abs(a), abs(b)), abs(c))
                if g != 0:
                    a //= g
                    b //= g
                    c //= g
                # Ensure consistent sign
                if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                    a = -a
                    b = -b
                    c = -c
                key = (a, b, c)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        
        ordinary_count = 0
        for key, pts_in_line in lines.items():
            if len(pts_in_line) == 2:
                ordinary_count += 1
        return ordinary_count

    def random_points():
        return [[random.randint(-20, 20), random.randint(-20, 20)] for _ in range(12)]

    def local_search(points):
        points = [list(p) for p in points]
        current_cost = count_ordinary(points)
        best_points = [list(p) for p in points]
        best_cost = current_cost
        
        for step in range(5000):
            if time.time() > deadline:
                break
            # Pick a random point to move
            i = random.randint(0, 11)
            old_x, old_y = points[i]
            
            # Try moving to a nearby point
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
            
            # Ensure distinctness
            if any(points[j] == [new_x, new_y] for j in range(12) if j != i):
                continue
                
            points[i] = [new_x, new_y]
            new_cost = count_ordinary(points)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_points = [list(p) for p in points]
            else:
                # Revert if not better (greedy)
                # Or accept with probability (simulated annealing)
                # For simplicity, greedy first
                points[i] = [old_x, old_y]
                
        return best_points, best_cost

    best_points = None
    best_cost = float('inf')
    
    # Initial configurations
    # 1. 3x3 grid + 3 random
    base = [[x, y] for x in range(3) for y in range(3)]
    for _ in range(5):
        pts = base + [[random.randint(-5, 5), random.randint(-5, 5)] for _ in range(3)]
        # Ensure distinct
        seen = set()
        unique_pts = []
        for p in pts:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique_pts.append(p)
        while len(unique_pts) < 12:
            p = [random.randint(-5, 5), random.randint(-5, 5)]
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique_pts.append(p)
        
        res_pts, res_cost = local_search(unique_pts)
        if res_cost < best_cost:
            best_cost = res_cost
            best_points = res_pts
            if best_cost == 6:
                break

    # 2. Random starts
    for _ in range(10):
        pts = random_points()
        # Ensure distinct
        seen = set()
        unique_pts = []
        for p in pts:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique_pts.append(p)
        while len(unique_pts) < 12:
            p = [random.randint(-20, 20), random.randint(-20, 20)]
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique_pts.append(p)
        
        res_pts, res_cost = local_search(unique_pts)
        if res_cost < best_cost:
            best_cost = res_cost
            best_points = res_pts
            if best_cost == 6:
                break

    if best_points is None:
        # Fallback
        return [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
