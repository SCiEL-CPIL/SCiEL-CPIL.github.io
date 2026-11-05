import random
import time
import math

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _count_ordinary(points):
    lines = {}
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            g = _gcd(_gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in lines:
                lines[key] = 1
            else:
                lines[key] += 1
    return sum(1 for count in lines.values() if count == 1)

def construct():
    best_points = None
    best_score = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Initial seeds
    seeds = [
        [[x, y] for x in range(4) for y in range(3)],
        [[x, y] for x in range(3) for y in range(4)],
        # Random seed
        [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12)]
    ]
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        # Ensure distinct points for seed
        unique_seed = []
        seen = set()
        for p in seed:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique_seed.append(p)
        while len(unique_seed) < 12:
            x, y = random.randint(-10, 10), random.randint(-10, 10)
            if (x, y) not in seen:
                seen.add((x, y))
                unique_seed.append([x, y])
        
        current_points = unique_seed[:]
        current_score = _count_ordinary(current_points)
        
        if current_score < best_score:
            best_points = current_points[:]
            best_score = current_score
            
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                x, y = current_points[i]
                # Try neighbors
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if abs(nx) > 10**9 or abs(ny) > 10**9:
                            continue
                        new_points = current_points[:]
                        new_points[i] = [nx, ny]
                        
                        # Check distinctness
                        valid = True
                        for j in range(12):
                            if j != i and new_points[j] == new_points[i]:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        new_score = _count_ordinary(new_points)
                        if new_score < current_score:
                            current_points = new_points
                            current_score = new_score
                            improved = True
                            if current_score < best_score:
                                best_points = current_points[:]
                                best_score = current_score

    return best_points
