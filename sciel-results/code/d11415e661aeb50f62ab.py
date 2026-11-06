import math
import random
import time

def count_ordinary(points):
    n = len(points)
    lines = set()
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            A = p1[1] - p2[1]
            B = p2[0] - p1[0]
            C = -(A * p1[0] + B * p1[1])
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            lines.add((A, B, C))
            
    ordinary = 0
    for line in lines:
        A, B, C = line
        count = 0
        for p in points:
            if A * p[0] + B * p[1] + C == 0:
                count += 1
        if count == 2:
            ordinary += 1
    return ordinary

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Initial configuration: 3x4 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    current_count = count_ordinary(points)
    best_count = current_count
    best_points = [p[:] for p in points]
    
    # Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(12):
            x, y = points[i]
            # Try moving to nearby points
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    # Check if occupied
                    occupied = False
                    for j in range(12):
                        if j != i and points[j][0] == nx and points[j][1] == ny:
                            occupied = True
                            break
                    if occupied:
                        continue
                    # Move
                    points[i] = [nx, ny]
                    new_count = count_ordinary(points)
                    if new_count < current_count:
                        current_count = new_count
                        improved = True
                        if current_count < best_count:
                            best_count = current_count
                            best_points = [p[:] for p in points]
                        break
                    else:
                        points[i] = [x, y]
                        
    return best_points
