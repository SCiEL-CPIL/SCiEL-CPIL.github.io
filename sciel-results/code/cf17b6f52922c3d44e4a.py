import random
import time

def construct():
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    # Initial configuration: 4x5 grid
    # x in 0..3, y in 0..4
    # Scale to fit within 100 and avoid small integer issues? 
    # Grid 0..3 and 0..4 is fine.
    points = []
    for x in range(4):
        for y in range(5):
            points.append([x, y])
            
    # Function to check if 4 points are collinear
    def has_four_collinear(pts):
        n = len(pts)
        # Store lines as sets of indices
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                # Calculate normal vector or key
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                # Key for line (A, B, C) where A*x + B*y = C
                # A = y2 - y1, B = x1 - x2, C = A*x1 + B*y1
                A = y2 - y1
                B = x1 - x2
                C = A * x1 + B * y1
                # Normalize (A, B, C)
                # Find GCD
                from math import gcd
                g = gcd(gcd(abs(A), abs(B)), abs(C)) if C != 0 else gcd(abs(A), abs(B))
                if g == 0:
                    continue
                A //= g
                B //= g
                C //= g
                # Ensure sign consistency
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                key = (A, B, C)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        
        for key, indices in lines.items():
            if len(indices) >= 4:
                return True
        return False

    # Function to count collinear triples
    def count_triples(pts):
        n = len(pts)
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                A = y2 - y1
                B = x1 - x2
                C = A * x1 + B * y1
                from math import gcd
                g = gcd(gcd(abs(A), abs(B)), abs(C)) if C != 0 else gcd(abs(A), abs(B))
                if g == 0:
                    continue
                A //= g
                B //= g
                C //= g
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                key = (A, B, C)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        
        count = 0
        for key, indices in lines.items():
            if len(indices) >= 3:
                count += len(indices) * (len(indices) - 1) * (len(indices) - 2) // 6
        return count

    # Initial cleanup: break 4-collinearities
    # We will move points that are part of 4-collinear sets
    for _ in range(1000):
        if not has_four_collinear(points):
            break
        # Find a point to move
        # Simple heuristic: move a random point
        idx = random.randint(0, 19)
        # Try moving to a random location
        for _ in range(50):
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            if [new_x, new_y] in points:
                continue
            old_pt = points[idx][:]
            points[idx] = [new_x, new_y]
            if not has_four_collinear(points):
                break
            points[idx] = old_pt
        if has_four_collinear(points):
            # If stuck, restart with random points
            points = []
            while len(points) < 20:
                p = [random.randint(-100, 100), random.randint(-100, 100)]
                if p not in points:
                    points.append(p)
            continue

    # Local search to maximize triples
    best_triples = count_triples(points)
    best_points = [p[:] for p in points]
    
    current_triples = best_triples
    current_points = [p[:] for p in points]
    
    while time.time() - start_time < time_limit:
        # Pick a point to move
        idx = random.randint(0, 19)
        old_pt = current_points[idx][:]
        
        # Try a random move
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        
        if [new_x, new_y] in current_points:
            continue
            
        current_points[idx] = [new_x, new_y]
        
        if not has_four_collinear(current_points):
            new_triples = count_triples(current_points)
            if new_triples > current_triples:
                current_triples = new_triples
                if new_triples > best_triples:
                    best_triples = new_triples
                    best_points = [p[:] for p in current_points]
            else:
                # Accept with some probability (simulated annealing style)
                # Or just reject to keep it simple and fast
                # Let's use a simple acceptance: if new_triples >= current_triples - 2
                if new_triples >= current_triples - 2:
                    current_triples = new_triples
                else:
                    current_points[idx] = old_pt
        else:
            current_points[idx] = old_pt

    return best_points
