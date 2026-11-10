import time
import random
from collections import defaultdict

def count_ordinary_lines(points):
    n = len(points)
    lines = defaultdict(list)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Simplify line representation
            if dx == 0:
                key = (0, 1, x1)
            elif dy == 0:
                key = (1, 0, y1)
            else:
                # A = dy, B = -dx, C = dx*y1 - dy*x1
                # Normalize
                g = abs(dx)
                if g > 0:
                    # This is not GCD. We need GCD of dx, dy
                    pass
                # Use GCD
                import math
                g = math.gcd(dx, dy)
                if g < 0:
                    g = -g
                A = dy // g
                B = -dx // g
                C = (dx * y1 - dy * x1) // g
                # Ensure unique sign
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                key = (A, B, C)
            lines[key].append(i)
    
    ordinary = 0
    for key, pts in lines.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit
    
    # Initial configuration: 3x3 grid + 3 points
    # 3x3 grid has 8 ordinary lines.
    # Let's try to find a better one.
    
    # Known good configuration for 12 points with 6 ordinary lines:
    # It is the Pappus configuration (9 points) + 3 points.
    # Let's generate Pappus points.
    # A1(0,0), A2(1,0), A3(0,1)
    # B1(2,2), B2(3,1), B3(1,2) ?
    # Center (1,1)?
    # A1(0,0) -> B1(2,2)
    # A2(1,0) -> B2(3,1)
    # A3(0,1) -> B3(1,2)
    # Pappus points:
    # A1A2 (y=0) and B1B2 (y=0.5) -> intersect at infinity?
    # Let's use the standard rational Pappus.
    
    # Points:
    p0 = (0, 0)
    p1 = (1, 0)
    p2 = (0, 1)
    p3 = (2, 2)
    p4 = (3, 1)
    p5 = (1, 2)
    
    # Intersections:
    # Line A1A2 (y=0) and B1B2 (passing (2,2) and (3,1) -> y = -x+4). Intersect: 0 = -x+4 => x=4. P6(4,0).
    # Line A2A3 (x+y=1) and B2B3 (passing (3,1) and (1,2) -> y = -0.5x+2.5).
    # x + (-0.5x+2.5) = 1 => 0.5x = -1.5 => x=-3. y=4. P7(-3,4).
    # Line A3A1 (x=0) and B3B1 (passing (1,2) and (2,2) -> y=2). Intersect: (0,2). P8(0,2).
    
    p6 = (4, 0)
    p7 = (-3, 4)
    p8 = (0, 2)
    
    points = [p0, p1, p2, p3, p4, p5, p6, p7, p8]
    
    # Check ordinary lines for these 9 points.
    # Should be 0.
    
    # Now add 3 points.
    # Let's add points that are "far away" or specific.
    # Try adding (10, 10), (10, -10), (-10, 10).
    p9 = (10, 10)
    p10 = (10, -10)
    p11 = (-10, 10)
    
    points.append(p9)
    points.append(p10)
    points.append(p11)
    
    best_points = points[:]
    best_score = count_ordinary_lines(best_points)
    
    # Local search
    random.seed(42)
    while time.time() < deadline:
        # Pick a random point and move it
        i = random.randint(0, 11)
        old_x, old_y = points[i]
        # Try a random new position
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        
        points[i] = (new_x, new_y)
        score = count_ordinary_lines(points)
        
        if score < best_score:
            best_score = score
            best_points = points[:]
            if best_score == 6:
                break
        else:
            points[i] = (old_x, old_y)
            
    return [[x, y] for x, y in best_points]
