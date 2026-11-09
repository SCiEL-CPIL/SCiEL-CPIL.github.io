import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Initial configuration: Perturbed 4x5 grid
    # Grid points: x in [0, 1, 2, 3], y in [0, 1, 2, 3, 4]
    # Scale and shift to center and spread out
    base_points = []
    for x in range(4):
        for y in range(5):
            base_points.append([x * 10, y * 10])
            
    # Randomly perturb each point
    points = []
    for p in base_points:
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        points.append([p[0] + dx, p[1] + dy])
        
    # Ensure distinctness and bounds
    def is_valid(pts):
        if len(pts) != 20:
            return False
        seen = set()
        for p in pts:
            if p in seen:
                return False
            seen.add(p)
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        # Check no 4 collinear
        for i in range(20):
            for j in range(i+1, 20):
                for k in range(j+1, 20):
                    # Check if there is a 4th point on the line
                    # Line through p_i, p_j, p_k
                    # We can just check all other points
                    # But this is O(N^4) which is 20^4 = 160k, acceptable
                    pass
        # More efficient check for 4 collinear
        lines = {}
        for i in range(20):
            for j in range(i+1, 20):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                # Slope representation
                if x1 == x2:
                    key = ('inf', x1)
                else:
                    slope = (y2 - y1) / (x2 - x1)
                    # Use a normalized representation to avoid float issues
                    # dy/dx reduced
                    dx = x2 - x1
                    dy = y2 - y1
                    g = math.gcd(abs(dx), abs(dy))
                    dx //= g
                    dy //= g
                    if dx < 0:
                        dx = -dx
                        dy = -dy
                    key = (dy, dx, y1 - (dy/dx)*x1 if dx != 0 else x1)
                    # Better: use tuple of reduced (dx, dy) and a point on the line
                    # Actually, just count points on each line defined by pairs
                    # If any line has >= 4 points, invalid.
                    # Let's just store lines by (slope, intercept) or similar.
                    # Simpler: for each pair, calculate line, then check all other points.
                    # But that's slow.
                    # Let's use a dictionary of lines.
                    # Key: (dx, dy, c) where line is dy*x - dx*y = c
                    # If dx=0, vertical line x = x1. Key: (0, 1, x1)
                    # If dy=0, horizontal line y = y1. Key: (1, 0, y1)
                    # General: dy*x - dx*y = c. c = dy*x1 - dx*y1.
                    # Normalize (dx, dy) by gcd and sign.
                    pass
        
        # Let's write a helper to check 4 collinear
        for i in range(20):
            for j in range(i+1, 20):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                if g == 0: continue # same point, but distinctness checked
                dx //= g
                dy //= g
                # Line: dy*(x - x1) - dx*(y - y1) = 0 => dy*x - dx*y = dy*x1 - dx*y1
                c = dy * x1 - dx * y1
                # Count points on this line
                count = 0
                for k in range(20):
                    xk, yk = pts[k]
                    if dy * xk - dx * yk == c:
                        count += 1
                if count >= 4:
                    return False
        return True

    # The above is_valid is O(N^3) per check, 20^3 = 8000. 
    # We can optimize by precomputing lines.
    
    def count_triples(pts):
        # Returns number of collinear triples
        # Also checks if any line has >= 4 points
        line_counts = {}
        for i in range(20):
            for j in range(i+1, 20):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                # Ensure consistent sign for direction
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                c = dy * x1 - dx * y1
                key = (dx, dy, c)
                if key not in line_counts:
                    line_counts[key] = set()
                line_counts[key].add(i)
                line_counts[key].add(j)
        
        triples = 0
        for key, points_on_line in line_counts.items():
            n = len(points_on_line)
            if n >= 4:
                return -1 # Invalid
            if n == 3:
                triples += 1
            # If n > 3, invalid
        return triples

    # Initial setup
    while not is_valid(points):
        # Re-perturb if invalid
        for i in range(20):
            points[i][0] += random.randint(-1, 1)
            points[i][1] += random.randint(-1, 1)
            # Clamp to bounds
            points[i][0] = max(-100, min(100, points[i][0]))
            points[i][1] = max(-100, min(100, points[i][1]))
        # Check distinctness
        if len(set(tuple(p) for p in points)) < 20:
            continue
            
    current_triples = count_triples(points)
    if current_triples < 0:
        current_triples = 0
        
    best_points = [p[:] for p in points]
    best_triples = current_triples
    
    # Local Search
    while time.time() < deadline:
        # Pick a random point and a random direction
        idx = random.randint(0, 19)
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        if dx == 0 and dy == 0:
            continue
            
        new_point = [points[idx][0] + dx, points[idx][1] + dy]
        
        # Check bounds
        if abs(new_point[0]) > 100 or abs(new_point[1]) > 100:
            continue
            
        # Check distinctness
        is_dup = False
        for i in range(20):
            if i != idx and points[i] == new_point:
                is_dup = True
                break
        if is_dup:
            continue
            
        # Try the move
        old_point = points[idx][:]
        points[idx] = new_point
        
        new_triples = count_triples(points)
        
        if new_triples > 0 and new_triples > current_triples:
            current_triples = new_triples
            if current_triples > best_triples:
                best_triples = current_triples
                best_points = [p[:] for p in points]
        else:
            # Revert
            points[idx] = old_point

    return best_points
