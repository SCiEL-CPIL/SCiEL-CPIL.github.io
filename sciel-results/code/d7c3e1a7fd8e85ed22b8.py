import time
import random
from itertools import combinations
import math

def count_ordinary_lines(points):
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c)) if (a != 0 or b != 0 or c != 0) else 1
            if g == 0: g = 1
            a, b, c = a // g, b // g, c // g
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for count in line_counts.values() if count == 1)

def construct():
    deadline = time.time() + 140
    # Start with 3x4 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    best_ordinary = count_ordinary_lines(points)
    best_points = [p[:] for p in points]
    
    # Helper to get candidate moves
    def get_candidate_moves(pts):
        candidates = []
        # 1. Random moves
        for _ in range(5):
            candidates.append([random.randint(-100, 100), random.randint(-100, 100)])
        
        # 2. Moves to form 3-point lines
        # For each pair of points, find a third point on the line? No, move a point TO a line.
        # Pick two points A, B. The line AB exists.
        # If we move C to a point D on line AB, then A,B,D are collinear.
        # But D must be an integer point distinct from A, B, and other points.
        # Also, we should move a point that is currently "expensive".
        
        # Simple approach: try moving each point to a random location on a random line defined by two other points.
        for i in range(12):
            # Pick two other points
            others = [j for j in range(12) if j != i]
            if len(others) < 2: continue
            j, k = random.sample(others, 2)
            x1, y1 = pts[j]
            x2, y2 = pts[k]
            
            # Line through (x1,y1) and (x2,y2)
            # Parametric: (x1 + t*dx, y1 + t*dy)
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            if g == 0: g = 1
            dx //= g
            dy //= g
            
            # Try a few integer steps t
            for t in range(-5, 6):
                if t == 0: continue # Don't move to existing point? Actually t=0 is (x1,y1) which is pts[j].
                # We are moving pts[i] to (x1 + t*dx, y1 + t*dy)
                nx = x1 + t * dx
                ny = y1 + t * dy
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    candidates.append([nx, ny])
                    
        return candidates

    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 11)
        old_point = points[idx][:]
        
        moves = get_candidate_moves(points)
        
        for move in moves:
            # Check distinctness
            if any(p[0] == move[0] and p[1] == move[1] for i, p in enumerate(points) if i != idx):
                continue
            
            points[idx] = move
            new_ordinary = count_ordinary_lines(points)
            
            if new_ordinary < best_ordinary:
                best_ordinary = new_ordinary
                best_points = [p[:] for p in points]
                if best_ordinary <= 6:
                    return [[p[0], p[1]] for p in best_points]
            else:
                points[idx] = old_point
                # If we found a better one, we might want to stay there and explore further?
                # Standard hill climbing reverts if not better.
                # But if we are stuck, we can accept equal or slightly worse to escape?
                # For now, strict improvement.
                
    return [[p[0], p[1]] for p in best_points]
