import time
import math
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, -x1)
    if dy == 0:
        return (0, 1, -y1)
    g = _gcd(dx, dy)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    g2 = _gcd(dy, dx)
    if g2 > 1:
        dy //= g2
        dx //= g2
        c //= g2
    return (dy, -dx, c)

def _count_ordinary(points):
    n = len(points)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return sum(1 for pts in lines.values() if len(pts) == 2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # The 3x4 grid has 12 points. Let's check its score.
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    score_3x4 = _count_ordinary(grid_3x4)
    
    # Theoretical minimum is 6. 
    # If the 3x4 grid is not 6, we might try other small grids or structured sets.
    # However, the 3x4 grid is a very strong candidate for minimal ordinary lines.
    # Let's also check a few other structured configurations to be safe.
    
    candidates = []
    
    # 1. 3x4 grid
    candidates.append(grid_3x4)
    
    # 2. 2x6 grid
    grid_2x6 = [[x, y] for x in range(6) for y in range(2)]
    candidates.append(grid_2x6)
    
    # 3. 1x12 line (all collinear, but task says "not all collinear")
    # So we skip pure lines.
    
    # 4. 4x3 grid (same as 3x4 rotated)
    # 5. Triangle-like structure: 4 points on each side of a triangle, no shared vertices
    #    Side 1: y=0, x=0,1,2,3
    #    Side 2: x=4, y=0,1,2,3 -> (4,0) is shared? No, let's make them disjoint.
    #    Let's try:
    #    P1: (0,0), (1,0), (2,0), (3,0)
    #    P2: (0,1), (0,2), (0,3), (0,4)
    #    P3: (1,4), (2,4), (3,4), (4,4)
    #    This is a "staircase" or similar.
    #    Let's just evaluate the candidates we have.
    
    best_points = None
    best_score = float('inf')
    
    for cand in candidates:
        if len(cand) != 12:
            continue
        # Check distinctness
        if len(set(map(tuple, cand))) != 12:
            continue
        # Check not all collinear
        if len(set([_line_key(cand[0], cand[i]) for i in range(1, 12)])) == 1:
            continue
            
        s = _count_ordinary(cand)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in cand]
    
    if best_points is None:
        # Fallback: return 3x4 grid
        best_points = grid_3x4
        best_score = _count_ordinary(best_points)
    
    # If the score is already 6 (the theoretical minimum), we can return immediately.
    if best_score == 6:
        return best_points
    
    # If not 6, we could try a local search, but given the budget and the strong performance of the 3x4 grid
    # (which often yields 6 or close to it for these small n), we might just return the best found.
    # However, to be safe and potentially improve, let's do a very fast local search if time permits.
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Simple local search: try moving one point to a random position in a small neighborhood
    # This is a fallback in case the initial candidates are suboptimal.
    
    import random
    max_iter = 5000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, 11)
        old_x, old_y = current_points[idx]
        
        # Try a few random moves
        improved = False
        for _ in range(10):
            nx = old_x + random.randint(-5, 5)
            ny = old_y + random.randint(-5, 5)
            
            # Check bounds
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
                
            # Check distinctness
            if [nx, ny] in current_points:
                continue
                
            current_points[idx] = [nx, ny]
            new_score = _count_ordinary(current_points)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
                improved = True
                break
            else:
                current_points[idx] = [old_x, old_y]
        
        if not improved:
            # Restart from best
            current_points = [p[:] for p in best_points]
            current_score = best_score
    
    return best_points
