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
    deadline = start_time + 140.0 # Safety margin
    
    best_points = None
    best_score = float('inf')
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in pts]
            
    # 1. Evaluate standard small grids
    # 3x4 grid
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    update_best(grid_3x4, _count_ordinary(grid_3x4))
    
    # 2x6 grid
    grid_2x6 = [[x, y] for x in range(6) for y in range(2)]
    update_best(grid_2x6, _count_ordinary(grid_2x6))
    
    # 4x3 grid (same as 3x4 but rotated, just in case of symmetry issues, though _line_key handles it)
    # 1x12 is invalid (all collinear)
    
    # 2. Pappus Configuration
    # Pappus theorem: If A, B, C are on line L1 and A', B', C' are on line L2, 
    # then the intersections of AB' and A'B, AC' and A'C, BC' and B'C are collinear.
    # We want 12 points. 
    # Let's try a specific known minimal configuration for n=12.
    # The "12-point configuration" with 6 ordinary lines is often related to the affine plane of order 3 or similar structures.
    
    # Let's try to construct a configuration based on 4 lines with 3 points each, plus 4 extra points?
    # Or 3 lines with 4 points each?
    # If we have 3 lines, each with 4 points, and they intersect nicely...
    # Let's try the "grid" approach but with affine transformations to break symmetries that might create more ordinary lines?
    # Actually, the 3x4 grid is an affine image of a subset of the affine plane.
    
    # Let's try a specific candidate: The "Hesse configuration" is 9 points. 
    # For 12, we can add 3 points to the Hesse configuration?
    
    # Let's try a systematic search over affine transformations of the 3x4 grid.
    # Points: (i, j) for i in 0..3, j in 0..2.
    # Transform: x' = a*x + b*y + c, y' = d*x + e*y + f.
    # We need integer coordinates.
    # If we pick a=1, b=0, c=0, d=0, e=1, f=0, we get the standard grid.
    # Let's try shears: x' = x + k*y, y' = y.
    
    for k in range(-5, 6):
        if time.time() > deadline: break
        pts = [[x + k*y, y] for x in range(4) for y in range(3)]
        # Check distinctness
        if len(set(map(tuple, pts))) != 12: continue
        # Check not all collinear
        if len(set([_line_key(pts[0], pts[i]) for i in range(1, 12)])) == 1: continue
        s = _count_ordinary(pts)
        update_best(pts, s)
        
        # Also try x' = x, y' = y + k*x
        pts2 = [[x, y + k*x] for x in range(4) for y in range(3)]
        if len(set(map(tuple, pts2))) != 12: continue
        if len(set([_line_key(pts2[0], pts2[i]) for i in range(1, 12)])) == 1: continue
        s2 = _count_ordinary(pts2)
        update_best(pts2, s2)
        
    # 3. Try configurations with 4 collinear points on 3 separate lines (12 points total)
    # Line 1: y=0, x=0,1,2,3
    # Line 2: y=1, x=10,11,12,13
    # Line 3: y=2, x=20,21,22,23
    # This will likely have many ordinary lines connecting the groups.
    # Better: Intersecting lines.
    # Triangle with 4 points on each side (excluding vertices to avoid sharing? Or including?)
    # If we use vertices, we have 3 vertices + 3*2 = 9 points? No.
    # Let's try:
    # L1: y=0, x=0..3
    # L2: x=0, y=1..4
    # L3: x+y=5, (1,4)..(4,1) -> (1,4) is on L2? No, L2 is x=0. (1,4) is not on x=0.
    # Let's define:
    # P1: (0,0), (1,0), (2,0), (3,0)
    # P2: (0,1), (0,2), (0,3), (0,4)
    # P3: (1,4), (2,3), (3,2), (4,1)
    # Check distinctness:
    # P1: y=0
    # P2: x=0, y>0
    # P3: x+y=5, x>0, y>0
    # Intersections:
    # P1 & P2: (0,0) is in P1, but P2 starts at (0,1). No overlap.
    # P1 & P3: y=0 and x+y=5 -> (5,0). Not in P1 (x<=3). No overlap.
    # P2 & P3: x=0 and x+y=5 -> (0,5). Not in P2 (y<=4). No overlap.
    # So 12 distinct points.
    tri_pts = []
    for x in range(4): tri_pts.append([x, 0])
    for y in range(1, 5): tri_pts.append([0, y])
    for x in range(1, 5): tri_pts.append([x, 5-x])
    
    if len(set(map(tuple, tri_pts))) == 12:
        if len(set([_line_key(tri_pts[0], tri_pts[i]) for i in range(1, 12)])) > 1:
             update_best(tri_pts, _count_ordinary(tri_pts))

    # 4. Try "3x4 grid" with one point moved to create a 4th collinear point on a diagonal?
    # The 3x4 grid has diagonals of length 2 (e.g. (0,0)-(1,1)-(2,2)).
    # If we add a point (3,3), we get a diagonal of length 4.
    # But we need 12 points. The 3x4 grid already has 12.
    # Let's check the 4x3 grid (rotated).
    
    # 5. Randomized Local Search with Better Initialization
    # Start from the best found so far.
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    import random
    random.seed(42)
    
    max_iter = 2000
    for it in range(max_iter):
        if time.time() > deadline: break
        
        # Pick a random point to move
        idx = random.randint(0, 11)
        old_x, old_y = current_points[idx]
        
        # Try a few random moves
        improved = False
        for _ in range(20):
            # Move within a reasonable range to keep integers small
            nx = old_x + random.randint(-10, 10)
            ny = old_y + random.randint(-10, 10)
            
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
