"""
Minimize ordinary lines for 12 integer points.
Strategy:
1. Brute-force search over small grids with local improvements
2. Try known geometric constructions (grids, finite field constructions)
3. Local search with larger moves from best found
"""
import time
import random
import math
from itertools import combinations, product

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return None
        
        # Normalize direction vector
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
            
        g = math.gcd(abs(dx), abs(dy))
        if g > 0:
            dx //= g
            dy //= g
            
        c = dy * x1 - dx * y1
        return (dx, dy, c)
    
    # Find all unique lines
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = line_key(points[i], points[j])
            if key is not None:
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
    
    # Count ordinary lines (exactly 2 points)
    ordinary = 0
    for key, pts_on_line in lines.items():
        if len(pts_on_line) == 2:
            ordinary += 1
    
    return ordinary

def construct():
    start_time = time.time()
    deadline = time.time() + 145.0  # 5s safety margin
    
    best_points = None
    best_count = float('inf')
    
    def update_best(pts, cnt):
        nonlocal best_points, best_count
        if cnt < best_count:
            best_count = cnt
            best_points = [p[:] for p in pts]
    
    def is_valid(pts):
        """Check if points are valid: 12 distinct integer points, |x|,|y| <= 10^9, not all collinear."""
        if len(pts) != 12:
            return False
        # Check distinct
        tuples = [tuple(p) for p in pts]
        if len(set(tuples)) != 12:
            return False
        # Check bounds
        for x, y in tuples:
            if abs(x) > 10**9 or abs(y) > 10**9:
                return False
        # Check not all collinear
        if len(set(tuples)) >= 3:
            x1, y1 = tuples[0]
            x2, y2 = tuples[1]
            dx = x2 - x1
            dy = y2 - y1
            for i in range(2, 12):
                xi, yi = tuples[i]
                if (xi - x1) * dy - (yi - y1) * dx != 0:
                    return True
            return False
        return False
    
    # Strategy 1: Try 3x4 grid and variations
    # 3x4 grid
    grid_34 = [[x, y] for x in range(4) for y in range(3)]
    if is_valid(grid_34):
        cnt = count_ordinary_lines(grid_34)
        update_best(grid_34, cnt)
    
    # 4x3 grid
    grid_43 = [[x, y] for x in range(3) for y in range(4)]
    if is_valid(grid_43):
        cnt = count_ordinary_lines(grid_43)
        update_best(grid_43, cnt)
    
    # Strategy 2: Brute force over small grids
    # Try all subsets of 12 points from small grids
    for max_coord in [3, 4, 5]:
        if time.time() > deadline - 30:
            break
        # Generate all points in the grid
        all_pts = [[x, y] for x in range(-max_coord, max_coord + 1) for y in range(-max_coord, max_coord + 1)]
        if len(all_pts) < 12:
            continue
        
        # Use random sampling for larger grids
        if len(all_pts) > 1000:
            continue
            
        # Try combinations - this is expensive, so use random sampling
        num_trials = 5000
        for _ in range(num_trials):
            if time.time() > deadline - 30:
                break
            idxs = random.sample(range(len(all_pts)), 12)
            pts = [all_pts[i] for i in idxs]
            if not is_valid(pts):
                continue
            cnt = count_ordinary_lines(pts)
            update_best(pts, cnt)
            if best_count <= 6:
                break
        if best_count <= 6:
            break
    
    # Strategy 3: Local search from best found
    if best_points is not None:
        def local_search(pts, max_iter=10000):
            nonlocal best_points, best_count
            cur = [p[:] for p in pts]
            cur_cnt = count_ordinary_lines(cur)
            improved = True
            iterations = 0
            while improved and iterations < max_iter and time.time() < deadline - 5:
                improved = False
                iterations += 1
                i = random.randint(0, 11)
                # Try several random moves
                for _ in range(20):
                    dx = random.randint(-5, 5)
                    dy = random.randint(-5, 5)
                    if dx == 0 and dy == 0:
                        continue
                    old_x, old_y = cur[i]
                    new_x, new_y = old_x + dx, old_y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    cur[i] = [new_x, new_y]
                    if not is_valid(cur):
                        cur[i] = [old_x, old_y]
                        continue
                    new_cnt = count_ordinary_lines(cur)
                    if new_cnt < cur_cnt:
                        cur_cnt = new_cnt
                        improved = True
                        update_best(cur, cur_cnt)
                        if best_count <= 6:
                            return cur
                        break
                    else:
                        cur[i] = [old_x, old_y]
            return cur
        
        # Run local search multiple times
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            local_search(best_points, max_iter=2000)
            if best_count <= 6:
                break
    
    # Strategy 4: Try specific known configurations
    # Points on few lines: e.g., 4 points on 3 lines
    # Try: 3 lines with 4 points each, sharing intersections
    # Line 1: y = 0, points at x = 0, 1, 2, 3
    # Line 2: y = 1, points at x = 0, 1, 2, 3
    # Line 3: x = 0, points at y = 0, 1, 2, 3 (but y=0 and y=1 already used)
    # So we have (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (0,3)
    # That's 10 points, need 2 more
    
    # Try a different configuration: 4 points on each of 3 lines
    # L1: (0,0), (1,0), (2,0), (3,0)
    # L2: (0,1), (1,1), (2,1), (3,1)  
    # L3: (0,2), (0,3), (0,4), (0,5)
    # But these share (0,0) and (0,1), so total = 4+4+4-2 = 10, need 2 more
    
    # Add 2 more points to create more collinearities
    # Try adding points that create diagonal lines
    config = [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (0,3), (1,2), (2,3)]
    if is_valid(config):
        cnt = count_ordinary_lines(config)
        update_best(config, cnt)
    
    # Try another configuration
    config2 = [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (3,2)]
    if is_valid(config2):
        cnt = count_ordinary_lines(config2)
        update_best(config2, cnt)
    
    # Strategy 5: More aggressive random search with larger coordinate range
    for box_size in [10, 20, 50]:
        if time.time() > deadline - 5:
            break
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            pts = [[random.randint(-box_size, box_size), random.randint(-box_size, box_size)] for _ in range(12)]
            if not is_valid(pts):
                continue
            cnt = count_ordinary_lines(pts)
            update_best(pts, cnt)
            if best_count <= 6:
                break
        if best_count <= 6:
            break
    
    if best_points is None:
        return [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
