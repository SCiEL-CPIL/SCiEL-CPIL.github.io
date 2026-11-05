import time
from itertools import combinations
from math import gcd

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly two of the points.
    """
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        g = _gcd(_gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
            
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
            
        return (A, B, C)

    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            lines[key] = lines.get(key, 0) + 1
            
    ordinary = 0
    for key, pair_count in lines.items():
        if pair_count == 1:
            ordinary += 1
            
    return ordinary

def _count_ordinary_fast(points):
    """Fast count using vectorized approach with tuples."""
    n = len(points)
    if n < 3:
        return 0
    
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = _gcd(_gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    return sum(1 for v in line_counts.values() if v == 1)

def _generate_integer_points_on_line(p1, p2, max_coord=10**9):
    """
    Generate all integer points on the line through p1 and p2 within bounds.
    The line through (x1,y1) and (x2,y2) has direction (dx, dy).
    Integer points on this line are p1 + t*(dx/g, dy/g) for integer t,
    where g = gcd(dx, dy).
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return [(x1, y1)]
    
    g = _gcd(abs(dx), abs(dy))
    if g == 0:
        return [(x1, y1)]
    step_x = dx // g
    step_y = dy // g
    
    points = set()
    # Find range of t such that |x1 + t*step_x| <= max_coord and |y1 + t*step_y| <= max_coord
    t_min = -10**6
    t_max = 10**6
    
    # Tighten bounds
    if step_x != 0:
        t_min_x = (-(max_coord) - x1) // step_x if step_x > 0 else (max_coord - x1) // step_x
        t_max_x = (max_coord - x1) // step_x if step_x > 0 else (-(max_coord) - x1) // step_x
        t_min = max(t_min, int(min(t_min_x, t_max_x)) - 1)
        t_max = min(t_max, int(max(t_min_x, t_max_x)) + 1)
    
    if step_y != 0:
        t_min_y = (-(max_coord) - y1) // step_y if step_y > 0 else (max_coord - y1) // step_y
        t_max_y = (max_coord - y1) // step_y if step_y > 0 else (-(max_coord) - y1) // step_y
        t_min = max(t_min, int(min(t_min_y, t_max_y)) - 1)
        t_max = min(t_max, int(max(t_min_y, t_max_y)) + 1)
    
    # Limit the range to avoid too many points
    t_min = max(t_min, -100)
    t_max = min(t_max, 100)
    
    for t in range(t_min, t_max + 1):
        px = x1 + t * step_x
        py = y1 + t * step_y
        if abs(px) <= max_coord and abs(py) <= max_coord:
            points.add((px, py))
    
    return list(points)

def _generate_line_candidates(points, exclude_idx=None):
    """
    Generate candidate positions for a new point by finding integer points
    on lines through pairs of existing points.
    """
    candidates = set()
    n = len(points)
    
    for i in range(n):
        if i == exclude_idx:
            continue
        for j in range(i + 1, n):
            if j == exclude_idx:
                continue
            pts = _generate_integer_points_on_line(points[i], points[j], max_coord=10**9)
            for pt in pts:
                candidates.add(pt)
    
    return list(candidates)

def _local_search(points, deadline, max_iter=200):
    """Local search: try moving individual points to reduce ordinary lines.
    Uses line-based candidates primarily."""
    n = len(points)
    current_score = _count_ordinary_fast(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    iters = 0
    
    while iters < max_iter and time.time() < deadline:
        iters += 1
        improved = False
        
        # Try moving each point
        for i in range(n):
            x, y = best_points[i]
            
            # Generate line-based candidates from other points
            other_points = [best_points[j] for j in range(n) if j != i]
            line_cands = _generate_line_candidates(other_points, exclude_idx=None)
            
            # Filter to reasonable candidates (not too far, not duplicates)
            existing = set(tuple(p) for p in best_points)
            valid_cands = []
            for cand in line_cands:
                cx, cy = cand
                if (cx, cy) in existing:
                    continue
                # Prefer candidates close to current position
                dist = abs(cx - x) + abs(cy - y)
                if dist <= 200:
                    valid_cands.append((dist, cand))
            
            # Sort by distance, try closest first
            valid_cands.sort()
            
            for dist, (nx, ny) in valid_cands:
                new_pt = [nx, ny]
                new_points = best_points[:]
                new_points[i] = new_pt
                new_score = _count_ordinary_fast(new_points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = new_points
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Try small grid moves as fallback
            for i in range(n):
                x, y = best_points[i]
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if abs(nx) > 10**9 or abs(ny) > 10**9:
                            continue
                        new_pt = [nx, ny]
                        if any(new_pt == p for p in best_points):
                            continue
                        new_points = best_points[:]
                        new_points[i] = new_pt
                        new_score = _count_ordinary_fast(new_points)
                        if new_score < best_score:
                            best_score = new_score
                            best_points = new_points
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Known good configurations to try - designed for many collinearities
    configs = []
    
    # Config 1: 3x4 grid (12 points) - has many 3 and 4 point lines
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # Config 2: 4x3 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # Config 3: 2x6 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # Config 4: 6x2 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # Config 5: 3x3 grid + 3 extensions on lines
    base33 = [[x, y] for x in range(3) for y in range(3)]
    configs.append(base33 + [[3, 0], [0, 3], [3, 3]])
    
    # Config 6: 3x3 grid + diagonal extensions
    configs.append(base33 + [[3, 1], [1, 3], [4, 2]])
    
    # Config 7: 3x3 grid + other extensions
    configs.append(base33 + [[-1, 1], [1, -1], [4, 2]])
    
    # Config 8: Symmetric with center
    configs.append([[-1,-1],[0,-1],[1,-1],
                   [-1,0],[0,0],[1,0],
                   [-1,1],[0,1],[1,1],
                   [2,0],[0,2],[0,-2]])
    
    # Config 9: 4-point lines structure
    configs.append([[0,0],[1,0],[2,0],[3,0],
                   [0,1],[1,1],[2,1],[3,1],
                   [0,2],[1,2],[2,2],[3,2]])
    
    # Config 10: Mixed structure
    configs.append([[0,0],[1,0],[2,0],[3,0],
                   [0,1],[0,2],[0,3],
                   [1,1],[2,2],
                   [4,1],[1,4]])
    
    # Config 11: 12 points on few lines - try 4 lines of 3
    # Line 1: (0,0),(1,0),(2,0)
    # Line 2: (0,1),(1,1),(2,1)  
    # Line 3: (0,2),(1,2),(2,2)
    # Line 4: (3,0),(3,1),(3,2) - vertical line
    configs.append([[0,0],[1,0],[2,0],
                   [0,1],[1,1],[2,1],
                   [0,2],[1,2],[2,2],
                   [3,0],[3,1],[3,2]])
    
    # Config 12: 3 lines of 4
    configs.append([[0,0],[1,0],[2,0],[3,0],
                   [0,1],[1,1],[2,1],[3,1],
                   [0,2],[1,2],[2,2],[3,2]])
    
    # Config 13: 2 lines of 4 + 2 lines of 2
    configs.append([[0,0],[1,0],[2,0],[3,0],
                   [0,1],[1,1],[2,1],[3,1],
                   [4,2],[5,2],
                   [4,3],[5,3]])
    
    # Config 14: Hexagonal-like arrangement
    configs.append([[0,0],[1,0],[2,0],
                   [0,1],[1,1],[2,1],
                   [0,2],[1,2],[2,2],
                   [3,0],[3,2],[4,1]])
    
    # Config 15: 12 points with 4-point lines in multiple directions
    configs.append([[0,0],[1,0],[2,0],[3,0],
                   [0,1],[1,1],[2,1],[3,1],
                   [0,2],[1,2],[2,2],[3,2]])
    
    # Evaluate all initial configs
    best_points = None
    best_score = float('inf')
    
    for cfg in configs:
        if len(set(map(tuple, cfg))) != 12:
            continue
        score = _count_ordinary_fast(cfg)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in cfg]
    
    # Local search from the best initial config
    if best_points is not None:
        best_points, best_score = _local_search(best_points, deadline, max_iter=150)
    
    # If we still have time, try local search from other configs
    if time.time() < deadline - 5:
        for cfg in configs:
            if len(set(map(tuple, cfg))) != 12:
                continue
            if best_points and cfg == best_points:
                continue
            score = _count_ordinary_fast(cfg)
            if score < best_score + 2:  # Only if close to best
                pts, sc = _local_search([p[:] for p in cfg], deadline, max_iter=100)
                if sc < best_score:
                    best_score = sc
                    best_points = pts
    
    # Final check
    if best_points is None:
        best_points = [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]]
    
    return best_points
