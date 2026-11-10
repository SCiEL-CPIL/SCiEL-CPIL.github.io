import time
import random
from math import gcd

def _line_key(p1, p2):
    """Return normalized line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    # Standardize sign
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def count_ordinary_lines(points):
    """Count lines containing exactly 2 points."""
    n = len(points)
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    return sum(1 for pts in line_pts.values() if len(pts) == 2)

def _build_line_map(points):
    """Build line -> set of point indices map."""
    n = len(points)
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    return line_pts

def _count_ordinary_from_line_map(line_pts):
    return sum(1 for pts in line_pts.values() if len(pts) == 2)

def _boroczky_seed():
    """
    Böröczky-inspired seed: 6 points on a 'hexagon' and 6 dual points.
    Using rational coordinates scaled to integers.
    
    Regular hexagon vertices (radius 2): (2,0), (1,√3), (-1,√3), (-2,0), (-1,-√3), (1,-√3)
    The dual (intersection of opposite sides) gives points like (0, 2√3), etc.
    
    For integer approximation, use a parametric family.
    Here we use a known good structured seed: 4 points on each of 3 lines through origin,
    plus adjustments, then let local search fix it.
    
    Actually, let's use a concrete Böröczky-type construction:
    Points on parabola y=x^2: (0,0), (1,1), (2,4), (3,9), (4,16), (5,25)
    Points on line y=0: (-6,0), (-5,0), (-4,0), (-3,0), (-2,0), (-1,0)
    This gives many 3-point lines from the parabola structure.
    """
    # Parabola points
    para = [(i, i*i) for i in range(6)]
    # Line points (on x-axis, offset to avoid overlap)
    line_pts = [(-i-1, 0) for i in range(6)]
    return para + line_pts

def _grid_seed():
    """4x3 grid as fallback seed."""
    return [(x, y) for x in range(4) for y in range(3)]

def _hexagon_seed():
    """Approximate hexagon + dual with integer coords."""
    # Hexagon: (3,0), (1,2), (-2,2), (-3,0), (-1,-2), (2,-2)
    hex_pts = [(3, 0), (1, 2), (-2, 2), (-3, 0), (-1, -2), (2, -2)]
    # Dual-like points: intersections extended
    dual_pts = [(0, 4), (-4, 1), (-4, -1), (0, -4), (4, -1), (4, 1)]
    return hex_pts + dual_pts

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin (evaluator kills at 150s)
    
    def elapsed_ok():
        return time.time() < deadline
    
    # Try multiple seeds
    seeds = [
        _boroczky_seed(),
        _hexagon_seed(),
        _grid_seed(),
        # 11-on-line + 1-off (known 11 ordinary lines)
        [(i, 0) for i in range(11)] + [(0, 1)],
        # Two parallel lines with 6 each
        [(i, 0) for i in range(6)] + [(i, 1) for i in range(6)],
    ]
    
    best_points = None
    best_score = float('inf')
    
    # Evaluate all seeds
    for seed in seeds:
        if not elapsed_ok():
            break
        score = count_ordinary_lines(seed)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in seed]
    
    # Local search with fast incremental updates
    def try_improve(points, max_rounds=5000):
        n = len(points)
        current_score = count_ordinary_lines(points)
        line_map = _build_line_map(points)
        
        for round_num in range(max_rounds):
            if not elapsed_ok():
                break
            
            improved = False
            for i in range(n):
                if not elapsed_ok():
                    break
                cx, cy = points[i]
                
                # Try several random moves
                for _ in range(20):
                    if not elapsed_ok():
                        break
                    nx = cx + random.randint(-5, 5)
                    ny = cy + random.randint(-5, 5)
                    
                    if nx == cx and ny == cy:
                        continue
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    
                    # Check distinctness
                    dup = False
                    for j in range(n):
                        if j != i and points[j][0] == nx and points[j][1] == ny:
                            dup = True
                            break
                    if dup:
                        continue
                    
                    # Temporarily move
                    old_x, old_y = cx, cy
                    points[i] = [nx, ny]
                    
                    new_score = count_ordinary_lines(points)
                    
                    if new_score < current_score:
                        current_score = new_score
                        line_map = _build_line_map(points)
                        improved = True
                        break
                    else:
                        points[i] = [old_x, old_y]
                
                if improved:
                    break
            
            if not improved:
                break
        
        return points, current_score
    
    # Run local search from best seed
    if best_points is not None and elapsed_ok():
        improved_pts, improved_score = try_improve([list(p) for p in best_points])
        if improved_score < best_score:
            best_score = improved_score
            best_points = improved_pts
    
    # Try local search from other seeds too if time permits
    for seed in seeds:
        if not elapsed_ok():
            break
        if seed == best_points:
            continue
        s_score = count_ordinary_lines(seed)
        if s_score >= best_score:
            continue
        improved_pts, improved_score = try_improve([list(p) for p in seed])
        if improved_score < best_score:
            best_score = improved_score
            best_points = improved_pts
    
    # Final verification
    result = [[int(p[0]), int(p[1])] for p in best_points]
    
    # Ensure distinctness
    seen = set()
    new_result = []
    for p in result:
        tp = (p[0], p[1])
        if tp not in seen:
            seen.add(tp)
            new_result.append(p)
    result = new_result
    
    # Ensure exactly 12 points
    while len(result) < 12:
        result.append([random.randint(-10**9, 10**9), random.randint(-10**9, 10**9)])
    result = result[:12]
    
    return result
