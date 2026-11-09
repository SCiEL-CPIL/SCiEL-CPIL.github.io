import time
import math
from itertools import combinations

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    a = dy
    b = -dx
    c = -a * x1 - b * y1
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _count_ordinary(points):
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = 0
            lines[key] += 1
    ordinary = 0
    for key, cnt in lines.items():
        if cnt == 2:
            ordinary += 1
    return ordinary

def _valid(points):
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if not isinstance(p, list) or len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    if cross == 0:
        for i in range(3, 12):
            xi, yi = points[i]
            cross_i = (x2 - x1) * (yi - y1) - (y2 - y1) * (xi - x1)
            if cross_i != 0:
                return True
        return False
    return True

def _build_incidence(points):
    """
    Build incidence structure:
    - pair_to_line: dict mapping (i,j) -> line_key
    - line_to_pairs: dict mapping line_key -> list of (i,j) pairs
    - line_to_points: dict mapping line_key -> set of point indices
    """
    n = len(points)
    pair_to_line = {}
    line_to_pairs = {}
    line_to_points = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            pair_to_line[(i, j)] = key
            if key not in line_to_pairs:
                line_to_pairs[key] = []
            line_to_pairs[key].append((i, j))
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    
    return pair_to_line, line_to_pairs, line_to_points

def _evaluate_move(points, idx, new_pos, pair_to_line, line_to_pairs, line_to_points):
    """
    Quickly evaluate the ordinary line count after moving point idx to new_pos.
    Returns the new count, or None if invalid.
    """
    if abs(new_pos[0]) > 10**9 or abs(new_pos[1]) > 10**9:
        return None
    
    # Check collision
    for j in range(12):
        if j != idx and points[j][0] == new_pos[0] and points[j][1] == new_pos[1]:
            return None
    
    # Check not all collinear
    new_points = [p[:] for p in points]
    new_points[idx] = list(new_pos)
    if not _valid(new_points):
        return None
    
    # Recompute only the lines affected by moving point idx
    # Lines that change: all lines through idx (66 - 55 = 11 lines)
    # We need to recompute the line membership for lines involving idx
    
    n = 12
    others = [j for j in range(n) if j != idx]
    
    # Build new line structure from scratch (faster than incremental for 12 points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(new_points[i], new_points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    ordinary = 0
    for key, members in lines.items():
        if len(members) == 2:
            ordinary += 1
    
    return ordinary

def _grid_search(sx, sy, deadline):
    """
    Search over small grids for 12-point configurations with few ordinary lines.
    """
    best_count = float('inf')
    best_points = None
    
    # Generate all points in the grid
    grid_points = []
    for x in range(-sx, sx + 1):
        for y in range(-sy, sy + 1):
            grid_points.append([x, y])
    
    n_grid = len(grid_points)
    
    # If grid is small enough, enumerate all 12-combinations
    if n_grid <= 30:
        for combo in combinations(range(n_grid), 12):
            if time.time() > deadline:
                break
            pts = [grid_points[i] for i in combo]
            if not _valid(pts):
                continue
            c = _count_ordinary(pts)
            if c < best_count:
                best_count = c
                best_points = [p[:] for p in pts]
    else:
        # For larger grids, use random sampling
        import random
        random.seed(42)
        for _ in range(50000):
            if time.time() > deadline:
                break
            combo = random.sample(range(n_grid), 12)
            pts = [grid_points[i] for i in combo]
            if not _valid(pts):
                continue
            c = _count_ordinary(pts)
            if c < best_count:
                best_count = c
                best_points = [p[:] for p in pts]
    
    return best_points, best_count

def _structured_initializations():
    """
    Generate several structured initial configurations known to have high collinearity.
    """
    inits = []
    
    # 1. 3x4 grid
    pts = [[x, y] for x in range(3) for y in range(4)]
    inits.append(pts)
    
    # 2. 4x3 grid
    pts = [[x, y] for x in range(4) for y in range(3)]
    inits.append(pts)
    
    # 3. 2x6 grid
    pts = [[x, y] for x in range(2) for y in range(6)]
    inits.append(pts)
    
    # 4. 6x2 grid
    pts = [[x, y] for x in range(6) for y in range(2)]
    inits.append(pts)
    
    # 5. 3x3 grid plus 3 extra points
    pts = [[x, y] for x in range(3) for y in range(3)]
    pts.append([0, 3])
    pts.append([1, 3])
    pts.append([2, 3])
    inits.append(pts)
    
    # 6. Hexagonal pattern
    pts = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1], [2, 0], [-2, 0], [0, 2]]
    inits.append(pts)
    
    # 7. Diamond pattern
    pts = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [2, 0], [-2, 0], [0, 2], [0, -2], [1, 1], [-1, 1], [1, -1]]
    inits.append(pts)
    
    # 8. 12 points on a conic-like structure (parabola y=x^2)
    pts = [[x, x*x] for x in range(12)]
    inits.append(pts)
    
    # 9. Mixed: some on lines, some off
    pts = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0],  # 6 on x-axis
           [0, 1], [0, 2], [0, 3], [0, 4],  # 4 on y-axis
           [1, 1], [2, 2]]  # 2 on diagonal
    inits.append(pts)
    
    # 10. 3 lines of 4 points each (but overlapping at intersection)
    pts = [[0, 0], [1, 0], [2, 0], [3, 0],  # line 1
           [0, 0], [0, 1], [0, 2], [0, 3],  # line 2 (shares origin)
           [0, 0], [1, 1], [2, 2], [3, 3]]  # line 3 (shares origin)
    # Remove duplicates
    unique_pts = []
    seen = set()
    for p in pts:
        if p not in seen:
            unique_pts.append(p)
            seen.add(p)
    if len(unique_pts) == 12:
        inits.append(unique_pts)
    
    return inits

def _local_search(points, deadline):
    """
    Fast local search using pre-computed line structure.
    """
    current_points = [p[:] for p in points]
    current_count = _count_ordinary(current_points)
    
    stuck_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # Try moving each point to positions that create new collinearities
        for i in range(12):
            if time.time() > deadline:
                break
            
            x, y = current_points[i]
            others = [j for j in range(12) if j != i]
            
            # Generate candidate positions
            candidates = set()
            
            # Small perturbations
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                        candidates.add((nx, ny))
            
            # Points on lines through pairs of other points
            for j in others:
                for k in others:
                    if j >= k:
                        continue
                    xj, yj = current_points[j]
                    xk, yk = current_points[k]
                    dx = xk - xj
                    dy = yk - yj
                    g = math.gcd(math.gcd(abs(dx), abs(dy)), 1)
                    if g == 0:
                        continue
                    dx_n = dx // g
                    dy_n = dy // g
                    
                    # Find t values that give integer points near (x, y)
                    if dx_n != 0:
                        t_approx = (x - xj) / dx_n
                    elif dy_n != 0:
                        t_approx = (y - yj) / dy_n
                    else:
                        continue
                    
                    for t in range(int(t_approx) - 2, int(t_approx) + 3):
                        nx = xj + t * dx_n
                        ny = yj + t * dy_n
                        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                            is_same = False
                            for m in range(12):
                                if m != i and current_points[m][0] == nx and current_points[m][1] == ny:
                                    is_same = True
                                    break
                            if not is_same:
                                candidates.add((nx, ny))
            
            # Evaluate candidates
            best_move = None
            best_move_count = current_count
            
            for nx, ny in candidates:
                if time.time() > deadline:
                    break
                new_count = _evaluate_move(current_points, i, (nx, ny), None, None, None)
                if new_count is not None and new_count < best_move_count:
                    best_move_count = new_count
                    best_move = (i, nx, ny, new_count)
            
            if best_move:
                idx, nx, ny, new_count = best_move
                current_points[idx] = [nx, ny]
                current_count = new_count
                improved = True
                stuck_count = 0
                
                if current_count <= 6:
                    return current_points
        
        if not improved:
            stuck_count += 1
            
            # Random restart
            if stuck_count >= 2:
                import random
                random.seed(int(time.time() * 1000) % 10000)
                new_pts = []
                for _ in range(12):
                    ri = random.randint(-10, 10)
                    rj = random.randint(-10, 10)
                    new_pts.append([ri, rj])
                if _valid(new_pts):
                    cnt = _count_ordinary(new_pts)
                    if cnt < current_count:
                        current_points = new_pts
                        current_count = cnt
                        stuck_count = 0
                else:
                    break
    
    return current_points

def construct():
    deadline = time.time() + 115
    
    # Phase 1: Grid search for small grids
    best_points = None
    best_count = float('inf')
    
    # Try small grids
    for sx, sy in [(2, 2), (3, 2), (2, 3), (3, 3), (4, 2), (2, 4)]:
        if time.time() > deadline:
            break
        pts, cnt = _grid_search(sx, sy, deadline)
        if pts is not None and cnt < best_count:
            best_count = cnt
            best_points = pts
    
    # Phase 2: Structured initializations
    if best_points is None:
        best_points = _structured_initializations()[0]
        best_count = _count_ordinary(best_points)
    else:
        for init in _structured_initializations():
            if time.time() > deadline:
                break
            if not _valid(init):
                continue
            c = _count_ordinary(init)
            if c < best_count:
                best_count = c
                best_points = [p[:] for p in init]
    
    # Phase 3: Local search from best configuration
    result = _local_search(best_points, deadline)
    
    return result
