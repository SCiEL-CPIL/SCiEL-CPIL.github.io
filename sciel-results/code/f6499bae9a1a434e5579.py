import time
import random
from math import gcd

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points (n=12)."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)

def _get_ordinary_line_info(points):
    """Return (ordinary_count, set of ordinary line keys, dict of line->points_on_line)."""
    n = len(points)
    line_points = {}  # key -> list of point indices
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = []
            line_points[key].append(i)
            if j not in line_points[key]:
                line_points[key].append(j)
    # Deduplicate indices per line
    for key in line_points:
        line_points[key] = list(set(line_points[key]))
    
    ordinary_keys = set()
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary_keys.add(key)
    
    return len(ordinary_keys), ordinary_keys, line_points

def _collinear_check(points):
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True

def _valid(points):
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        t = (x, y)
        if t in seen:
            return False
        seen.add(t)
    return not _collinear_check(points)

def _line_through_two(p1, p2):
    """Get normalized line coefficients (A, B, C) for line through p1, p2."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
    if g:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _find_integer_points_on_line(A, B, C, x_range, y_range):
    """Find integer points (x,y) with |x|,|y| <= bound on line Ax+By+C=0."""
    points = []
    if A == 0 and B == 0:
        return points
    if A == 0:
        # By + C = 0 => y = -C/B
        if B != 0 and C % B == 0:
            y = -C // B
            if abs(y) <= y_range:
                for x in range(-x_range, x_range + 1):
                    if abs(x) <= x_range:
                        points.append((x, y))
        return points
    if B == 0:
        # Ax + C = 0 => x = -C/A
        if A != 0 and C % A == 0:
            x = -C // A
            if abs(x) <= x_range:
                for y in range(-y_range, y_range + 1):
                    if abs(y) <= y_range:
                        points.append((x, y))
        return points
    
    # General case: Ax + By + C = 0
    # We need to find integer solutions. Use the fact that we can parametrize.
    # Find one particular solution using extended gcd
    g = gcd(abs(A), abs(B))
    if C % g != 0:
        return points
    
    A_r = A // g
    B_r = B // g
    C_r = C // g
    
    # Solve A_r * x + B_r * y = -C_r
    # Use extended Euclidean to find one solution
    def extended_gcd(a, b):
        if b == 0:
            return (a, 1, 0)
        g, x, y = extended_gcd(b, a % b)
        return (g, y, x - (a // b) * y)
    
    g2, x0, y0 = extended_gcd(A_r, B_r)
    # A_r * x0 + B_r * y0 = g2 = 1
    # So A_r * (x0 * (-C_r)) + B_r * (y0 * (-C_r)) = -C_r
    px = x0 * (-C_r)
    py = y0 * (-C_r)
    
    # General solution: x = px + B_r * t, y = py - A_r * t
    # Find range of t such that |x| <= x_range and |y| <= y_range
    t_min = max(-x_range - px, -y_range + py) // B_r if B_r > 0 else min(-x_range - px, -y_range + py) // B_r
    t_max = min(x_range - px, y_range - py) // B_r if B_r > 0 else max(x_range - px, y_range - py) // B_r
    
    # More careful bounds
    if B_r > 0:
        t_min = max((-x_range - px) // B_r, (-y_range + py) // B_r)
        t_max = min((x_range - px) // B_r, (y_range - py) // B_r)
    else:
        t_min = max((x_range - px) // B_r, (y_range - py) // B_r)
        t_max = min((-x_range - px) // B_r, (-y_range + py) // B_r)
    
    # Ensure t_min <= t_max
    if t_min > t_max:
        t_min, t_max = t_max, t_min
    
    # Limit the number of points we return
    max_points = 200
    step = max(1, (t_max - t_min + 1) // max_points)
    
    t = t_min
    while t <= t_max and len(points) < max_points:
        x = px + B_r * t
        y = py - A_r * t
        if abs(x) <= x_range and abs(y) <= y_range:
            points.append((x, y))
        t += step
    
    return points

def _try_line_completion(points, line_points, ordinary_keys, max_iters=50000):
    """
    Structural search: for each ordinary line, try to move one of its two points
    to a position on another line that already has 2+ points, reducing ordinary count.
    """
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    best = [p[:] for p in cur]
    best_score = cur_score
    
    rng = random.Random(12345)
    iters = 0
    stagnation = 0
    
    while iters < max_iters and time.time() < _DEADLINE - 8:
        iters += 1
        
        # Recompute line info periodically
        if iters % 100 == 0:
            cur_score, ordinary_keys, line_points = _get_ordinary_line_info(cur)
            if cur_score < best_score:
                best_score = cur_score
                best = [p[:] for p in cur]
                stagnation = 0
            else:
                stagnation += 1
            if stagnation > 20:
                break
        
        if not ordinary_keys:
            break
        
        # Pick a random ordinary line
        ord_list = list(ordinary_keys)
        line_key = rng.choice(ord_list)
        
        # Get the two points on this ordinary line
        pts_on_line = line_points.get(line_key, [])
        if len(pts_on_line) != 2:
            continue
        
        # Try to move one of these two points to be on a non-ordinary line
        # that already has 2+ other points (so the new line would have 3+)
        for move_idx in rng.sample(pts_on_line, 2):
            old_pt = cur[move_idx]
            
            # Find lines that have 2+ points (excluding the ordinary line)
            candidate_lines = []
            for lk, lp in line_points.items():
                if lk == line_key:
                    continue
                if len(lp) >= 2:
                    candidate_lines.append((lk, lp))
            
            if not candidate_lines:
                continue
            
            # Try moving to a random point on one of these lines
            for _ in range(10):
                lk, lp = rng.choice(candidate_lines)
                A, B, C = lk
                
                # Find integer points on this line near the old point
                x_range = max(abs(old_pt[0]) + 50, 100)
                y_range = max(abs(old_pt[1]) + 50, 100)
                x_range = min(x_range, 10**9)
                y_range = min(y_range, 10**9)
                
                int_pts = _find_integer_points_on_line(A, B, C, x_range, y_range)
                if not int_pts:
                    continue
                
                # Filter out points already in cur
                cur_set = set(tuple(p) for p in cur)
                valid_pts = [p for p in int_pts if tuple(p) not in cur_set]
                
                if not valid_pts:
                    continue
                
                new_pt = rng.choice(valid_pts)
                new_pts = [p[:] for p in cur]
                new_pts[move_idx] = [new_pt[0], new_pt[1]]
                
                if not _valid(new_pts):
                    continue
                
                new_score = _count_ordinary(new_pts)
                delta = new_score - cur_score
                
                # Accept if improvement, or with small probability if slight worsening
                if delta < 0 or (delta == 0 and rng.random() < 0.3) or (delta == 1 and rng.random() < 0.1):
                    cur = new_pts
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best = [p[:] for p in cur]
                        stagnation = 0
                    break
    
    return best, best_score

def _try_known_configs():
    """Try known good configurations and their variants."""
    configs = []
    
    # 4x3 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 3x4 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 2x6 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # 6x2 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # Cubic curve points: (t, t^2) for t in 0..11
    configs.append([[t, t*t] for t in range(12)])
    
    # Parabola: (t, t^2) for t in -5..4
    configs.append([[t, t*t] for t in range(-5, 5)])
    
    # 3x3 grid + 3 extra points
    base = [[x, y] for x in range(3) for y in range(3)]
    configs.append(base + [[3, 0], [3, 1], [3, 2]])
    configs.append(base + [[0, 3], [1, 3], [2, 3]])
    
    # Points on two parallel lines: 6 on each
    configs.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # 3 lines with 4 points each (Hessian-like configuration)
    # Line 1: y=0, x=0,1,2,3
    # Line 2: x=0, y=0,1,2,3  
    # Line 3: y=x, points (1,1),(2,2),(3,3) + one more
    configs.append([[0, 0], [1, 0], [2, 0], [3, 0],
                    [0, 1], [0, 2], [0, 3],
                    [1, 1], [2, 2], [3, 3],
                    [4, 4], [5, 5]])
    
    # Configuration based on complete quadrilateral + diagonal points
    # 4 lines, 6 intersection points, plus 6 more
    configs.append([[0, 0], [1, 0], [2, 0], [3, 0],
                    [0, 1], [1, 1], [2, 1], [3, 1],
                    [0, 2], [1, 2], [2, 2], [3, 2]])
    
    # Try a configuration with more collinearities
    # 4 points on x-axis, 4 on y-axis, 4 on diagonal
    configs.append([[0, 0], [1, 0], [2, 0], [3, 0],
                    [0, 1], [0, 2], [0, 3],
                    [1, 1], [2, 2], [3, 3], [4, 4]])
    
    return configs

def _local_search_sa(points, max_iters=200000, start_temp=10.0, end_temp=0.1):
    """Simulated annealing local search to minimize ordinary lines."""
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    best = [p[:] for p in cur]
    best_score = cur_score
    
    def get_pool(pts):
        pool = set()
        for p in pts:
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    pool.add((p[0] + dx, p[1] + dy))
        # Extend lines through pairs
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy)) if (dx or dy) else 1
                if g:
                    dx //= g
                    dy //= g
                for k in range(-3, 4):
                    if k == 0:
                        continue
                    pool.add((x1 + k*dx, y1 + k*dy))
                    pool.add((x2 + k*dx, y2 + k*dy))
        return list(pool)
    
    pool = get_pool(cur)
    iters = 0
    rng = random.Random(42)
    
    while iters < max_iters and time.time() < _DEADLINE - 5:
        iters += 1
        temp = start_temp * (end_temp / start_temp) ** (iters / max_iters)
        
        idx = rng.randint(0, 11)
        old = cur[idx]
        cur_set = set(tuple(p) for p in cur)
        
        # Try several random candidates
        n_try = min(15, len(pool))
        candidates = rng.sample(pool, n_try)
        improved = False
        
        for cand in candidates:
            if cand in cur_set:
                continue
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            
            delta = new_score - cur_score
            if delta < 0 or (temp > 0 and rng.random() < min(1.0, 1.0 / (1.0 + delta / temp))):
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                improved = True
                if cur_score < best_score:
                    best = [p[:] for p in cur]
                    best_score = cur_score
                break
        
        if not improved and iters % 500 == 0:
            pool = get_pool(cur)
            
    return best, best_score

def construct():
    best_pts = None
    best_score = float('inf')
    
    # Try known configurations
    for cfg in _try_known_configs():
        if not _valid(cfg):
            continue
        score = _count_ordinary(cfg)
        if score < best_score:
            best_score = score
            best_pts = [p[:] for p in cfg]
    
    # If we haven't found a good config, use 4x3 grid as fallback
    if best_pts is None:
        best_pts = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary(best_pts)
    
    # Run line-completion structural search (primary method)
    if time.time() < _DEADLINE - 20:
        _, _, line_points = _get_ordinary_line_info(best_pts)
        _, ordinary_keys, _ = _get_ordinary_line_info(best_pts)
        pts, score = _try_line_completion(best_pts, line_points, ordinary_keys, max_iters=30000)
        if score < best_score:
            best_pts = pts
            best_score = score
    
    # Run simulated annealing from best config
    if time.time() < _DEADLINE - 15:
        pts, score = _local_search_sa(best_pts, max_iters=100000)
        if score < best_score:
            best_pts = pts
            best_score = score
    
    # Run line-completion from other starting points if time allows
    if time.time() < _DEADLINE - 20:
        for cfg in _try_known_configs()[:5]:
            if not _valid(cfg):
                continue
            if time.time() >= _DEADLINE - 10:
                break
            _, _, lp = _get_ordinary_line_info(cfg)
            _, ok, _ = _get_ordinary_line_info(cfg)
            pts, score = _try_line_completion(cfg, lp, ok, max_iters=20000)
            if score < best_score:
                best_pts = pts
                best_score = score
    
    # Final SA polish if time allows
    if time.time() < _DEADLINE - 10:
        pts, score = _local_search_sa(best_pts, max_iters=50000)
        if score < best_score:
            best_pts = pts
            best_score = score
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))
