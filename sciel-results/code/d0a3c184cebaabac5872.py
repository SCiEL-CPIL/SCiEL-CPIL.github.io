import time
import random
import itertools
from collections import defaultdict

def _count_lines_exactly_4(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    count = 0
    # For each pair of points, determine the line, count how many points are on it
    # Use a set of normalized line keys to avoid double counting
    seen_lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize direction
            g = _gcd(abs(dx), abs(dy))
            if g > 0:
                dx_n = dx // g
                dy_n = dy // g
            else:
                continue
            # Normalize sign
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            # Line: point (x1,y1) with direction (dx_n, dy_n)
            # A point (x,y) is on this line if (x-x1)*dy_n == (y-y1)*dx_n
            # Use a canonical representation: the line through (x1,y1) with direction (dx_n,dy_n)
            # Canonical: use the point with smallest (x,y) on the line? Or use a hash.
            # Better: represent line as (a, b, c) where ax + by = c, with (a,b) normalized
            # For direction (dx_n, dy_n), normal is (-dy_n, dx_n)
            a = -dy_n
            b = dx_n
            c = a * x1 + b * y1
            # Normalize (a, b, c): make gcd(a,b,c) = 1, and a > 0 or (a==0 and b > 0)
            g2 = _gcd(_gcd(abs(a), abs(b)), abs(c))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            if key in seen_lines:
                continue
            # Count points on this line
            cnt = 0
            for k in range(n):
                xk, yk = points[k]
                if a * xk + b * yk == c:
                    cnt += 1
            if cnt == 4:
                count += 1
            seen_lines.add(key)
    return count

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _score(points):
    """Score = number of lines with exactly 4 points. Penalize if 5+ collinear."""
    n = len(points)
    # Check no 5 collinear
    seen_lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = _gcd(abs(dx), abs(dy))
            if g > 0:
                dx_n = dx // g
                dy_n = dy // g
            else:
                continue
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            a = -dy_n
            b = dx_n
            c = a * x1 + b * y1
            g2 = _gcd(_gcd(abs(a), abs(b)), abs(c))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            if key in seen_lines:
                continue
            cnt = 0
            for k in range(n):
                xk, yk = points[k]
                if a * xk + b * yk == c:
                    cnt += 1
            if cnt >= 5:
                return -1000  # Invalid
            seen_lines.add(key)
    return _count_lines_exactly_4(points)

def _local_search(points, deadline):
    """Local search: try moving individual points to improve score."""
    best_points = [p[:] for p in points]
    best_score = _score(best_points)
    n = 16
    coord_range = range(-50, 51)
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            bx, by = best_points[i]
            # Try moving point i to nearby positions
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    nx, ny = bx + dx, by + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    # Check distinctness
                    if any(best_points[j] == [nx, ny] for j in range(n) if j != i):
                        continue
                    new_points = [p[:] for p in best_points]
                    new_points[i] = [nx, ny]
                    s = _score(new_points)
                    if s > best_score:
                        best_score = s
                        best_points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try random restarts within the neighborhood
            for _ in range(100):
                if time.time() > deadline:
                    break
                i = random.randint(0, n - 1)
                bx, by = best_points[i]
                nx = bx + random.randint(-20, 20)
                ny = by + random.randint(-20, 20)
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if any(best_points[j] == [nx, ny] for j in range(n) if j != i):
                    continue
                new_points = [p[:] for p in best_points]
                new_points[i] = [nx, ny]
                s = _score(new_points)
                if s > best_score:
                    best_score = s
                    best_points = new_points
                    improved = True
                    break
            if not improved:
                break
    return best_points, best_score

def construct():
    deadline = time.time() + 140  # Safety margin
    
    # Start from a 4x4 grid (known to give 8)
    base_points = [[x, y] for x in range(4) for y in range(4)]
    
    best_overall = [p[:] for p in base_points]
    best_score_overall = _score(best_overall)
    
    # Try multiple initial configurations
    initial_configs = []
    
    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 5x5 grid subset (pick 16 from 25)
    for _ in range(20):
        grid_5x5 = [[x, y] for x in range(5) for y in range(5)]
        random.shuffle(grid_5x5)
        initial_configs.append(grid_5x5[:16])
    
    # 6x4 grid subset
    for _ in range(20):
        grid_6x4 = [[x, y] for x in range(6) for y in range(4)]
        random.shuffle(grid_6x4)
        initial_configs.append(grid_6x4[:16])
    
    # 5x4 grid + some extras
    for _ in range(10):
        grid_5x4 = [[x, y] for x in range(5) for y in range(4)]
        random.shuffle(grid_5x4)
        subset = grid_5x4[:12]
        # Add 4 more random points
        extra = []
        while len(extra) < 4:
            px = random.randint(-3, 8)
            py = random.randint(-3, 8)
            p = [px, py]
            if p not in subset and p not in extra:
                extra.append(p)
        initial_configs.append(subset + extra)
    
    # Random configurations
    for _ in range(30):
        pts = []
        while len(pts) < 16:
            px = random.randint(-10, 10)
            py = random.randint(-10, 10)
            p = [px, py]
            if p not in pts:
                pts.append(p)
        initial_configs.append(pts)
    
    # Run local search from each initial config
    for config in initial_configs:
        if time.time() > deadline:
            break
        # Ensure distinct points
        seen = set()
        unique_config = []
        for p in config:
            key = (p[0], p[1])
            if key not in seen:
                seen.add(key)
                unique_config.append(p[:])
        if len(unique_config) < 16:
            continue
        
        result, score = _local_search(unique_config, deadline)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = result
    
    # Final: ensure validity
    if _score(best_overall) < 0:
        # Fallback to 4x4 grid
        best_overall = [[x, y] for x in range(4) for y in range(4)]
    
    return best_overall
