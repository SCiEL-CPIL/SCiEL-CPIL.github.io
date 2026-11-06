"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_map = defaultdict(set)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            g = _gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            c = dy_n * xi - dx_n * yi
            line_key = (dx_n, dy_n, c)
            line_map[line_key].add(i)
            line_map[line_key].add(j)
    ordinary = 0
    for key, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _get_line_structure(points):
    """Return dict mapping line_key -> set of point indices."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            g = _gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            c = dy_n * xi - dx_n * yi
            line_key = (dx_n, dy_n, c)
            line_map[line_key].add(i)
            line_map[line_key].add(j)
    return line_map

def _try_place_on_line(points, idx, line_key):
    """Try to find an integer point on the given line that's not already occupied.
    Returns the point if found, else None."""
    dx_n, dy_n, c = line_key
    # Line: dy_n * x - dx_n * y = c
    # Parametric: x = x0 + dx_n * t, y = y0 + dy_n * t for some base point
    # Find a base point: if dx_n != 0, set y=0: x = c/dy_n... messy.
    # Instead, try small t values around existing points on the line.
    existing_pts = set()
    for i in range(len(points)):
        if i != idx:
            existing_pts.add(tuple(points[i]))
    
    # Find one point on the line to use as base
    base = None
    for i in range(len(points)):
        xi, yi = points[i]
        if dy_n * xi - dx_n * yi == c:
            base = (xi, yi)
            break
    
    if base is None:
        # Need to find any integer point on the line
        # dy_n * x - dx_n * y = c
        # Try small values
        for x in range(-20, 21):
            val = c + dx_n * 0  # dy_n * x = c + dx_n * y
            # dy_n * x - dx_n * y = c => dy_n * x = c + dx_n * y
            # For y=0: dy_n * x = c, so x = c/dy_n if dy_n != 0
            if dy_n != 0 and c % dy_n == 0:
                x0 = c // dy_n
                if (x0, 0) not in existing_pts:
                    return [x0, 0]
                base = (x0, 0)
                break
            if dx_n != 0:
                # x = 0: -dx_n * y = c => y = -c/dx_n
                if c % dx_n == 0:
                    y0 = -c // dx_n
                    if (0, y0) not in existing_pts:
                        return [0, y0]
                    base = (0, y0)
                    break
        if base is None:
            return None
    
    # Now try points along the line from base
    for t in range(-10, 11):
        if t == 0:
            continue
        nx = base[0] + dx_n * t
        ny = base[1] + dy_n * t
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        if (nx, ny) not in existing_pts:
            return [nx, ny]
    return None

def _smart_local_search(points, max_iters=30000, deadline=None):
    """Local search with smart moves: place points on existing multi-point lines."""
    n = len(points)
    best = [p[:] for p in points]
    best_cnt = _count_ordinary_lines(best)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        if deadline and time.time() > deadline:
            break
        improved = False
        line_map = _get_line_structure(best)
        
        for idx in range(n):
            if deadline and time.time() > deadline:
                break
            xi, yi = best[idx]
            
            # Strategy 1: Try to move point onto an existing line with 3+ points
            for line_key, pts_on_line in line_map.items():
                if len(pts_on_line) >= 3 and idx not in pts_on_line:
                    new_pt = _try_place_on_line(best, idx, line_key)
                    if new_pt is not None:
                        best[idx] = new_pt
                        cnt = _count_ordinary_lines(best)
                        best[idx] = [xi, yi]
                        if cnt < best_cnt:
                            best[idx] = new_pt
                            best_cnt = cnt
                            improved = True
                            iters += 1
                            break
            
            # Strategy 2: Nearby moves
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = [xi + dx, yi + dy]
                    if abs(new_pt[0]) > 200 or abs(new_pt[1]) > 200:
                        continue
                    valid = True
                    for j in range(n):
                        if j != idx and best[j] == new_pt:
                            valid = False
                            break
                    if not valid:
                        continue
                    best[idx] = new_pt
                    cnt = _count_ordinary_lines(best)
                    best[idx] = [xi, yi]
                    if cnt < best_cnt:
                        best[idx] = new_pt
                        best_cnt = cnt
                        improved = True
                        iters += 1
                        break
            
            # Strategy 3: Random distant moves
            for _ in range(5):
                if deadline and time.time() > deadline:
                    break
                rx = random.randint(-100, 100)
                ry = random.randint(-100, 100)
                new_pt = [rx, ry]
                valid = True
                for j in range(n):
                    if j != idx and best[j] == new_pt:
                        valid = False
                        break
                if not valid:
                    continue
                best[idx] = new_pt
                cnt = _count_ordinary_lines(best)
                best[idx] = [xi, yi]
                if cnt < best_cnt:
                    best[idx] = new_pt
                    best_cnt = cnt
                    improved = True
                    iters += 1
                    break
        iters += 1
    return best, best_cnt

def _pair_swap_search(points, max_iters=10000, deadline=None):
    """Try swapping pairs of points with random new positions."""
    n = len(points)
    best = [p[:] for p in points]
    best_cnt = _count_ordinary_lines(best)
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        i, j = random.sample(range(n), 2)
        old_i, old_j = best[i][:], best[j][:]
        for _ in range(5):
            ni = [random.randint(-50, 50), random.randint(-50, 50)]
            nj = [random.randint(-50, 50), random.randint(-50, 50)]
            if ni == nj:
                continue
            valid = True
            for k in range(n):
                if k != i and k != j:
                    if best[k] == ni or best[k] == nj:
                        valid = False
                        break
                if k == i and ni == old_j:
                    valid = False
                if k == j and nj == old_i:
                    valid = False
            if not valid:
                continue
            best[i] = ni
            best[j] = nj
            cnt = _count_ordinary_lines(best)
            best[i] = old_i
            best[j] = old_j
            if cnt < best_cnt:
                best[i] = ni
                best[j] = nj
                best_cnt = cnt
    return best, best_cnt

def _make_grid(w, h, offset=(0, 0)):
    """Create a w x h grid."""
    pts = []
    for x in range(w):
        for y in range(h):
            pts.append([x + offset[0], y + offset[1]])
    return pts

def _make_hexagonal(rows=4):
    """Create a hexagonal-ish arrangement."""
    pts = []
    for r in range(rows):
        for c in range(rows):
            x = c + (r % 2)
            y = r
            pts.append([x, y])
    return pts[:12]

def _make_near_pencil():
    """11 points on a line, 1 off."""
    pts = [[i, 0] for i in range(11)]
    pts.append([5, 1])
    return pts

def _make_staggered():
    """Staggered rows."""
    pts = []
    for r in range(4):
        for c in range(3):
            x = c + (r % 2) * 2
            y = r * 2
            pts.append([x, y])
    return pts

def _make_diagonal_grid():
    """Points on diagonal lines."""
    pts = []
    for i in range(4):
        for j in range(3):
            pts.append([i + j, i - j])
    return pts

def _make_three_parallel_lines():
    """4 points on each of 3 parallel lines."""
    pts = []
    for y in [0, 1, 2]:
        for x in range(4):
            pts.append([x, y])
    return pts

def _make_tetrahedral():
    """Points arranged to maximize collinear triples."""
    # 4 points on each of 3 lines through a common point
    pts = [[0, 0]]
    # Line y=0: 3 more points
    pts.extend([[1, 0], [2, 0], [3, 0]])
    # Line x=0: 3 more points  
    pts.extend([[0, 1], [0, 2], [0, 3]])
    # Line y=x: 3 more points
    pts.extend([[1, 1], [2, 2], [3, 3]])
    return pts

def _make_optimal_structure():
    """Try to construct a configuration with few ordinary lines.
    Based on known results: 12 points can have as few as 6 ordinary lines."""
    # Try a configuration based on the Hesse configuration or similar
    # 3x4 grid has many collinear points
    pts = []
    for x in range(3):
        for y in range(4):
            pts.append([x, y])
    return pts

def _scale_points(points, factor=100):
    """Scale points to use more of the coordinate space."""
    return [[x * factor, y * factor] for x, y in points]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    random.seed(42)
    
    # Generate diverse initial configurations
    seeds = []
    seeds.append(_make_grid(4, 3))
    seeds.append(_make_grid(3, 4))
    seeds.append(_make_hexagonal())
    seeds.append(_make_near_pencil())
    seeds.append(_make_staggered())
    seeds.append(_make_diagonal_grid())
    seeds.append(_make_three_parallel_lines())
    seeds.append(_make_tetrahedral())
    seeds.append(_make_optimal_structure())
    
    # Add random configurations
    for _ in range(8):
        pts = []
        while len(pts) < 12:
            p = [random.randint(-30, 30), random.randint(-30, 30)]
            if p not in pts:
                pts.append(p)
        seeds.append(pts)
    
    best_pts = None
    best_cnt = float('inf')
    
    for seed in seeds:
        if time.time() > deadline:
            break
        cnt = _count_ordinary_lines(seed)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = [p[:] for p in seed]
        
        # Smart local search from this seed
        pts, cnt = _smart_local_search(seed, max_iters=15000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = [p[:] for p in pts]
        
        # Pair swap from best so far
        if best_pts:
            pts, cnt = _pair_swap_search(best_pts, max_iters=3000, deadline=deadline)
            if cnt < best_cnt:
                best_cnt = cnt
                best_pts = [p[:] for p in pts]
    
    # Final refinement
    if best_pts and time.time() < deadline - 10:
        pts, cnt = _smart_local_search(best_pts, max_iters=30000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = [p[:] for p in pts]
    
    if best_pts is None:
        best_pts = _make_grid(4, 3)
    
    # Verify distinctness and bounds
    seen = set()
    final = []
    for p in best_pts:
        tp = tuple(p)
        if tp not in seen and abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9:
            seen.add(tp)
            final.append(p)
    
    while len(final) < 12:
        p = [random.randint(-10**9, 10**9), random.randint(-10**9, 10**9)]
        if tuple(p) not in seen:
            seen.add(tuple(p))
            final.append(p)
    
    return final
# EVOLVE-BLOCK-END
