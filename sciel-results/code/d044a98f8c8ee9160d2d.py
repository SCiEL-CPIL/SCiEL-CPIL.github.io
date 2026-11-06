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

def _get_lines(points):
    """Return dict mapping line_key -> set of point indices on that line."""
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

def _smart_local_search(points, max_iters=30000, deadline=None):
    """
    Smart local search: for each point, try placing it on an existing line
    (one that already has 2+ other points) to create a 3+ point line,
    which reduces ordinary line count.
    """
    n = len(points)
    best = [p[:] for p in points]
    best_cnt = _count_ordinary_lines(best)
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        if deadline and time.time() > deadline:
            break
        improved = False
        lines = _get_lines(best)
        
        for idx in range(n):
            if deadline and time.time() > deadline:
                break
            xi, yi = best[idx]
            
            # Strategy 1: Try to place on an existing line with 2+ other points
            # For each line not containing idx, if it has >= 2 points, try extending it
            for line_key, pts_on_line in lines.items():
                if idx in pts_on_line:
                    continue
                if len(pts_on_line) < 2:
                    continue
                
                # Get two points on this line to determine direction
                pts_list = list(pts_on_line)
                p1 = best[pts_list[0]]
                p2 = best[pts_list[1]]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if dx == 0 and dy == 0:
                    continue
                g = _gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                
                # Find integer points on this line near the current point
                # Line: dy_n * x - dx_n * y = c
                c = dy_n * p1[0] - dx_n * p1[1]
                
                # Try a few positions along this line
                base_x = p1[0]
                base_y = p1[1]
                for t in range(-5, 6):
                    new_x = base_x + t * dx_n
                    new_y = base_y + t * dy_n
                    # Check if this is on the line
                    if dy_n * new_x - dx_n * new_y != c:
                        continue
                    new_pt = [new_x, new_y]
                    # Check bounds and distinctness
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
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
                    if cnt < best_cnt:
                        best_cnt = cnt
                        improved = True
                        iters += 1
                        # Rebuild lines after improvement
                        lines = _get_lines(best)
                    else:
                        best[idx] = [xi, yi]
            
            # Strategy 2: Try nearby lattice points
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = [xi + dx, yi + dy]
                    if abs(new_pt[0]) > 100 or abs(new_pt[1]) > 100:
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
                        lines = _get_lines(best)
            
            # Strategy 3: Random moves
            for _ in range(5):
                if deadline and time.time() > deadline:
                    break
                rx = random.randint(-50, 50)
                ry = random.randint(-50, 50)
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
                    lines = _get_lines(best)
        
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
            ni = [random.randint(-30, 30), random.randint(-30, 30)]
            nj = [random.randint(-30, 30), random.randint(-30, 30)]
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
    pts = []
    for x in range(w):
        for y in range(h):
            pts.append([x + offset[0], y + offset[1]])
    return pts

def _make_hesse_like():
    """
    Try to mimic Hesse configuration structure.
    Hesse: 9 points, 12 lines of 3 points each.
    For 12 points, we can try to extend this or use a similar structure.
    Start with a 3x3 grid and add 3 more points on existing lines.
    """
    pts = []
    # 3x3 grid
    for x in range(3):
        for y in range(3):
            pts.append([x, y])
    # Add points on lines that already have 2 points
    # Horizontal lines: y=0,1,2 each have 3 points already
    # Vertical lines: x=0,1,2 each have 3 points already
    # Diagonals: (0,0)-(1,1)-(2,2) has 3, (0,2)-(1,1)-(2,0) has 3
    # Other lines with 2 points: e.g., (0,0)-(2,1) direction (2,1)
    # Let's add points to create more 3-point lines
    # (0,0), (2,1), (4,2) - but 4,2 might be far
    pts.append([4, 2])   # on line through (0,0) and (2,1)
    pts.append([1, 4])   # on line through (0,2) and (2,4)? No, (0,2) to (1,4) is (1,2)
    # Actually let's be more careful
    # Line through (0,1) and (2,3): direction (2,2) -> (1,1), so (1,2) is on it? (0,1)+(1,1)=(1,2), yes
    pts.append([1, 2])   # already in grid! Skip
    # Let me restart with a cleaner approach
    return pts[:12]

def _make_three_parallel_lines():
    """
    4 points on each of 3 parallel lines.
    This gives 3 lines with 4 points each.
    Plus cross-lines between points on different lines.
    """
    pts = []
    for y in [0, 1, 2]:
        for x in [0, 1, 2, 3]:
            pts.append([x, y])
    return pts

def _make_tetrahedral():
    """
    Points arranged to maximize collinear triples.
    Use a structure based on projective plane ideas.
    """
    # Start with 4 points forming a complete quadrilateral
    # Then add points on the 6 lines determined by pairs
    pts = [[0, 0], [1, 0], [0, 1], [1, 1]]
    # Lines: y=0 (2pts), x=0 (2pts), y=1 (2pts), x=1 (2pts), 
    #         y=x (2pts: (0,0),(1,1)), y=-x+1 (2pts: (1,0),(0,1))
    # Add points on these lines to make them 3-point lines
    pts.append([2, 0])    # y=0 now has 3
    pts.append([0, 2])    # x=0 now has 3
    pts.append([2, 1])    # y=1 now has 3
    pts.append([1, 2])    # x=1 now has 3
    pts.append([2, 2])    # y=x now has 3
    pts.append([2, -1])   # y=-x+1: (1,0),(0,1),(2,-1)? Check: -2+1=-1, yes
    return pts

def _make_optimal_structure():
    """
    Try to construct a configuration close to optimal.
    For 12 points, minimum ordinary lines is 6 (Hesse bound).
    Structure: maximize number of lines with 3+ points.
    """
    # Use a 4x3 grid and perturb to create more collinear triples
    pts = []
    for x in range(4):
        for y in range(3):
            pts.append([x, y])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    random.seed(42)
    
    # Generate diverse initial configurations
    seeds = []
    seeds.append(_make_grid(4, 3))
    seeds.append(_make_grid(3, 4))
    seeds.append(_make_three_parallel_lines())
    seeds.append(_make_tetrahedral())
    seeds.append(_make_optimal_structure())
    
    # Hexagonal
    pts_hex = []
    for r in range(4):
        for c in range(3):
            x = c + (r % 2)
            y = r
            pts_hex.append([x, y])
    seeds.append(pts_hex)
    
    # Near pencil
    pts_pencil = [[i, 0] for i in range(11)]
    pts_pencil.append([5, 1])
    seeds.append(pts_pencil)
    
    # Random configurations
    for _ in range(8):
        pts = []
        while len(pts) < 12:
            p = [random.randint(-20, 20), random.randint(-20, 20)]
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
        
        # Smart local search
        pts, cnt = _smart_local_search(seed, max_iters=15000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts[:]
        
        # Pair swap
        if best_pts and time.time() < deadline:
            pts, cnt = _pair_swap_search(best_pts, max_iters=3000, deadline=deadline)
            if cnt < best_cnt:
                best_cnt = cnt
                best_pts = pts[:]
    
    # Final refinement
    if best_pts and time.time() < deadline - 5:
        pts, cnt = _smart_local_search(best_pts, max_iters=50000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts[:]
    
    if best_pts is None:
        best_pts = _make_grid(4, 3)
    
    # Verify distinctness
    seen = set()
    final = []
    for p in best_pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            final.append(p)
    
    while len(final) < 12:
        p = [random.randint(-10**9, 10**9), random.randint(-10**9, 10**9)]
        if tuple(p) not in seen:
            seen.add(tuple(p))
            final.append(p)
    
    return final
# EVOLVE-BLOCK-END
