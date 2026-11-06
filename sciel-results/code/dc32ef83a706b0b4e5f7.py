"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return canonical line key (a, b, c) for line through p1, p2: ax + by + c = 0."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    if a == 0 and b == 0:
        return None
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        g = 1
    a, b, c = a // g, b // g, c // g
    # Canonical sign: first non-zero of (a, b, c) is positive
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_ordinary(points):
    """Count lines containing exactly 2 of the points."""
    n = len(points)
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_pts.setdefault(key, []).append(j)
    ordinary = 0
    for key, pts in line_pts.items():
        if len(pts) == 1:
            ordinary += 1
    return ordinary

def _is_valid(points):
    """12 distinct integer points, not all collinear."""
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    for p in points:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
    # Not all collinear
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return True
    for i in range(3, 12):
        x4, y4 = points[i]
        if (x2 - x1) * (y4 - y1) != (y2 - y1) * (x4 - x1):
            return True
    return False

def _point_on_line_count(points):
    """For each point, return list of (count, line_key) for lines through it."""
    n = len(points)
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_pts.setdefault(key, set()).add(i)
                line_pts[key].add(j)
    # Build per-point line info
    point_lines = [[] for _ in range(n)]
    for key, pts in line_pts.items():
        cnt = len(pts)
        for i in pts:
            point_lines[i].append((cnt, key))
    return point_lines

def _eval_move(points, i, new_pt):
    """
    Evaluate the change in ordinary-line count when replacing points[i] with new_pt.
    Returns the new total ordinary count, or None if invalid.
    """
    new_points = [list(p) for p in points]
    new_points[i] = list(new_pt)
    if not _is_valid(new_points):
        return None
    return _count_ordinary(new_points)

def _try_hesse_configurations():
    """
    Try Hesse-configuration-like arrangements: 12 points, many 3-point lines.
    The Hesse configuration (dual of (9_3, 12_3)) has 12 points and 9 lines,
    each line through 3 points, each point on 3 lines -> 0 ordinary lines (if realizable).
    We try integer realizations.
    """
    best_points = None
    best_score = float('inf')
    
    # Configuration 1: 3x4 grid
    grid34 = [[i, j] for i in range(3) for j in range(4)]
    s = _count_ordinary(grid34)
    if s < best_score:
        best_score = s
        best_points = grid34
    
    # Configuration 2: 4x3 grid
    grid43 = [[i, j] for i in range(4) for j in range(3)]
    s = _count_ordinary(grid43)
    if s < best_score:
        best_score = s
        best_points = grid43
    
    # Configuration 3: Points on 3 parallel lines with 4 each, aligned
    for offset_x in range(-2, 3):
        for offset_y in range(-2, 3):
            pts = []
            # Line y=0: x = 0,1,2,3
            for x in range(4):
                pts.append([x, 0])
            # Line y=3: x = offset_x + 0..3
            for x in range(4):
                pts.append([offset_x + x, 3])
            # Line y=6: x = 2*offset_x + 0..3
            for x in range(4):
                pts.append([2*offset_x + x, 6])
            if len(set(tuple(p) for p in pts)) == 12 and _is_valid(pts):
                s = _count_ordinary(pts)
                if s < best_score:
                    best_score = s
                    best_points = [list(p) for p in pts]
    
    # Configuration 4: 4 points on each of 3 concurrent lines
    # Lines through origin with different slopes
    for m1 in [1, 2, 3]:
        for m2 in [1, 2, 3]:
            if m1 == m2:
                continue
            for m3 in [1, 2, 3, -1]:
                if m3 in (m1, m2):
                    continue
                pts = []
                for t in [1, 2, 3, 4]:
                    pts.append([t, m1 * t])
                for t in [1, 2, 3, 4]:
                    pts.append([t, m2 * t])
                for t in [1, 2, 3, 4]:
                    pts.append([t, m3 * t])
                if len(set(tuple(p) for p in pts)) == 12 and _is_valid(pts):
                    s = _count_ordinary(pts)
                    if s < best_score:
                        best_score = s
                        best_points = [list(p) for p in pts]
    
    # Configuration 5: 12 points as vertices + face centers of a "cube-like" arrangement
    # Try: 8 cube vertices + 4 face centers
    cube_pts = []
    for x in [0, 1]:
        for y in [0, 1]:
            for z in [0, 1]:
                cube_pts.append([x, y])
    # That's only 4 unique 2D points... Let me use a different approach.
    # Try: 12 points on a conic-like structure
    
    # Configuration 6: 2x6 grid
    grid26 = [[i, j] for i in range(2) for j in range(6)]
    s = _count_ordinary(grid26)
    if s < best_score:
        best_score = s
        best_points = grid26
    
    # Configuration 7: 6x2 grid
    grid62 = [[i, j] for i in range(6) for j in range(2)]
    s = _count_ordinary(grid62)
    if s < best_score:
        best_score = s
        best_points = grid62
    
    # Configuration 8: 12 points on 4 lines of 3
    # 4 lines, each with 3 points, sharing some points
    # Line 1: y=0, x=0,1,2
    # Line 2: y=1, x=0,1,2
    # Line 3: y=2, x=0,1,2
    # Line 4: x=0, y=3,4,5 -> but need 12 distinct
    # Better: 4 vertical lines x=0,1,2,3 with y=0,1,2 each
    # That's just 4x3 grid already tried
    
    # Configuration 9: Try (3,3,3,3) on 4 parallel lines with offsets
    for ox1 in range(-2, 3):
        for ox2 in range(-2, 3):
            for ox3 in range(-2, 3):
                pts = []
                for x in range(3):
                    pts.append([x, 0])
                for x in range(3):
                    pts.append([ox1 + x, 1])
                for x in range(3):
                    pts.append([ox2 + x, 2])
                for x in range(3):
                    pts.append([ox3 + x, 3])
                if len(set(tuple(p) for p in pts)) == 12 and _is_valid(pts):
                    s = _count_ordinary(pts)
                    if s < best_score:
                        best_score = s
                        best_points = [list(p) for p in pts]
    
    return best_points, best_score

def _local_search(points, deadline):
    """
    Local search: try moving individual points to nearby integer positions
    or to points on existing lines, to reduce ordinary line count.
    """
    current = [list(p) for p in points]
    current_score = _count_ordinary(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    no_improve = 0
    max_no_improve = 50
    
    # Build a candidate pool
    def _get_candidates(pts):
        cands = set()
        for i in range(12):
            for j in range(i + 1, 12):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    g = 1
                dx //= g
                dy //= g
                for t in range(-2, 3):
                    cands.add((x1 + t * dx, y1 + t * dy))
            # Nearby points
            x, y = pts[i]
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    cands.add((x + dx, y + dy))
        # Remove current points
        cur_set = set(tuple(p) for p in pts)
        cands -= cur_set
        # Filter by bounds
        cands = {(x, y) for x, y in cands if abs(x) <= 10**9 and abs(y) <= 10**9}
        return list(cands)
    
    candidate_list = _get_candidates(current)
    random.shuffle(candidate_list)
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        random.shuffle(candidate_list)
        
        # Ensure candidates are fresh
        if len(candidate_list) < 20:
            candidate_list = _get_candidates(current)
            random.shuffle(candidate_list)
        
        for i in range(12):
            if time.time() >= deadline:
                break
            old_pt = tuple(current[i])
            found = False
            for cand in candidate_list:
                if cand == old_pt:
                    continue
                if cand in set(tuple(p) for p in current if tuple(p) != old_pt):
                    continue
                
                new_pts = [list(p) for p in current]
                new_pts[i] = list(cand)
                
                if not _is_valid(new_pts):
                    continue
                
                score = _count_ordinary(new_pts)
                if score < current_score:
                    current = new_pts
                    current_score = score
                    improved = True
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best = [list(p) for p in current]
                    found = True
                    # Update candidates
                    candidate_list = _get_candidates(current)
                    random.shuffle(candidate_list)
                    break
            if found:
                break
        if not improved:
            no_improve += 1
    
    return best, best_score

def _two_point_move_search(points, deadline):
    """
    Try moving two points simultaneously for more powerful local search.
    """
    current = [list(p) for p in points]
    current_score = _count_ordinary(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    no_improve = 0
    max_no_improve = 30
    
    def _get_candidates(pts):
        cands = set()
        for i in range(12):
            for j in range(i + 1, 12):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    g = 1
                dx //= g
                dy //= g
                for t in range(-2, 3):
                    cands.add((x1 + t * dx, y1 + t * dy))
            x, y = pts[i]
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    cands.add((x + dx, y + dy))
        cur_set = set(tuple(p) for p in pts)
        cands -= cur_set
        cands = {(x, y) for x, y in cands if abs(x) <= 10**9 and abs(y) <= 10**9}
        return list(cands)
    
    candidate_list = _get_candidates(current)
    random.shuffle(candidate_list)
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        random.shuffle(candidate_list)
        
        if len(candidate_list) < 20:
            candidate_list = _get_candidates(current)
            random.shuffle(candidate_list)
        
        # Try pairs of points
        for i in range(12):
            for j in range(i + 1, 12):
                if time.time() >= deadline:
                    break
                old_i = tuple(current[i])
                old_j = tuple(current[j])
                other_set = set(tuple(p) for p in current if p[0] != old_i[0] or p[1] != old_i[1])
                
                found = False
                for cand_i in candidate_list:
                    if cand_i == old_i or cand_i == old_j:
                        continue
                    if cand_i in other_set:
                        continue
                    for cand_j in candidate_list:
                        if cand_j == old_i or cand_j == old_j:
                            continue
                        if cand_j == cand_i:
                            continue
                        if cand_j in other_set:
                            continue
                        
                        new_pts = [list(p) for p in current]
                        new_pts[i] = list(cand_i)
                        new_pts[j] = list(cand_j)
                        
                        if not _is_valid(new_pts):
                            continue
                        
                        score = _count_ordinary(new_pts)
                        if score < current_score:
                            current = new_pts
                            current_score = score
                            improved = True
                            no_improve = 0
                            if current_score < best_score:
                                best_score = current_score
                                best = [list(p) for p in current]
                            candidate_list = _get_candidates(current)
                            random.shuffle(candidate_list)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        
        if not improved:
            no_improve += 1
    
    return best, best_score

def construct():
    """Return 12 integer points minimizing ordinary lines."""
    deadline = time.time() + 140
    
    # Phase 1: Try structured configurations
    best_points, best_score = _try_hesse_configurations()
    
    # Phase 2: Single-point local search
    if time.time() < deadline - 20:
        ls_points, ls_score = _local_search(best_points, deadline - 20)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points
    
    # Phase 3: Two-point move search
    if best_score > 6 and time.time() < deadline - 10:
        tp_points, tp_score = _two_point_move_search(best_points, deadline - 10)
        if tp_score < best_score:
            best_score = tp_score
            best_points = tp_points
    
    # Phase 4: More random restarts if still above 6
    if best_score > 6 and time.time() < deadline - 15:
        random.seed(42)
        for restart in range(50):
            if time.time() >= deadline - 10:
                break
            
            # Generate random structured configuration
            pts = []
            # Choose 3-4 lines and place 3-4 points on each
            num_lines = random.choice([3, 4])
            points_per_line = [4] * num_lines if num_lines == 3 else [3, 3, 3, 3]
            
            lines = []
            # Line types: horizontal, vertical, or diagonal
            for _ in range(num_lines):
                ltype = random.choice(['h', 'v', 'd1', 'd2'])
                if ltype == 'h':
                    y = random.randint(-3, 3)
                    xs = random.sample(range(-4, 5), points_per_line[len(lines)])
                    lines.append(('h', y, xs))
                elif ltype == 'v':
                    x = random.randint(-3, 3)
                    ys = random.sample(range(-4, 5), points_per_line[len(lines)])
                    lines.append(('v', x, ys))
                elif ltype == 'd1':
                    b = random.randint(-3, 3)
                    xs = random.sample(range(-4, 5), points_per_line[len(lines)])
                    lines.append(('d1', b, xs))
                else:
                    b = random.randint(-3, 3)
                    xs = random.sample(range(-4, 5), points_per_line[len(lines)])
                    lines.append(('d2', b, xs))
            
            for ltype, param, coords in lines:
                for c in coords:
                    if ltype == 'h':
                        pt = [c, param]
                    elif ltype == 'v':
                        pt = [param, c]
                    elif ltype == 'd1':
                        pt = [c, c + param]
                    else:
                        pt = [c, -c + param]
                    if all(tuple(pt) != tuple(p) for p in pts):
                        pts.append(pt)
            
            # Fill remaining with random points
            while len(pts) < 12:
                x = random.randint(-5, 5)
                y = random.randint(-5, 5)
                if all((x, y) != tuple(p) for p in pts):
                    pts.append([x, y])
            
            if len(pts) != 12 or not _is_valid(pts):
                continue
            
            s = _count_ordinary(pts)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in pts]
            
            # Local search from this configuration
            if time.time() < deadline - 10:
                ls_points, ls_score = _local_search(pts, deadline - 10)
                if ls_score < best_score:
                    best_score = ls_score
                    best_points = ls_points
    
    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
