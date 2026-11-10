import numpy as np
import time
import random
from math import gcd
from collections import defaultdict

def _normalize_line(a, b, c):
    """Return canonical (A, B, C) for a*x + b*y + c = 0. Returns None if zero line."""
    a, b, c = int(a), int(b), int(c)
    if a == 0 and b == 0:
        return None
    g = gcd(abs(a), gcd(abs(b), abs(c)))
    if g == 0:
        return None
    a, b, c = a // g, b // g, c // g
    for v in [a, b, c]:
        if v != 0:
            if v < 0:
                a, b, c = -a, -b, -c
            break
    return (a, b, c)

def _is_valid(points, lines):
    """Validate against contract."""
    try:
        p = np.asarray(points)
        l = np.asarray(lines)
    except Exception:
        return False
    if p.shape != (54, 2) or l.shape != (27, 3):
        return False
    for arr in [p, l]:
        if not np.all(np.isfinite(arr)):
            return False
        if not np.all(np.abs(arr - np.round(arr)) < 1e-6):
            return False
    if len(set(map(tuple, np.round(p, 6).astype(int).tolist()))) != 54:
        return False
    seen_lines = set()
    for row in np.round(l, 6).astype(int).tolist():
        canon = _normalize_line(row[0], row[1], row[2])
        if canon is None:
            return False
        if canon in seen_lines:
            return False
        seen_lines.add(canon)
    return True

def _incidence_count(points, lines):
    """Compute exact incidence count. Returns int."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    x = pts[:, 0, None]
    y = pts[:, 1, None]
    A = lns[:, 0, None, None]
    B = lns[:, 1, None, None]
    C = lns[:, 2, None, None]
    vals = A * x + B * y + C
    return int(np.sum(vals == 0))

def _line_intersection(l1, l2):
    """Return the intersection point (x, y) of two lines as (num_x, den, num_y, den) or None if parallel."""
    A1, B1, C1 = l1
    A2, B2, C2 = l2
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None
    # x = (B1*C2 - B2*C1) / det
    # y = (A2*C1 - A1*C2) / det
    nx = B1 * C2 - B2 * C1
    ny = A2 * C1 - A1 * C2
    # Check if integer
    if nx % det != 0 or ny % det != 0:
        return None
    return (nx // det, ny // det)

def _generate_line_pool(max_coeff=8, max_c=30):
    """Generate a pool of candidate lines with small coefficients."""
    pool = []
    seen = set()
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-max_c, max_c + 1):
                canon = _normalize_line(A, B, C)
                if canon is not None and canon not in seen:
                    seen.add(canon)
                    pool.append(canon)
    return pool

def _greedy_line_selection(line_pool, n_lines=27):
    """
    Greedily select n_lines from line_pool to maximize total pairwise intersections.
    At each step, pick the line that maximizes (intersections with already-selected lines).
    """
    # Precompute intersection map: for each pair of lines, their intersection point or None
    # This is expensive for large pools, so we'll compute on the fly for selected lines
    # but precompute a quick lookup for the pool
    
    n_pool = len(line_pool)
    selected = []
    selected_set = set()
    
    # For efficiency, track intersection points of selected lines
    # intersection_count[p] = number of selected lines passing through point p
    point_counts = defaultdict(int)
    # line_to_points: for each selected line, set of intersection points it participates in
    # We'll track: for each candidate line, how many of its intersection points with selected lines are unique/new
    
    # First, pick a random starting line (or the one with most potential)
    # Let's pick the first line
    start_idx = 0
    selected.append(line_pool[start_idx])
    selected_set.add(start_idx)
    
    # Track which points are hit by which selected lines
    # point_to_lines: point -> set of selected line indices
    point_to_lines = defaultdict(set)
    
    # For the first line, all its intersections with other lines are potential
    # But we need to track per-point multiplicity
    
    # Better approach: for each candidate line, compute sum over selected lines of:
    #   1 if they intersect at a point not yet "fully covered" (but this is complex)
    # Simpler: just count intersections with selected lines, weighted by whether the point is new
    
    # Actually, the simplest effective heuristic:
    # For each candidate line L, score = number of intersection points with selected lines
    # (each intersection is a point that will be covered by both L and the selected line)
    # But we want to maximize total distinct points covered, so prefer lines whose
    # intersections with selected lines are at NEW points (not already covered by 2+ selected lines)
    
    for step in range(1, n_lines):
        best_idx = -1
        best_score = -1
        best_intersections = None
        
        for idx in range(n_pool):
            if idx in selected_set:
                continue
            L = line_pool[idx]
            score = 0
            intersections = []
            for sel_idx in selected:
                pt = _line_intersection(L, line_pool[sel_idx])
                if pt is not None:
                    # Check if this point is already heavily covered
                    current_cover = point_counts[pt]
                    # Marginal gain: if point is already covered by k selected lines,
                    # adding L gives 1 more incidence at this point, but the point already
                    # contributes k incidences. The NEW incidences added = 1 (for L at this point)
                    # plus the point might get a new "slot" but since we have fixed 54 points,
                    # what matters is: this point will be one of our 54, and it'll be on (k+1) lines.
                    # The total contribution of this point to I is (k+1) if we place a point there.
                    # But we can only place 54 points total.
                    # So the marginal value of adding L is: sum over its intersection points
                    # of (1 if point is new to our consideration, else the increase in multiplicity)
                    # Simplified: count each intersection as 1, but weight by whether point is new
                    if current_cover == 0:
                        score += 2  # New point, will be on 2 lines
                    elif current_cover == 1:
                        score += 1  # Point already on 1 line, now on 2
                    else:
                        score += 0.5  # Point already on 2+ lines, marginal gain small
                    intersections.append((pt, sel_idx))
            
            if score > best_score:
                best_score = score
                best_idx = idx
                best_intersections = intersections
        
        if best_idx == -1:
            break
        
        selected.append(line_pool[best_idx])
        selected_set.add(best_idx)
        
        # Update point_counts
        for pt, _ in best_intersections:
            point_counts[pt] += 1
    
    return [line_pool[i] for i in selected]

def _place_points_greedy(lines, n_points=54):
    """
    Given n_lines lines, place n_points points to maximize incidences.
    Greedy: at each step, place a point at the location that maximizes the number of
    lines it's on (i.e., highest-multiplicity intersection point), preferring unoccupied locations.
    """
    # Find all intersection points of line pairs
    point_line_membership = defaultdict(set)  # point -> set of line indices
    n_lines = len(lines)
    
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(lines[i], lines[j])
            if pt is not None:
                point_line_membership[pt].add(i)
                point_line_membership[pt].add(j)
    
    # Also consider points that lie on a single line (if no intersection exists there)
    # For now, only use intersection points
    
    # Sort points by number of lines they're on (descending)
    sorted_points = sorted(point_line_membership.items(), key=lambda x: -len(x[1]))
    
    # Greedily select points
    selected_points = []
    used_positions = set()
    
    for pt, line_set in sorted_points:
        if len(selected_points) >= n_points:
            break
        if pt not in used_positions:
            selected_points.append(pt)
            used_positions.add(pt)
    
    # If we don't have enough, add arbitrary points
    if len(selected_points) < n_points:
        # Add points from the grid
        for x in range(-10, 11):
            for y in range(-10, 11):
                if (x, y) not in used_positions and len(selected_points) < n_points:
                    selected_points.append((x, y))
                    used_positions.add((x, y))
                if len(selected_points) >= n_points:
                    break
            if len(selected_points) >= n_points:
                break
    
    return selected_points[:n_points]

def _local_search(points, lines, start_time, deadline, seed=42):
    """
    Fast local search: try swapping each point with a better candidate, and each line with a better candidate.
    Uses precomputed structures for speed.
    """
    rng = random.Random(seed)
    pts = np.array(points, dtype=np.int64)
    lns = np.array(lines, dtype=np.int64)
    n_pts = 54
    n_lines = 27
    
    # Build incidence matrix
    point_line_inc = np.zeros((n_pts, n_lines), dtype=bool)
    for i in range(n_pts):
        for j in range(n_lines):
            A, B, C = lns[j]
            if A * pts[i, 0] + B * pts[i, 1] + C == 0:
                point_line_inc[i, j] = True
    
    total_inc = int(point_line_inc.sum())
    best_total = total_inc
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    # Generate candidate points around current points
    cand_points_set = set(map(tuple, pts.tolist()))
    for x in range(-15, 16):
        for y in range(-15, 16):
            cand_points_set.add((x, y))
    cand_points = list(cand_points_set)
    
    # Generate candidate lines
    cand_lines_set = set()
    for A in range(-8, 9):
        for B in range(-8, 9):
            if A == 0 and B == 0:
                continue
            for C in range(-30, 31):
                canon = _normalize_line(A, B, C)
                if canon:
                    cand_lines_set.add(canon)
    cand_lines = list(cand_lines_set)
    
    # Precompute: for each candidate point, which lines it's on (among current lines)
    # This is expensive, so compute on the fly but cache
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        improved = False
        
        # Try point swaps
        for i in range(n_pts):
            if time.time() > deadline - 10:
                break
            old_pos = tuple(pts[i])
            best_delta = 0
            best_new_pos = None
            
            # Compute current incidences for point i
            current_inc_i = int(point_line_inc[i, :].sum())
            
            # Try a random subset of candidates for speed
            n_tries = min(50, len(cand_points))
            for _ in range(n_tries):
                new_pos = cand_points[rng.randint(0, len(cand_points) - 1)]
                if new_pos == old_pos:
                    continue
                if new_pos in set(map(tuple, pts.tolist())) - {old_pos}:
                    continue
                
                # Compute new incidences for point i
                new_inc_i = 0
                for j in range(n_lines):
                    A, B, C = lns[j]
                    if A * new_pos[0] + B * new_pos[1] + C == 0:
                        new_inc_i += 1
                
                delta = new_inc_i - current_inc_i
                if delta > best_delta:
                    best_delta = delta
                    best_new_pos = new_pos
            
            if best_new_pos is not None and best_delta > 0:
                # Apply the swap
                for j in range(n_lines):
                    A, B, C = lns[j]
                    point_line_inc[i, j] = (A * best_new_pos[0] + B * best_new_pos[1] + C == 0)
                pts[i] = best_new_pos
                total_inc += best_delta
                improved = True
        
        # Try line swaps
        for j in range(n_lines):
            if time.time() > deadline - 10:
                break
            old_line = tuple(lns[j])
            best_delta = 0
            best_new_line = None
            
            current_inc_j = int(point_line_inc[:, j].sum())
            
            n_tries = min(50, len(cand_lines))
            for _ in range(n_tries):
                new_line = cand_lines[rng.randint(0, len(cand_lines) - 1)]
                if new_line == old_line:
                    continue
                # Check if new_line is already used
                if any(tuple(lns[k]) == new_line for k in range(n_lines) if k != j):
                    continue
                
                new_inc_j = 0
                A, B, C = new_line
                for i in range(n_pts):
                    if A * pts[i, 0] + B * pts[i, 1] + C == 0:
                        new_inc_j += 1
                
                delta = new_inc_j - current_inc_j
                if delta > best_delta:
                    best_delta = delta
                    best_new_line = new_line
            
            if best_new_line is not None and best_delta > 0:
                # Apply the swap
                A, B, C = best_new_line
                for i in range(n_pts):
                    point_line_inc[i, j] = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
                lns[j] = best_new_line
                total_inc += best_delta
                improved = True
        
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
            if total_inc > best_total:
                best_total = total_inc
                best_pts = pts.copy()
                best_lns = lns.copy()
    
    return best_pts, best_lns, best_total

def _grid_construction(k=3):
    """Baseline grid."""
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    return points, lines

def _structured_config_1():
    """6x9 grid with best lines."""
    from collections import defaultdict
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    line_counts = defaultdict(int)
    n_pts = len(pts)
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            A = -dy
            B = dx
            C = dy * x1 - dx * y1
            canon = _normalize_line(A, B, C)
            if canon:
                line_counts[canon] += 1
    
    all_lines = list(line_counts.keys())
    line_inc_counts = []
    for canon in all_lines:
        A, B, C = canon
        cnt = 0
        for pt in pts:
            if A * pt[0] + B * pt[1] + C == 0:
                cnt += 1
        line_inc_counts.append((cnt, canon))
    
    line_inc_counts.sort(key=lambda x: -x[0])
    selected = []
    seen = set()
    for cnt, canon in line_inc_counts:
        if canon not in seen:
            selected.append(canon)
            seen.add(canon)
            if len(selected) == 27:
                break
    
    if len(selected) == 27:
        lines = np.array(selected, dtype=np.int64)
        return pts, lines
    return None, None

def _structured_config_2():
    """9x6 grid with best lines."""
    from collections import defaultdict
    xs = np.arange(9)
    ys = np.arange(6)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    line_counts = defaultdict(int)
    n_pts = len(pts)
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            A = -dy
            B = dx
            C = dy * x1 - dx * y1
            canon = _normalize_line(A, B, C)
            if canon:
                line_counts[canon] += 1
    
    all_lines = list(line_counts.keys())
    line_inc_counts = []
    for canon in all_lines:
        A, B, C = canon
        cnt = 0
        for pt in pts:
            if A * pt[0] + B * pt[1] + C == 0:
                cnt += 1
        line_inc_counts.append((cnt, canon))
    
    line_inc_counts.sort(key=lambda x: -x[0])
    selected = []
    seen = set()
    for cnt, canon in line_inc_counts:
        if canon not in seen:
            selected.append(canon)
            seen.add(canon)
            if len(selected) == 27:
                break
    
    if len(selected) == 27:
        lines = np.array(selected, dtype=np.int64)
        return pts, lines
    return None, None

def _structured_config_7x8():
    """7x8 grid with best lines - supports longer diagonals."""
    from collections import defaultdict
    xs = np.arange(7)
    ys = np.arange(8)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    line_counts = defaultdict(int)
    n_pts = len(pts)
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            A = -dy
            B = dx
            C = dy * x1 - dx * y1
            canon = _normalize_line(A, B, C)
            if canon:
                line_counts[canon] += 1
    
    all_lines = list(line_counts.keys())
    line_inc_counts = []
    for canon in all_lines:
        A, B, C = canon
        cnt = 0
        for pt in pts:
            if A * pt[0] + B * pt[1] + C == 0:
                cnt += 1
        line_inc_counts.append((cnt, canon))
    
    line_inc_counts.sort(key=lambda x: -x[0])
    selected = []
    seen = set()
    for cnt, canon in line_inc_counts:
        if canon not in seen:
            selected.append(canon)
            seen.add(canon)
            if len(selected) == 27:
                break
    
    if len(selected) == 27:
        lines = np.array(selected, dtype=np.int64)
        return pts, lines
    return None, None

def _structured_config_8x7():
    """8x7 grid with best lines."""
    from collections import defaultdict
    xs = np.arange(8)
    ys = np.arange(7)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    line_counts = defaultdict(int)
    n_pts = len(pts)
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            A = -dy
            B = dx
            C = dy * x1 - dx * y1
            canon = _normalize_line(A, B, C)
            if canon:
                line_counts[canon] += 1
    
    all_lines = list(line_counts.keys())
    line_inc_counts = []
    for canon in all_lines:
        A, B, C = canon
        cnt = 0
        for pt in pts:
            if A * pt[0] + B * pt[1] + C == 0:
                cnt += 1
        line_inc_counts.append((cnt, canon))
    
    line_inc_counts.sort(key=lambda x: -x[0])
    selected = []
    seen = set()
    for cnt, canon in line_inc_counts:
        if canon not in seen:
            selected.append(canon)
            seen.add(canon)
            if len(selected) == 27:
                break
    
    if len(selected) == 27:
        lines = np.array(selected, dtype=np.int64)
        return pts, lines
    return None, None

def _greedy_construction(deadline):
    """
    Full greedy construction: select lines to maximize intersections, then place points.
    """
    # Generate line pool
    line_pool = _generate_line_pool(max_coeff=6, max_c=25)
    
    # Greedily select 27 lines
    selected_lines = _greedy_line_selection(line_pool, n_lines=27)
    
    if len(selected_lines) < 27:
        return None, None
    
    # Place 54 points greedily
    points = _place_points_greedy(selected_lines, n_points=54)
    
    if len(points) < 54:
        return None, None
    
    pts = np.array(points, dtype=np.int64)
    lns = np.array(selected_lines, dtype=np.int64)
    
    # Validate
    if not _is_valid(pts, lns):
        return None, None
    
    return pts, lns

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences with 54 points and 27 lines on integer lattice.
    Uses greedy combinatorial construction followed by local search.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    configs = []
    
    # Baseline grid
    p1, l1 = _grid_construction(3)
    configs.append((p1, l1, _incidence_count(p1, l1)))
    
    # Structured grids
    p2, l2 = _structured_config_1()
    if p2 is not None:
        configs.append((p2, l2, _incidence_count(p2, l2)))
    
    p3, l3 = _structured_config_2()
    if p3 is not None:
        configs.append((p3, l3, _incidence_count(p3, l3)))
    
    p7, l7 = _structured_config_7x8()
    if p7 is not None:
        configs.append((p7, l7, _incidence_count(p7, l7)))
    
    p8, l8 = _structured_config_8x7()
    if p8 is not None:
        configs.append((p8, l8, _incidence_count(p8, l8)))
    
    # Greedy combinatorial construction
    if time.time() < deadline - 120:
        pg, lg = _greedy_construction(deadline)
        if pg is not None:
            configs.append((pg, lg, _incidence_count(pg, lg)))
    
    # Sort by incidence count descending
    configs.sort(key=lambda x: -x[2])
    
    best_points = configs[0][0].copy()
    best_lines = configs[0][1].copy()
    best_count = configs[0][2]
    
    # Run local search from the best config
    if time.time() < deadline - 30:
        ls_pts, ls_lns, ls_count = _local_search(
            best_points, best_lines, start_time, deadline, seed=42
        )
        if ls_count > best_count:
            best_points = ls_pts
            best_lines = ls_lns
            best_count = ls_count
    
    # If time remains, try local search from second-best config
    if len(configs) > 1 and time.time() < deadline - 60:
        ls_pts2, ls_lns2, ls_count2 = _local_search(
            configs[1][0], configs[1][1], start_time, deadline, seed=123
        )
        if ls_count2 > best_count:
            best_points = ls_pts2
            best_lines = ls_lns2
            best_count = ls_count2
    
    # If time remains, try local search from third-best config
    if len(configs) > 2 and time.time() < deadline - 90:
        ls_pts3, ls_lns3, ls_count3 = _local_search(
            configs[2][0], configs[2][1], start_time, deadline, seed=999
        )
        if ls_count3 > best_count:
            best_points = ls_pts3
            best_lines = ls_lns3
            best_count = ls_count3
    
    # If time remains, try local search from fourth-best config
    if len(configs) > 3 and time.time() < deadline - 120:
        ls_pts4, ls_lns4, ls_count4 = _local_search(
            configs[3][0], configs[3][1], start_time, deadline, seed=777
        )
        if ls_count4 > best_count:
            best_points = ls_pts4
            best_lines = ls_lns4
            best_count = ls_count4
    
    # Final validation
    best_points = np.array(best_points, dtype=np.int64)
    best_lines = np.array(best_lines, dtype=np.int64)
    
    if not _is_valid(best_points, best_lines):
        # Fallback to safe grid
        best_points, best_lines = _grid_construction(3)
    
    return best_points, best_lines
