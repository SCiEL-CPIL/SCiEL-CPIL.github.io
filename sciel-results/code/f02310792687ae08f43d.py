import numpy as np
import time
import random
from math import gcd
from collections import defaultdict

def _normalize_line(a, b, c):
    a, b, c = int(a), int(b), int(c)
    if a == 0 and b == 0:
        return None
    g = gcd(abs(a), gcd(abs(b), abs(c)))
    if g == 0:
        return None
    a, b, c = a // g, b // g, c // g
    for v in (a, b, c):
        if v != 0:
            if v < 0:
                a, b, c = -a, -b, -c
            break
    return (a, b, c)

def _is_valid(points, lines):
    try:
        p = np.asarray(points)
        l = np.asarray(lines)
    except Exception:
        return False
    if p.shape != (54, 2) or l.shape != (27, 3):
        return False
    for arr in (p, l):
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
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    vals = lns[:, 0, None, None] * pts[:, 0, None] + lns[:, 1, None, None] * pts[:, 1, None] + lns[:, 2, None, None]
    return int(np.sum(vals == 0))

def _line_intersection(l1, l2):
    A1, B1, C1 = l1
    A2, B2, C2 = l2
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None
    x_num = B1 * C2 - B2 * C1
    y_num = A2 * C1 - A1 * C2
    if x_num % det != 0 or y_num % det != 0:
        return None
    x = x_num // det
    y = y_num // det
    if abs(x) > 200 or abs(y) > 200:
        return None
    return (x, y)

def _egcd(a, b):
    if b == 0:
        return (abs(a), 1 if a > 0 else -1, 0)
    g, x1, y1 = _egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def _points_on_line(line, max_coord=200, limit=1000):
    A, B, C = line
    g = gcd(abs(A), abs(B))
    if g == 0 or C % g != 0:
        return []
    a = A // g
    b = B // g
    c = -C // g
    g2, x1, y1 = _egcd(a, b)
    if g2 != 1:
        # Should not happen since gcd(a,b)=1
        return []
    x0 = x1 * c
    y0 = y1 * c
    pts = []
    for t in range(-max_coord, max_coord + 1):
        x = x0 + b * t
        y = y0 - a * t
        if abs(x) <= max_coord and abs(y) <= max_coord:
            pts.append((x, y))
        if len(pts) >= limit:
            break
    return pts

def _build_pool(max_coord=12, c_range=40):
    pool = []
    seen = set()
    for A in range(-max_coord, max_coord + 1):
        for B in range(-max_coord, max_coord + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-c_range, c_range + 1):
                canon = _normalize_line(A, B, C)
                if canon and canon not in seen:
                    seen.add(canon)
                    pool.append(canon)
    return pool

def _greedy_line_selection(pool, n_lines=27, deadline=None):
    """
    Greedily select n_lines lines from pool.
    Score: maximize the number of integer intersection points (with multiplicity)
    that lie within a reasonable bound.
    """
    n_pool = len(pool)
    if n_pool < n_lines:
        return list(pool)

    # Precompute intersection points for all pairs to speed up scoring
    # This is O(N^2) which for N~2000 is 4M ops. Acceptable.
    # But we only need it for the greedy step.
    # Let's compute a "potential" score for each line: number of other lines it intersects with at an integer point.
    potential = [0] * n_pool
    intersection_cache = {}  # (i, j) -> point or None
    
    # Optimization: only compute for pairs that are "close" in index? No, just do it all if pool isn't too huge.
    # If pool is huge, we might need to subsample or limit.
    # Let's limit the pool size for this step if it's too big.
    if n_pool > 3000:
        # Randomly subsample to 3000 for the greedy selection phase
        pool = pool[:3000]
        n_pool = len(pool)
        potential = [0] * n_pool

    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            pt = _line_intersection(pool[i], pool[j])
            if pt is not None:
                potential[i] += 1
                potential[j] += 1
            # Cache for later? Not really needed if we just use potential for initial sort.

    # Sort indices by potential descending
    order = sorted(range(n_pool), key=lambda i: -potential[i])

    selected = []
    selected_set = set()
    
    # Greedy selection
    # Score of adding line L to current set S:
    # 1. Number of new intersection points created with lines in S.
    # 2. If an intersection point already exists (from other pairs in S), we don't gain a new point,
    #    but we gain an incidence for that point (if we later place a point there).
    #    Actually, the objective is to maximize incidences.
    #    If we have a set of lines, the max incidences is sum over points of (number of lines through point).
    #    This is equivalent to sum over pairs of lines of (1 if they intersect at an integer point).
    #    Wait, sum over pairs of lines of indicator(intersect) = sum over points of C(k_p, 2) where k_p is lines through p.
    #    This is NOT the same as sum k_p.
    #    However, maximizing sum C(k_p, 2) tends to maximize sum k_p as well (clustering).
    #    So greedy selection based on "how many new intersection points do I create" is a good heuristic.
    
    # Let's refine the score:
    # When adding line L, for each existing line L' in S:
    #   If L and L' intersect at P:
    #     If P is not already an intersection of two other lines in S, we gain 1 (new point with k=2).
    #     If P is already an intersection, we gain 0 new points, but the count k_p increases.
    #     Actually, the "value" of a point with k lines is k (incidences).
    #     Adding a line to a point with k lines increases total incidences by 1.
    #     So the delta in total incidences when adding line L is exactly the number of points on L that are intersections of pairs in S?
    #     No. If L intersects L' at P, and P is already on k other lines, then adding L increases the incidence count at P by 1.
    #     So the total gain in incidences is the number of existing intersection points on L.
    #     Plus, if L creates new intersection points (with L'), those points now have 2 lines.
    #     Wait, the incidence count I = sum_{p} k_p.
    #     Initially I=0.
    #     Add L1: I=0 (no pairs).
    #     Add L2: Intersect L1 at P. P has 2 lines. I = 2.
    #     Add L3: Intersect L1 at P1, L2 at P2.
    #       If P1 != P2: P1 has 2 lines, P2 has 2 lines. I = 2 (from P) + 2 (P1) + 2 (P2) = 6?
    #       No, P had 2 lines. Now P has 2 lines (L1, L2). L3 doesn't pass through P.
    #       So I = 2 (P) + 2 (P1) + 2 (P2) = 6.
    #       Previous I was 2. Delta = 4.
    #       Number of intersections of L3 with existing lines = 2.
    #       So delta is NOT just the number of intersections.
    #       
    #     Let's re-evaluate.
    #     I = sum_{p} k_p.
    #     When adding a new line L, it intersects existing lines at some set of points P_L.
    #     For each point P in P_L, k_P increases by 1. So I increases by 1 for each P in P_L.
    #     So Delta I = |P_L| = number of intersection points of L with existing lines.
    #     
    #     Therefore, to maximize total incidences, we should greedily pick the line that has the MOST intersections with the currently selected lines.
    
    # This simplifies the greedy score significantly!
    # Score(L) = number of lines in S such that L intersects line in S at an integer point.
    
    for idx in order:
        if len(selected) >= n_lines:
            break
        if idx in selected_set:
            continue
            
        if len(selected) == 0:
            # First line doesn't matter for intersections, pick highest potential
            selected.append(pool[idx])
            selected_set.add(idx)
            continue
        
        # Calculate score: intersections with existing selected lines
        score = 0
        for sel_idx in selected_set:
            pt = _line_intersection(pool[idx], pool[sel_idx])
            if pt is not None:
                score += 1
        
        # To make it more robust, we might want to prefer lines that intersect at DIFFERENT points?
        # No, if they intersect at the SAME point, that point gets a higher k, which is good for I.
        # Example: 3 lines through 1 point: I=3. 3 lines in triangle: I=6.
        # So we actually PREFER lines that intersect at DIFFERENT points (forming a grid) over concurrent lines.
        # Wait.
        # If L3 intersects L1 at P and L2 at P (same point P), then P has 3 lines. I = 3.
        # If L3 intersects L1 at P1 and L2 at P2 (different), then P1 has 2, P2 has 2, P (L1,L2) has 2. I = 6.
        # So we want to maximize the NUMBER of distinct intersection points?
        # No, we want to maximize I = sum k_p.
        # If we have a grid of lines, we get many points with k=2 or k=3.
        # If we have concurrent lines, we get few points with high k.
        # Generally, grids are better for I because C(n,2) grows faster?
        # Let's check: 27 lines.
        # Grid 3x9: 3 vertical, 9 horizontal.
        # Intersections: 27 points, each k=2 (1 vert, 1 horiz).
        # I = 27 * 2 = 54.
        # Concurrent: 27 lines through 1 point.
        # 1 point, k=27.
        # I = 27.
        # So grids are much better.
        
        # How to encourage grids?
        # The score "number of intersections" rewards both.
        # We need a penalty for multiple lines intersecting at the same point?
        # Or reward for NEW points?
        # Delta I when adding L:
        # For each existing line L', let P be intersection.
        # If P is new (not seen before), it contributes 2 to I (L and L').
        # If P is old (seen before, with k lines), it contributes 1 to I (L is the new one).
        # So Delta I = (# new points) * 2 + (# old points) * 1.
        # This is better than just counting intersections.
        
        # Let's implement this refined score.
        
        # We need to track which points are already "active" (intersection of >= 2 selected lines).
        # Actually, we can just compute the score on the fly by checking intersections.
        
        # Let's stick to the simpler "number of intersections" for now, but maybe add a tie-breaker for distinct points.
        # Or just use the refined score.
        
        # Refined Score Implementation:
        # We need to know if an intersection point P is already an intersection of two OTHER selected lines.
        # This requires checking pairs of selected lines.
        # Since n_lines is small (27), this is feasible.
        
        # Let's precompute intersections of selected lines?
        # Or just compute score by iterating selected lines and checking if the point is "new".
        
        # For each selected line L':
        #   P = intersect(L, L')
        #   if P is None: continue
        #   Check if P is intersection of any other pair (L_a, L_b) in selected.
        #   If yes: it's an "old" point (contributes 1).
        #   If no: it's a "new" point (contributes 2).
        
        # This is O(N_sel^2) per candidate. N_sel <= 27. 27^2 = 729.
        # Candidates ~ 3000. 3000 * 729 = 2M ops. Acceptable.
        
        # Let's implement this.
        
        # To speed up, we can maintain a set of "existing intersection points" from selected lines.
        # existing_pts = set of points that are intersections of pairs in selected.
        # If P in existing_pts: old. Else: new.
        
        # Let's restructure the loop to maintain this set.
        
        pass # Placeholder, will implement in the main loop below

    # Re-implementing the greedy loop with the refined score
    selected = []
    selected_set = set()
    existing_intersections = set()
    
    # Initial sort by potential is still a good heuristic for the first few lines
    for idx in order:
        if len(selected) >= n_lines:
            break
        if idx in selected_set:
            continue
        
        if len(selected) == 0:
            selected.append(pool[idx])
            selected_set.add(idx)
            continue
            
        # Calculate refined score
        score = 0
        new_pts_for_this_line = set()
        
        for sel_idx in selected_set:
            pt = _line_intersection(pool[idx], pool[sel_idx])
            if pt is not None:
                if pt in existing_intersections:
                    score += 1  # Old point, +1 incidence
                else:
                    score += 2  # New point, +2 incidences (this line and the sel line)
                    new_pts_for_this_line.add(pt)
        
        # Tie-breaker: prefer lines that create more NEW points?
        # The score already weights new points higher.
        
        # Accept
        selected.append(pool[idx])
        selected_set.add(idx)
        
        # Update existing_intersections
        # Add all intersections of the new line with existing lines
        # And intersections between existing lines that might have been formed?
        # No, existing_intersections should contain ALL points that are intersections of ANY pair in selected.
        # When we add a new line, we add its intersections with all existing lines.
        # We do NOT need to re-check intersections between existing lines, as they are already in the set.
        
        for sel_idx in selected_set:
            if sel_idx == idx:
                continue
            pt = _line_intersection(pool[idx], pool[sel_idx])
            if pt is not None:
                existing_intersections.add(pt)
        
        if deadline and time.time() > deadline:
            break
    
    # Fill remaining if needed
    if len(selected) < n_lines:
        for idx in order:
            if len(selected) >= n_lines:
                break
            if idx not in selected_set:
                selected.append(pool[idx])
                selected_set.add(idx)
                # Update existing_intersections
                for sel_idx in selected_set:
                    if sel_idx == idx:
                        continue
                    pt = _line_intersection(pool[idx], pool[sel_idx])
                    if pt is not None:
                        existing_intersections.add(pt)
    
    return selected[:n_lines]

def _place_points_greedy(lines, n_points=54):
    """
    Given a set of lines, find the n_points lattice points that maximize incidences.
    1. Find all integer intersection points of pairs of lines.
    2. Count how many lines pass through each intersection point.
    3. Sort points by count descending.
    4. Pick top n_points.
    5. If not enough, pick points on individual lines.
    """
    point_counts = defaultdict(int)
    
    n_lines = len(lines)
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(lines[i], lines[j])
            if pt is not None:
                point_counts[pt] += 1
    
    # Sort points by count descending
    sorted_points = sorted(point_counts.items(), key=lambda x: -x[1])
    
    points = []
    seen = set()
    
    # Add intersection points
    for pt, cnt in sorted_points:
        if len(points) >= n_points:
            break
        if pt not in seen:
            points.append(pt)
            seen.add(pt)
    
    # If we need more points, find points on individual lines
    if len(points) < n_points:
        for line in lines:
            A, B, C = line
            pts_on_line = _points_on_line(line, max_coord=200)
            for pt in pts_on_line:
                if pt not in seen:
                    points.append(pt)
                    seen.add(pt)
                    if len(points) >= n_points:
                        break
            if len(points) >= n_points:
                break
    
    # If still not enough, add arbitrary lattice points
    if len(points) < n_points:
        for x in range(-100, 101):
            for y in range(-100, 101):
                if len(points) >= n_points:
                    break
                pt = (x, y)
                if pt not in seen:
                    points.append(pt)
                    seen.add(pt)
    
    return points[:n_points]

def _local_search(points, lines, start_time, deadline):
    """
    Local search: swap a point to a nearby intersection, or swap a line to a nearby line.
    """
    rng = random.Random(42)
    pts = np.array(points, dtype=np.int64)
    lns = np.array(lines, dtype=np.int64)
    n_pts = 54
    n_lines = 27
    
    # Build incidence matrix
    inc = np.zeros((n_pts, n_lines), dtype=bool)
    for i in range(n_pts):
        for j in range(n_lines):
            A, B, C = lns[j]
            if A * pts[i, 0] + B * pts[i, 1] + C == 0:
                inc[i, j] = True
    
    total_inc = int(inc.sum())
    best_total = total_inc
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    # Candidate lines pool for swaps
    pool = _build_pool(max_coord=15, c_range=50)
    
    # Candidate points: intersections of current lines + nearby lattice
    cand_pts = set()
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(lns[i], lns[j])
            if pt is not None:
                cand_pts.add(pt)
    
    # Add nearby lattice points
    for x in range(-20, 21):
        for y in range(-20, 21):
            cand_pts.add((x, y))
            
    cand_pts = list(cand_pts)
    
    no_improve = 0
    T = 2.0
    
    while time.time() < deadline - 10 and no_improve < 1000:
        # Random move
        if rng.random() < 0.5:
            # Move point
            i = rng.randint(0, n_pts - 1)
            new_pt = cand_pts[rng.randint(0, len(cand_pts) - 1)]
            
            # Check duplicate
            if any(pts[k] == new_pt for k in range(n_pts) if k != i):
                continue
            
            old_pt = pts[i]
            pts[i] = new_pt
            
            # Update inc row i
            for j in range(n_lines):
                A, B, C = lns[j]
                inc[i, j] = (A * new_pt[0] + B * new_pt[1] + C == 0)
            
            new_total = int(inc.sum())
            
            if new_total >= total_inc or rng.random() < np.exp((new_total - total_inc) / T):
                total_inc = new_total
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                pts[i] = old_pt
                for j in range(n_lines):
                    A, B, C = lns[j]
                    inc[i, j] = (A * old_pt[0] + B * old_pt[1] + C == 0)
                no_improve += 1
        else:
            # Move line
            j = rng.randint(0, n_lines - 1)
            new_line = pool[rng.randint(0, len(pool) - 1)]
            
            # Check duplicate
            if any(_normalize_line(*lns[k]) == new_line for k in range(n_lines) if k != j):
                continue
            
            old_line = lns[j]
            lns[j] = new_line
            
            # Update inc col j
            for i in range(n_pts):
                A, B, C = new_line
                inc[i, j] = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
            
            new_total = int(inc.sum())
            
            if new_total >= total_inc or rng.random() < np.exp((new_total - total_inc) / T):
                total_inc = new_total
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                lns[j] = old_line
                for i in range(n_pts):
                    A, B, C = old_line
                    inc[i, j] = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
                no_improve += 1
        
        T *= 0.999
        
    return best_pts, best_lns, best_total

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600
    
    # 1. Generate a large pool of lines
    pool = _build_pool(max_coord=15, c_range=60)
    
    # 2. Greedy line selection
    selected_lines = _greedy_line_selection(pool, n_lines=27, deadline=deadline - 300)
    
    # 3. Greedy point placement
    points = _place_points_greedy(selected_lines, n_points=54)
    lines = np.array(selected_lines, dtype=np.int64)
    
    # 4. Local search / Simulated Annealing polish
    pts, lns, count = _local_search(points, lines, start_time, deadline)
    
    # 5. Validate and fallback
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
    if not _is_valid(pts, lns):
        # Fallback to a simple grid
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        line_counts = defaultdict(int)
        n_pts = len(pts)
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
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
            lns = np.array(selected, dtype=np.int64)
        else:
            # Absolute fallback
            lns = np.zeros((27, 3), dtype=np.int64)
            for i in range(27):
                lns[i] = [1, 0, i] # x = i
        
        if not _is_valid(pts, lns):
            # Last resort
            pts = np.zeros((54, 2), dtype=np.int64)
            for i in range(54):
                pts[i] = [i, 0]
            lns = np.zeros((27, 3), dtype=np.int64)
            for i in range(27):
                lns[i] = [0, 1, i] # y = i

    return pts, lns
