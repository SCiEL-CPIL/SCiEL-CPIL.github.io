import numpy as np
import time
import random
from math import gcd
from collections import defaultdict
from itertools import combinations

def _normalize_line(a, b, c):
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
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    vals = lns[:, 0, None, None] * pts[:, 0, None] + lns[:, 1, None, None] * pts[:, 1, None] + lns[:, 2, None, None]
    return int(np.sum(vals == 0))

def _line_intersection(l1, l2):
    """Return integer point (x,y) if two lines intersect at an integer point, else None."""
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
    if abs(x) > 100 or abs(y) > 100:
        return None
    return (x, y)

def _greedy_line_selection(pool, n_lines=27, deadline=None):
    """
    Greedily select n_lines lines from pool to maximize sum of pairwise intersection multiplicities.
    Score of a candidate line given current selection = sum over existing lines of (1 if they intersect at an integer point within bounds).
    This approximates the number of unique intersection points created.
    """
    selected = []
    selected_set = set()
    
    n_pool = len(pool)
    if n_pool < n_lines:
        return list(pool)
    
    # Compute potential score for each line: number of other lines it intersects with
    potential = [0] * n_pool
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            pt = _line_intersection(pool[i], pool[j])
            if pt is not None:
                potential[i] += 1
                potential[j] += 1
    
    # Sort indices by potential descending
    order = sorted(range(n_pool), key=lambda i: -potential[i])
    
    # Greedy selection
    for idx in order:
        if len(selected) >= n_lines:
            break
        if idx in selected_set:
            continue
        selected.append(pool[idx])
        selected_set.add(idx)
        
        if deadline and time.time() > deadline:
            break
    
    # If we didn't fill up, add remaining
    if len(selected) < n_lines:
        for idx in order:
            if len(selected) >= n_lines:
                break
            if idx not in selected_set:
                selected.append(pool[idx])
                selected_set.add(idx)
    
    return selected[:n_lines]

def _place_points_greedy(lines, n_points=54):
    """
    Given a set of lines, find the n_points lattice points that maximize incidences.
    Each point's value = number of lines it lies on.
    """
    point_count = defaultdict(int)
    
    n_lines = len(lines)
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(lines[i], lines[j])
            if pt is not None:
                point_count[pt] += 1
    
    sorted_points = sorted(point_count.items(), key=lambda x: -x[1])
    
    points = []
    seen = set()
    
    for pt, cnt in sorted_points:
        if len(points) >= n_points:
            break
        if pt not in seen:
            points.append(pt)
            seen.add(pt)
    
    if len(points) < n_points:
        for line in lines:
            A, B, C = line
            g = gcd(abs(A), abs(B))
            if g == 0 or C % g != 0:
                continue
            a = A // g
            b = B // g
            c = -C // g
            
            def egcd(a, b):
                if b == 0:
                    return (a, 1, 0)
                g, x1, y1 = egcd(b, a % b)
                return (g, y1, x1 - (a // b) * y1)
            
            g2, x1, y1 = egcd(a, b)
            x0 = x1 * c
            y0 = y1 * c
            for t in range(-50, 51):
                x = x0 + b * t
                y = y0 - a * t
                pt = (x, y)
                if pt not in seen and abs(x) <= 100 and abs(y) <= 100:
                    points.append(pt)
                    seen.add(pt)
                    if len(points) >= n_points:
                        break
            if len(points) >= n_points:
                break
    
    if len(points) < n_points:
        for x in range(-50, 51):
            for y in range(-50, 51):
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
    
    inc = np.zeros((n_pts, n_lines), dtype=bool)
    for i in range(n_pts):
        for j in range(n_lines):
            A, B, C = lns[j]
            inc[i, j] = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
    
    total_inc = int(inc.sum())
    best_total = total_inc
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    cand_points = set()
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(tuple(lns[i]), tuple(lns[j]))
            if pt:
                cand_points.add(pt)
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        cand_points.add((pt[0] + dx, pt[1] + dy))
    
    cand_lines = set()
    for A in range(-10, 11):
        for B in range(-10, 11):
            if A == 0 and B == 0:
                continue
            for C in range(-30, 31):
                canon = _normalize_line(A, B, C)
                if canon:
                    cand_lines.add(canon)
    
    cand_points = list(cand_points)
    cand_lines = list(cand_lines)
    
    T_start = 3.0
    T_end = 0.1
    iter_count = 0
    n_iters = 100000
    
    while time.time() < deadline - 10:
        iter_count += 1
        T = T_start * (T_end / T_start) ** (min(iter_count, n_iters) / n_iters)
        
        if rng.random() < 0.5:
            i = rng.randint(0, n_pts - 1)
            old_pos = tuple(pts[i])
            new_pos = cand_points[rng.randint(0, len(cand_points) - 1)]
            
            if new_pos == old_pos:
                continue
            if any(tuple(pts[k]) == new_pos for k in range(n_pts) if k != i):
                continue
            
            delta = 0
            for j in range(n_lines):
                was = inc[i, j]
                A, B, C = lns[j]
                now = (A * new_pos[0] + B * new_pos[1] + C == 0)
                delta += (1 if now else 0) - (1 if was else 0)
            
            new_total = total_inc + delta
            if new_total >= total_inc or rng.random() < np.exp(delta / max(T, 1e-10)):
                total_inc = new_total
                pts[i] = new_pos
                for j in range(n_lines):
                    A, B, C = lns[j]
                    inc[i, j] = (A * new_pos[0] + B * new_pos[1] + C == 0)
                
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
        else:
            j = rng.randint(0, n_lines - 1)
            old_canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            new_line = cand_lines[rng.randint(0, len(cand_lines) - 1)]
            new_canon = _normalize_line(int(new_line[0]), int(new_line[1]), int(new_line[2]))
            
            if new_canon is None or new_canon == old_canon:
                continue
            if any(_normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2])) == new_canon for k in range(n_lines) if k != j):
                continue
            
            delta = 0
            A, B, C = new_line
            for i in range(n_pts):
                was = inc[i, j]
                now = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
                delta += (1 if now else 0) - (1 if was else 0)
            
            new_total = total_inc + delta
            if new_total >= total_inc or rng.random() < np.exp(delta / max(T, 1e-10)):
                total_inc = new_total
                lns[j] = new_line
                for i in range(n_pts):
                    inc[i, j] = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
                
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
    
    return best_pts, best_lns, best_total

def _build_line_pool(max_coeff=8, max_const=40):
    """Build a diverse pool of candidate lines with small integer coefficients."""
    pool = []
    seen = set()
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-max_const, max_const + 1):
                canon = _normalize_line(A, B, C)
                if canon and canon not in seen:
                    seen.add(canon)
                    pool.append(canon)
    return pool

def _evaluate_configuration(points, lines):
    """Count incidences between points and lines."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    vals = lns[:, 0, None] * pts[None, :, 0] + lns[:, 1, None] * pts[None, :, 1] + lns[:, 2, None]
    return int(np.sum(vals == 0))

def _select_lines_for_points(points, line_pool, n_lines=27, deadline=None):
    """
    Select n_lines from line_pool that maximize total incidences with the given points.
    This is the dual-space approach: fix points, optimize lines.
    """
    pts = np.asarray(points, dtype=np.int64)
    n_pts = len(pts)
    
    # Precompute incidence count for each candidate line
    line_scores = []
    for idx, line in enumerate(line_pool):
        A, B, C = line
        # Count how many points lie on this line
        count = 0
        for i in range(n_pts):
            if A * pts[i, 0] + B * pts[i, 1] + C == 0:
                count += 1
        line_scores.append((count, idx))
    
    # Sort by score descending
    line_scores.sort(key=lambda x: -x[0])
    
    # Greedily select lines, tracking which points are already "covered" by high-multiplicity lines
    # to encourage diversity
    selected = []
    selected_set = set()
    
    # Track point coverage: how many selected lines pass through each point
    point_coverage = np.zeros(n_pts, dtype=np.int32)
    
    # First pass: select top-scoring lines
    for count, idx in line_scores:
        if len(selected) >= n_lines:
            break
        if idx in selected_set:
            continue
        if deadline and time.time() > deadline:
            break
        
        # Check if this line adds new incidences (not all its points are already covered by other selected lines)
        A, B, C = line_pool[idx]
        new_incidences = 0
        for i in range(n_pts):
            if A * pts[i, 0] + B * pts[i, 1] + C == 0:
                # This point is on the line; check if it's already well-covered
                if point_coverage[i] == 0:
                    new_incidences += 1
        
        # Accept if it has any new incidences or if we haven't filled up yet
        if new_incidences > 0 or len(selected) < n_lines:
            selected.append(line_pool[idx])
            selected_set.add(idx)
            
            # Update point coverage
            for i in range(n_pts):
                if A * pts[i, 0] + B * pts[i, 1] + C == 0:
                    point_coverage[i] += 1
    
    # If not enough, fill from remaining
    if len(selected) < n_lines:
        for count, idx in line_scores:
            if len(selected) >= n_lines:
                break
            if idx not in selected_set:
                selected.append(line_pool[idx])
                selected_set.add(idx)
    
    return selected[:n_lines]

def _optimize_points_given_lines(lines, n_points=54, deadline=None):
    """
    Given fixed lines, find n_points that maximize total incidences.
    Each candidate point's value = number of lines it lies on.
    """
    lns = np.asarray(lines, dtype=np.int64)
    n_lines = len(lns)
    
    # Find all integer intersection points of pairs of lines
    point_line_count = defaultdict(int)
    
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(tuple(lns[i]), tuple(lns[j]))
            if pt is not None:
                point_line_count[pt] += 1
    
    # Sort by count descending
    sorted_points = sorted(point_line_count.items(), key=lambda x: -x[1])
    
    points = []
    seen = set()
    
    for pt, cnt in sorted_points:
        if len(points) >= n_points:
            break
        if pt not in seen:
            points.append(pt)
            seen.add(pt)
    
    # Fill remaining with points on individual lines
    if len(points) < n_points:
        for line in lines:
            A, B, C = line
            g = gcd(abs(A), abs(B))
            if g == 0 or C % g != 0:
                continue
            a = A // g
            b = B // g
            c = -C // g
            
            def egcd(a, b):
                if b == 0:
                    return (a, 1, 0)
                g, x1, y1 = egcd(b, a % b)
                return (g, y1, x1 - (a // b) * y1)
            
            g2, x1, y1 = egcd(a, b)
            x0 = x1 * c
            y0 = y1 * c
            for t in range(-50, 51):
                x = x0 + b * t
                y = y0 - a * t
                pt = (x, y)
                if pt not in seen and abs(x) <= 100 and abs(y) <= 100:
                    points.append(pt)
                    seen.add(pt)
                    if len(points) >= n_points:
                        break
            if len(points) >= n_points:
                break
    
    # Fill with arbitrary lattice points if needed
    if len(points) < n_points:
        for x in range(-50, 51):
            for y in range(-50, 51):
                if len(points) >= n_points:
                    break
                pt = (x, y)
                if pt not in seen:
                    points.append(pt)
                    seen.add(pt)
    
    return points[:n_points]

def _alternating_optimization(line_pool, n_points=54, n_lines=27, deadline=None, max_iter=20):
    """
    Dual-space hypergraph approach:
    1. Start with an initial configuration
    2. Fix lines, optimize points to maximize incidences
    3. Fix points, select lines that best cover those points
    4. Repeat until convergence or deadline
    """
    rng = random.Random(123)
    
    # Initial configuration: greedy line selection then point placement
    initial_lines = _greedy_line_selection(line_pool, n_lines, deadline)
    points = _optimize_points_given_lines(initial_lines, n_points, deadline)
    
    best_score = _evaluate_configuration(points, initial_lines)
    best_points = [p for p in points]
    best_lines = [l for l in initial_lines]
    
    for iteration in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Step 1: Fix points, select best lines
        new_lines = _select_lines_for_points(points, line_pool, n_lines, deadline)
        
        # Step 2: Fix lines, optimize points
        new_points = _optimize_points_given_lines(new_lines, n_points, deadline)
        
        # Evaluate
        score = _evaluate_configuration(new_points, new_lines)
        
        if score > best_score:
            best_score = score
            best_points = [p for p in new_points]
            best_lines = [l for l in new_lines]
            points = new_points
            lines = new_lines
        else:
            # Try a perturbation: randomly swap one point or line
            if rng.random() < 0.5:
                # Perturb a point
                i = rng.randint(0, len(points) - 1)
                # Move to a nearby intersection
                cand_pts = set()
                for a in range(n_lines):
                    for b in range(a + 1, n_lines):
                        pt = _line_intersection(tuple(lines[a]), tuple(lines[b]))
                        if pt:
                            cand_pts.add(pt)
                            for dx in range(-2, 3):
                                for dy in range(-2, 3):
                                    cand_pts.add((pt[0] + dx, pt[1] + dy))
                cand_pts = list(cand_pts)
                if cand_pts:
                    new_pt = cand_pts[rng.randint(0, len(cand_pts) - 1)]
                    if new_pt != points[i]:
                        # Check duplicate
                        if not any(p == new_pt for k, p in enumerate(points) if k != i):
                            points[i] = new_pt
                            score = _evaluate_configuration(points, lines)
                            if score > best_score:
                                best_score = score
                                best_points = [p for p in points]
                                best_lines = [l for l in lines]
            else:
                # Perturb a line
                j = rng.randint(0, n_lines - 1)
                cand_line = line_pool[rng.randint(0, len(line_pool) - 1)]
                if _normalize_line(*cand_line) != _normalize_line(*lines[j]):
                    # Check duplicate
                    if not any(_normalize_line(*l) == _normalize_line(*cand_line) for k, l in enumerate(lines) if k != j):
                        lines[j] = cand_line
                        score = _evaluate_configuration(points, lines)
                        if score > best_score:
                            best_score = score
                            best_points = [p for p in points]
                            best_lines = [l for l in lines]
    
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600
    
    # Build a larger, more diverse line pool
    line_pool = _build_line_pool(max_coeff=8, max_const=40)
    
    # Phase 1: Alternating optimization (dual-space hypergraph approach)
    if time.time() < deadline - 180:
        alt_points, alt_lines, alt_score = _alternating_optimization(
            line_pool, n_points=54, n_lines=27, deadline=deadline - 180, max_iter=30
        )
        pts = np.array(alt_points, dtype=np.int64)
        lines = np.array(alt_lines, dtype=np.int64)
    else:
        # Fallback to original approach
        selected_lines = _greedy_line_selection(line_pool, 27, deadline - 120)
        points = _place_points_greedy(selected_lines, 54)
        pts = np.array(points, dtype=np.int64)
        lines = np.array(selected_lines, dtype=np.int64)
    
    # Phase 2: Local search polish
    if time.time() < deadline - 60:
        ls_pts, ls_lns, ls_count = _local_search(pts, lines, start_time, deadline - 30)
        pts = ls_pts
        lines = ls_lns
    
    # Final validation
    pts = np.array(pts, dtype=np.int64)
    lines = np.array(lines, dtype=np.int64)
    
    if not _is_valid(pts, lines):
        # Fallback: simple grid
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = np.array([(1, 0, -i) for i in range(6)] + [(0, 1, -j) for j in range(9)] + [(1, 1, -k) for k in range(12)], dtype=np.int64)
    
    return pts, lines
