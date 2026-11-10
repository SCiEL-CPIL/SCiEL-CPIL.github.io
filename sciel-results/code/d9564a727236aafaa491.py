import numpy as np
import time
import itertools
import sys
from math import gcd
from collections import defaultdict

def _incidence_count(points, lines):
    """Compute exact incidence count. Returns int."""
    pts = np.asarray(points)
    lns = np.asarray(lines)
    x = pts[:, 0, None]
    y = pts[:, 1, None]
    A = lns[:, 0, None, None]
    B = lns[:, 1, None, None]
    C = lns[:, 2, None, None]
    vals = A * x + B * y + C
    return int(np.sum(vals == 0))

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

def _grid_construction(k=3):
    """Baseline grid: {1..k} x {1..2k^2}, lines y=ax+b for a in 1..k, b in 1..k^2."""
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

def _evaluate_config(points, lines):
    """Return (valid, incidence_count)."""
    if not _is_valid(points, lines):
        return False, 0
    return True, _incidence_count(points, lines)

def _generate_line_candidates(max_coeff=8, max_const=15):
    """Generate a diverse pool of candidate lines with small coefficients."""
    candidates = set()
    # y = ax + b => ax - y + b = 0
    for a in range(-max_coeff, max_coeff + 1):
        for b in range(-max_const, max_const + 1):
            if a != 0:
                canon = _normalize_line(a, -1, b)
                if canon:
                    candidates.add(canon)
    # x = c => x - c = 0
    for c in range(-max_const, max_const + 1):
        canon = _normalize_line(1, 0, -c)
        if canon:
            candidates.add(canon)
    # y = c => y - c = 0
    for c in range(-max_const, max_const + 1):
        canon = _normalize_line(0, 1, -c)
        if canon:
            candidates.add(canon)
    # General small-coefficient lines
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-max_const, max_const + 1):
                canon = _normalize_line(A, B, C)
                if canon:
                    candidates.add(canon)
    return list(candidates)

def _point_line_incidence(point, line):
    """Check if point lies on line: A*x + B*y + C == 0"""
    A, B, C = line
    x, y = point
    return A * x + B * y + C == 0

def _line_intersection(l1, l2):
    """Compute intersection of two lines. Returns (x, y) as integers if they intersect at integer point, else None."""
    A1, B1, C1 = l1
    A2, B2, C2 = l2
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None  # Parallel or coincident
    x_num = B1 * C2 - B2 * C1
    y_num = A2 * C1 - A1 * C2
    if x_num % det != 0 or y_num % det != 0:
        return None  # Non-integer intersection
    x = x_num // det
    y = y_num // det
    return (x, y)

def _count_incidences_for_point(point, lines):
    """Count how many lines pass through a given point."""
    count = 0
    for line in lines:
        if _point_line_incidence(point, line):
            count += 1
    return count

def _greedy_line_selection(candidate_lines, num_lines=27, max_points=54):
    """
    Greedily select lines to maximize total point coverage.
    Strategy: prioritize lines that create many new intersection points.
    """
    selected_lines = []
    covered_points = set()
    
    # Precompute intersections between all candidate line pairs
    n_cands = len(candidate_lines)
    intersections = defaultdict(set)  # line_idx -> set of intersection points
    
    for i in range(n_cands):
        for j in range(i + 1, n_cands):
            pt = _line_intersection(candidate_lines[i], candidate_lines[j])
            if pt is not None:
                intersections[i].add(pt)
                intersections[j].add(pt)
    
    # Greedy: repeatedly pick the line that adds the most new intersection points
    available = set(range(n_cands))
    
    for _ in range(num_lines):
        best_idx = None
        best_new_points = -1
        
        for idx in available:
            # Count new points this line would add
            new_pts = intersections[idx] - covered_points
            # Also consider that this line might be extended to include more points
            # For now, just use intersection points
            score = len(new_pts)
            if score > best_new_points:
                best_new_points = score
                best_idx = idx
        
        if best_idx is None:
            break
        
        selected_lines.append(candidate_lines[best_idx])
        covered_points.update(intersections[best_idx])
        available.remove(best_idx)
    
    return selected_lines, covered_points

def _place_points_greedy(lines, num_points=54, max_coord=20):
    """
    Place points greedily to maximize incidences with the given lines.
    Try all integer points in a bounded region, pick those with highest incidence count.
    """
    # Generate candidate points
    candidate_points = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidate_points.append((x, y))
    
    # Score each candidate point by how many lines it lies on
    scored_points = []
    for pt in candidate_points:
        count = _count_incidences_for_point(pt, lines)
        if count > 0:
            scored_points.append((count, pt))
    
    # Sort by incidence count descending
    scored_points.sort(key=lambda x: (-x[0], x[1]))
    
    # Greedily select points
    selected_points = []
    used_points = set()
    
    for count, pt in scored_points:
        if len(selected_points) >= num_points:
            break
        if pt not in used_points:
            selected_points.append(pt)
            used_points.add(pt)
    
    # If we don't have enough, fill with arbitrary points
    if len(selected_points) < num_points:
        for pt in candidate_points:
            if len(selected_points) >= num_points:
                break
            if pt not in used_points:
                selected_points.append(pt)
                used_points.add(pt)
    
    return selected_points[:num_points]

def _local_search(points, lines, deadline, max_iterations=1000):
    """
    Local search: try swapping individual points or lines to improve incidence count.
    """
    points = list(points)
    lines = list(lines)
    best_count = _incidence_count(np.array(points), np.array(lines))
    
    # Generate candidate points and lines for swaps
    cand_pts = [(x, y) for x in range(-15, 16) for y in range(-15, 16)]
    cand_lines = _generate_line_candidates(max_coeff=6, max_const=12)
    
    improved = True
    iters = 0
    while improved and iters < max_iterations and time.time() < deadline - 30:
        improved = False
        iters += 1
        
        # Try swapping one point
        for i in range(len(points)):
            if time.time() > deadline - 30:
                break
            best_swap = None
            best_swap_count = best_count
            for new_pt in cand_pts:
                if new_pt in points:
                    continue
                old_pt = points[i]
                points[i] = new_pt
                cnt = _incidence_count(np.array(points), np.array(lines))
                points[i] = old_pt
                if cnt > best_swap_count:
                    best_swap_count = cnt
                    best_swap = new_pt
            if best_swap is not None:
                points[i] = best_swap
                best_count = best_swap_count
                improved = True
        
        # Try swapping one line
        for i in range(len(lines)):
            if time.time() > deadline - 30:
                break
            best_swap = None
            best_swap_count = best_count
            old_canon = _normalize_line(*lines[i])
            for new_line in cand_lines:
                new_canon = _normalize_line(*new_line)
                if new_canon == old_canon:
                    continue
                # Check uniqueness
                dup = False
                for k in range(len(lines)):
                    if k == i:
                        continue
                    if _normalize_line(*lines[k]) == new_canon:
                        dup = True
                        break
                if dup:
                    continue
                old_line = lines[i]
                lines[i] = new_line
                cnt = _incidence_count(np.array(points), np.array(lines))
                lines[i] = old_line
                if cnt > best_swap_count:
                    best_swap_count = cnt
                    best_swap = new_line
            if best_swap is not None:
                lines[i] = best_swap
                best_count = best_swap_count
                improved = True
    
    return points, lines, best_count

def _combinatorial_greedy_construction(deadline):
    """
    Main combinatorial greedy construction:
    1. Generate candidate line pool
    2. Greedily select 27 lines to maximize intersection point coverage
    3. Place 54 points greedily to maximize incidences
    4. Local search refinement
    """
    # Step 1: Generate candidate lines
    candidate_lines = _generate_line_candidates(max_coeff=10, max_const=18)
    
    # Step 2: Greedily select 27 lines
    selected_lines, covered_points = _greedy_line_selection(candidate_lines, num_lines=27)
    
    # Step 3: Place 54 points
    points = _place_points_greedy(selected_lines, num_points=54, max_coord=25)
    
    # Convert to arrays
    points_arr = np.array(points, dtype=np.int64)
    lines_arr = np.array(selected_lines, dtype=np.int64)
    
    # Step 4: Local search refinement
    if time.time() < deadline - 60:
        points_ls, lines_ls, count_ls = _local_search(points_arr.tolist(), lines_arr.tolist(), deadline)
        if count_ls > _incidence_count(points_arr, lines_arr):
            points_arr = np.array(points_ls, dtype=np.int64)
            lines_arr = np.array(lines_ls, dtype=np.int64)
    
    return points_arr, lines_arr

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences with 54 points and 27 lines on integer lattice.
    Uses combinatorial greedy construction: select lines to maximize shared points,
    then place points optimally, then refine via local search.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s wall-clock
    
    # Start with baseline grid as fallback
    best_points, best_lines = _grid_construction(3)
    best_count = _incidence_count(best_points, best_lines)
    
    # Try combinatorial greedy construction
    if time.time() < deadline - 120:
        try:
            p_cg, l_cg = _combinatorial_greedy_construction(deadline)
            valid, count_cg = _evaluate_config(p_cg, l_cg)
            if valid and count_cg > best_count:
                best_points = p_cg
                best_lines = l_cg
                best_count = count_cg
        except Exception:
            pass  # Keep baseline
    
    # Additional local search if time permits
    if time.time() < deadline - 30:
        try:
            p_ls, l_ls, c_ls = _local_search(best_points.tolist(), best_lines.tolist(), deadline)
            if c_ls > best_count:
                best_points = np.array(p_ls, dtype=np.int64)
                best_lines = np.array(l_ls, dtype=np.int64)
                best_count = c_ls
        except Exception:
            pass
    
    # Final validation and safety
    best_points = np.array(best_points, dtype=np.int64)
    best_lines = np.array(best_lines, dtype=np.int64)
    
    if not _is_valid(best_points, best_lines):
        best_points, best_lines = _grid_construction(3)
    
    return best_points, best_lines
