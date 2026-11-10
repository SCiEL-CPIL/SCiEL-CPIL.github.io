import numpy as np
import time
import math
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) by dividing by gcd and making first nonzero positive."""
    A, B, C = int(A), int(B), int(C)
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 1:
        A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count exact integer incidences between points and lines."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    total = 0
    for i in range(len(lns)):
        A, B, C = lns[i]
        val = A * pts[:, 0] + B * pts[:, 1] + C
        total += int(np.sum(val == 0))
    return total

def _is_valid(points, lines):
    """Validate points and lines per the task contract."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    if pts.shape != (54, 2) or lns.shape != (27, 3):
        return False
    if not np.all(np.isfinite(pts.astype(np.float64))):
        return False
    if not np.all(np.isfinite(lns.astype(np.float64))):
        return False
    seen_pts = set()
    for p in pts:
        t = (int(p[0]), int(p[1]))
        if t in seen_pts:
            return False
        seen_pts.add(t)
    seen_lines = set()
    for l in lns:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm is None:
            return False
        if norm in seen_lines:
            return False
        seen_lines.add(norm)
    return True

def _build_grid_6x9():
    """Build the 6x9 grid (54 points)."""
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points

def _build_grid_9x9():
    """Build a 9x9 grid and select 54 points to maximize coverage of
    9 vertical + 9 horizontal + 9 diagonal lines.
    
    Strategy: use all 81 positions of a 9x9 grid. We pick 54 of them.
    The lines are:
    - 9 vertical: x = 0..8
    - 9 horizontal: y = 0..8
    - 9 diagonals: x + y = s for s in a chosen set of 9 values
    
    For each point (x, y), it lies on:
    - vertical x (always)
    - horizontal y (always)
    - diagonal x+y (if x+y in chosen set)
    
    So each point contributes 2 or 3 incidences. To maximize total:
    pick 54 points that maximize the number with x+y in the diagonal set.
    """
    # The 9 best diagonals by full-grid point count:
    # s=8: 9 pts, s=7,9: 8 each, s=6,10: 7 each, s=5,11: 6 each, s=4,12: 5 each
    diag_s = [8, 7, 9, 6, 10, 5, 11, 4, 12]
    diag_set = set(diag_s)
    
    # For each of 81 positions, compute: is it on a chosen diagonal?
    # We want to pick 54 positions maximizing count of "on diagonal" positions.
    # Positions on diagonals: for each s in diag_s, the points (x, s-x) with 0<=x<=8, 0<=s-x<=8
    
    # Count how many points are on each diagonal
    # Total points on chosen diagonals (with overlap): 
    # A point is on exactly one diagonal (since x+y is unique per point)
    # So no overlap between diagonals.
    
    # Points per diagonal:
    diag_points = {}
    for s in diag_s:
        pts_on = []
        for x in range(9):
            y = s - x
            if 0 <= y <= 8:
                pts_on.append((x, y))
        diag_points[s] = pts_on
    
    # Total points on diagonals:
    total_diag_pts = sum(len(v) for v in diag_points.values())
    # We need 54 points. If total_diag_pts >= 54, pick 54 from diagonals.
    # If total_diag_pts < 54, pick all diagonal points + (54 - total_diag_pts) non-diagonal.
    
    all_diag_pts = []
    for s in diag_s:
        all_diag_pts.extend(diag_points[s])
    
    # Non-diagonal points
    all_non_diag = []
    diag_pos_set = set(all_diag_pts)
    for x in range(9):
        for y in range(9):
            if (x, y) not in diag_pos_set:
                all_non_diag.append((x, y))
    
    if len(all_diag_pts) >= 54:
        selected = all_diag_pts[:54]
    else:
        selected = all_diag_pts + all_non_diag[:54 - len(all_diag_pts)]
    
    points = np.array(selected, dtype=np.int64)
    
    # Build the 27 lines
    lines = []
    # 9 vertical: x = c -> 1*x + 0*y - c = 0
    for c in range(9):
        lines.append((1, 0, -c))
    # 9 horizontal: y = r -> 0*x + 1*y - r = 0
    for r in range(9):
        lines.append((0, 1, -r))
    # 9 diagonals: x + y = s -> 1*x + 1*y - s = 0
    for s in diag_s:
        lines.append((1, 1, -s))
    
    lines_arr = np.array(lines, dtype=np.int64)
    
    return points, lines_arr

def _find_best_lines_for_points(points, max_lines=27):
    """Find the top max_lines lines (by individual incidence count) for a given set of points."""
    pts = np.asarray(points, dtype=np.int64)
    n = len(pts)
    
    # Generate all unique lines determined by pairs of points
    line_map = {}  # normalized line -> set of point indices
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            norm = _normalize_line(A, B, C)
            if norm is None:
                continue
            if norm not in line_map:
                line_map[norm] = set()
            line_map[norm].add(i)
            line_map[norm].add(j)
    
    # Count incidences for each line
    line_data = []
    for norm, indices in line_map.items():
        cnt = len(indices)
        line_data.append((cnt, norm))
    
    # Sort by count descending
    line_data.sort(key=lambda x: x[0], reverse=True)
    
    # Pick top max_lines
    selected = line_data[:max_lines]
    lines_arr = np.array([ln for _, ln in selected], dtype=np.int64)
    
    return lines_arr

def _fast_local_search(points, lines, deadline, max_rounds=100):
    """
    Fast local search: iteratively try to improve by:
    1. Replacing a point with a nearby candidate
    2. Replacing a line with a better one
    
    Uses incremental scoring for speed.
    """
    pts = np.array(points, dtype=np.int64, copy=True)
    lns = np.array(lines, dtype=np.int64, copy=True)
    n_pts = len(pts)
    n_lines = len(lns)
    
    current_score = _count_incidences(pts, lns)
    
    # Precompute: for each point, which lines it's on
    def compute_point_line_membership(pts, lns):
        """Return (point_lines, line_points) where point_lines[i] = set of line indices,
        line_points[j] = set of point indices."""
        point_lines = [set() for _ in range(len(pts))]
        line_points = [set() for _ in range(len(lns))]
        for j in range(len(lns)):
            A, B, C = lns[j]
            val = A * pts[:, 0] + B * pts[:, 1] + C
            for i in range(len(pts)):
                if val[i] == 0:
                    point_lines[i].add(j)
                    line_points[j].add(i)
        return point_lines, line_points
    
    # Generate candidate positions in an expanded region
    min_x, max_x = int(min(pts[:, 0])), int(max(pts[:, 0]))
    min_y, max_y = int(min(pts[:, 1])), int(max(pts[:, 1]))
    expand = 4
    all_candidates = set()
    for x in range(min_x - expand, max_x + expand + 1):
        for y in range(min_y - expand, max_y + expand + 1):
            all_candidates.add((x, y))
    
    current_pos_set = set((int(p[0]), int(p[1])) for p in pts)
    candidate_list = list(all_candidates - current_pos_set)
    
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline - 30:
        improved = False
        rounds += 1
        
        # --- Phase 1: Try moving points ---
        for i in range(n_pts):
            if time.time() >= deadline - 30:
                break
            orig_p = pts[i].copy()
            x0, y0 = int(orig_p[0]), int(orig_p[1])
            
            # Try nearby candidates (sorted by distance)
            nearby = []
            for c in candidate_list:
                dx = c[0] - x0
                dy = c[1] - y0
                d2 = dx*dx + dy*dy
                if d2 <= 25:  # within radius ~5
                    nearby.append((d2, c))
            nearby.sort()
            
            for d2, cand in nearby[:30]:
                if time.time() >= deadline - 30:
                    break
                new_pts = pts.copy()
                new_pts[i] = cand
                new_score = _count_incidences(new_pts, lns)
                if new_score > current_score:
                    pts = new_pts
                    candidate_list.remove(cand)
                    candidate_list.append((x0, y0))
                    current_score = new_score
                    improved = True
                    break
        
        # --- Phase 2: Try replacing lines ---
        if time.time() < deadline - 30:
            # Find all lines through pairs of current points
            unique_lines = set()
            for a in range(n_pts):
                for b in range(a + 1, n_pts):
                    x1, y1 = pts[a]
                    x2, y2 = pts[b]
                    A = y1 - y2
                    B = x2 - x1
                    C = x1 * y2 - x2 * y1
                    norm = _normalize_line(A, B, C)
                    if norm is not None:
                        unique_lines.add(norm)
            
            # Remove current lines
            current_norms = set()
            for l in lns:
                norm = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
                if norm is not None:
                    current_norms.add(norm)
            
            for norm in current_norms:
                unique_lines.discard(norm)
            
            # Score candidate lines
            line_scores = []
            for A, B, C in unique_lines:
                val = A * pts[:, 0] + B * pts[:, 1] + C
                cnt = int(np.sum(val == 0))
                line_scores.append((cnt, (A, B, C)))
            
            line_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Try replacing worst line with best candidate
            if line_scores:
                # Find the line with lowest incidence
                line_incs = []
                for j in range(n_lines):
                    A, B, C = lns[j]
                    val = A * pts[:, 0] + B * pts[:, 1] + C
                    cnt = int(np.sum(val == 0))
                    line_incs.append((cnt, j))
                line_incs.sort()
                
                worst_cnt, worst_idx = line_incs[0]
                best_cnt, best_line = line_scores[0]
                
                if best_cnt > worst_cnt:
                    new_ln = lns.copy()
                    new_ln[worst_idx] = best_line
                    new_score = _count_incidences(pts, new_ln)
                    if new_score > current_score:
                        lns = new_ln
                        current_score = new_score
                        improved = True
    
    return pts, lns, current_score

def _try_9x9_with_optimal_lines(deadline):
    """Build 9x9 grid points, then find optimal 27 lines."""
    pts, lns = _build_grid_9x9()
    score = _count_incidences(pts, lns)
    
    # Try to improve lines
    if time.time() < deadline - 60:
        lns_opt = _find_best_lines_for_points(pts, 27)
        score_opt = _count_incidences(pts, lns_opt)
        if score_opt > score:
            lns = lns_opt
            score = score_opt
    
    # Local search
    if time.time() < deadline - 30:
        pts_opt, lns_opt, score_opt = _fast_local_search(pts, lns, deadline)
        if score_opt > score:
            pts, lns, score = pts_opt, lns_opt, score_opt
    
    return pts, lns, score

def _try_6x9_with_optimal_lines(deadline):
    """Build 6x9 grid points, then find optimal 27 lines."""
    pts = _build_grid_6x9()
    lns = _find_best_lines_for_points(pts, 27)
    score = _count_incidences(pts, lns)
    
    # Local search
    if time.time() < deadline - 30:
        pts_opt, lns_opt, score_opt = _fast_local_search(pts, lns, deadline)
        if score_opt > score:
            pts, lns, score = pts_opt, lns_opt, score_opt
    
    return pts, lns, score

def _try_7x8_grid(deadline):
    """Try a 7x8 grid (56 positions, pick 54) with optimal lines."""
    xs = np.arange(7)
    ys = np.arange(8)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    all_pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Pick 54 of 56: remove the 2 with lowest "diagonal coverage"
    # Actually just pick first 54
    pts = all_pts[:54]
    lns = _find_best_lines_for_points(pts, 27)
    score = _count_incidences(pts, lns)
    
    if time.time() < deadline - 30:
        pts_opt, lns_opt, score_opt = _fast_local_search(pts, lns, deadline)
        if score_opt > score:
            pts, lns, score = pts_opt, lns_opt, score_opt
    
    return pts, lns, score

def _try_3x18_grid(deadline):
    """Try a 3x18 grid (54 points) with optimal lines."""
    xs = np.arange(3)
    ys = np.arange(18)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    lns = _find_best_lines_for_points(pts, 27)
    score = _count_incidences(pts, lns)
    
    if time.time() < deadline - 30:
        pts_opt, lns_opt, score_opt = _fast_local_search(pts, lns, deadline)
        if score_opt > score:
            pts, lns, score = pts_opt, lns_opt, score_opt
    
    return pts, lns, score

def szemeredi_trotter_incidences():
    deadline = time.time() + 580.0
    
    best_pts = None
    best_lines = None
    best_score = -1
    
    # Try multiple initial configurations
    configs = [
        ("9x9", _try_9x9_with_optimal_lines),
        ("6x9", _try_6x9_with_optimal_lines),
        ("7x8", _try_7x8_grid),
        ("3x18", _try_3x18_grid),
    ]
    
    for name, builder in configs:
        if time.time() >= deadline - 30:
            break
        try:
            pts, lns, score = builder(deadline)
            if _is_valid(pts, lns):
                if score > best_score:
                    best_pts = pts
                    best_lines = lns
                    best_score = score
        except Exception:
            pass
    
    # Final local search on best configuration
    if best_pts is not None and time.time() < deadline - 10:
        try:
            pts_opt, lns_opt, score_opt = _fast_local_search(best_pts, best_lines, deadline)
            if _is_valid(pts_opt, lns_opt) and score_opt > best_score:
                best_pts, best_lines, best_score = pts_opt, lns_opt, score_opt
        except Exception:
            pass
    
    if best_pts is not None:
        return best_pts, best_lines
    
    # Fallback
    pts = _build_grid_6x9()
    lns = _find_best_lines_for_points(pts, 27)
    return pts, lns
