# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import combinations

_T0 = time.time()
_DEADLINE = 620.0  # safety margin vs 630s kill

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _canon_line(A, B, C):
    """Canonical form: divide by gcd, make first nonzero positive."""
    A, B, C = int(A), int(B), int(C)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences_fast(points, lines):
    """Vectorized incidence count."""
    px = points[:, 0].astype(np.int64)
    py = points[:, 1].astype(np.int64)
    A = lines[:, 0].astype(np.int64)
    B = lines[:, 1].astype(np.int64)
    C = lines[:, 2].astype(np.int64)
    # Compute all incidences at once using broadcasting
    # shape: (m, n) where m=len(lines), n=len(points)
    vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
    return int(np.sum(vals == 0))

def _valid_points(points):
    if len(points) != 54:
        return False
    seen = set()
    for p in points:
        x, y = int(p[0]), int(p[1])
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return True

def _valid_lines(lines):
    if len(lines) != 27:
        return False
    seen = set()
    for l in lines:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            return False
        c = _canon_line(A, B, C)
        if c in seen:
            return False
        seen.add(c)
    return True

def _is_valid(points, lines):
    return _valid_points(points) and _valid_lines(lines)

def _compute_all_lines(points):
    """Compute all distinct lines through pairs of points with their incidence counts."""
    P = list(points)
    n = len(P)
    idx = np.array(P, dtype=np.int64)
    line_info = {}
    
    for i in range(n):
        x1, y1 = P[i]
        for j in range(i + 1, n):
            x2, y2 = P[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            ct = _canon_line(A, B, C)
            if ct not in line_info:
                A_, B_, C_ = ct
                mask = (A_ * idx[:, 0] + B_ * idx[:, 1] + C_ == 0)
                cnt = int(np.sum(mask))
                line_info[ct] = (A_, B_, C_, cnt)
    return line_info

def _greedy_select(line_info, n_lines=27):
    """Greedy: pick top-n_lines by incidence count."""
    cand_list = sorted(line_info.items(), key=lambda kv: -kv[1][3])
    chosen = []
    total = 0
    for ct, (A_, B_, C_, cnt) in cand_list:
        if len(chosen) >= n_lines:
            break
        chosen.append(ct)
        total += cnt
    return chosen, total

def _solve_one(points, n_lines=27):
    """Solve: compute all lines, greedy select."""
    line_info = _compute_all_lines(points)
    if not line_info:
        return [], 0
    # Sort by count descending, break ties by canonical form for determinism
    cand_list = sorted(line_info.items(), key=lambda kv: (-kv[1][3], kv[0]))
    chosen = []
    total = 0
    for ct, (A_, B_, C_, cnt) in cand_list:
        if len(chosen) >= n_lines:
            break
        chosen.append(ct)
        total += cnt
    chosen_coef = [line_info[ct][:3] for ct in chosen]
    return chosen_coef, total

def _szemeredi_baseline():
    k = 3
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

def _hypergraph_local_search(points, deadline, max_iters=10000):
    """
    Dual-space hypergraph local search.
    
    Vertices: candidate integer points in a bounding box around current points.
    Edges: lines determined by pairs of selected points, with size = number of
           selected points on that line.
    
    Objective: maximize sum of top-27 edge sizes.
    
    Move: swap one selected point with one candidate point, accept if it
          increases the objective (recomputed exactly).
    """
    n = len(points)
    pts = [(int(p[0]), int(p[1])) for p in points]
    selected = set(pts)
    
    # Build candidate set: current points plus neighbors in a small radius
    x_min = min(p[0] for p in pts) - 2
    x_max = max(p[0] for p in pts) + 2
    y_min = min(p[1] for p in pts) - 2
    y_max = max(p[1] for p in pts) + 2
    candidates = [(x, y) for x in range(x_min, x_max + 1)
                  for y in range(y_min, y_max + 1)
                  if (x, y) not in selected]
    
    # Current best
    lines_arr, total = _solve_one(pts)
    best_score = total
    best_pts = list(pts)
    best_lines_arr = lines_arr
    
    if best_score == 0:
        return best_pts, best_lines_arr, best_score
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try swapping each selected point with a random subset of candidates
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Try a subset of candidates (randomized for diversity)
            cand_trials = random.sample(candidates, min(15, len(candidates)))
            
            for new_pt in cand_trials:
                if new_pt in selected:
                    continue
                
                new_pts = list(pts)
                new_pts[i] = new_pt
                
                # Recompute exactly
                new_lines_arr, new_score = _solve_one(new_pts)
                
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts
                    pts = new_pts
                    selected = set(pts)
                    best_lines_arr = new_lines_arr
                    
                    # Update candidates
                    x_min = min(p[0] for p in pts) - 2
                    x_max = max(p[0] for p in pts) + 2
                    y_min = min(p[1] for p in pts) - 2
                    y_max = max(p[1] for p in pts) + 2
                    candidates = [(x, y) for x in range(x_min, x_max + 1)
                                  for y in range(y_min, y_max + 1)
                                  if (x, y) not in selected]
                    
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            random.shuffle(candidates)
    
    # Final exact computation
    lines_arr, total = _solve_one(best_pts)
    if total > best_score:
        best_score = total
    
    return best_pts, lines_arr, best_score

def _local_search(points, lines, deadline, max_iter=5000):
    """
    Local search: try moving individual points or replacing lines.
    Returns improved (points, lines, count).
    """
    count = _count_incidences_fast(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_count = count
    
    n = len(points)
    m = len(lines)
    
    # Precompute line incidence masks for faster evaluation
    # For each line, which points are on it
    px = points[:, 0].astype(np.int64)
    py = points[:, 1].astype(np.int64)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try moving each point with larger search radius
        for i in range(n):
            if time.time() > deadline:
                break
            x, y = int(points[i, 0]), int(points[i, 1])
            
            # Try a set of promising moves
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    new_points = points.copy()
                    new_points[i] = [nx, ny]
                    
                    if not _valid_points(new_points):
                        continue
                    
                    new_count = _count_incidences_fast(new_points, lines)
                    if new_count > best_count:
                        best_count = new_count
                        best_points = new_points
                        improved = True
                        points = new_points
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try replacing individual lines with better ones from point pairs
        for j in range(m):
            if time.time() > deadline:
                break
            # Try more candidates
            for _ in range(20):
                i1, i2 = random.sample(range(n), 2)
                x1, y1 = int(points[i1, 0]), int(points[i1, 1])
                x2, y2 = int(points[i2, 0]), int(points[i2, 1])
                
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                
                if A == 0 and B == 0:
                    continue
                
                new_lines = lines.copy()
                new_lines[j] = [A, B, C]
                
                if not _valid_lines(new_lines):
                    continue
                
                new_count = _count_incidences_fast(points, new_lines)
                if new_count > best_count:
                    best_count = new_count
                    best_lines = new_lines
                    improved = True
                    lines = new_lines
                    break
            if improved:
                break
        
        if not improved:
            # Try moving multiple points with larger steps
            for _ in range(10):
                if time.time() > deadline:
                    break
                idxs = random.sample(range(n), 2)
                new_points = points.copy()
                for idx in idxs:
                    x, y = int(new_points[idx, 0]), int(new_points[idx, 1])
                    dx, dy = random.randint(-5, 5), random.randint(-5, 5)
                    new_points[idx] = [x + dx, y + dy]
                
                if not _valid_points(new_points):
                    continue
                
                new_count = _count_incidences_fast(new_points, lines)
                if new_count > best_count:
                    best_count = new_count
                    best_points = new_points
                    points = new_points
                    improved = True
                    break
            
            if not improved:
                break
    
    return best_points, best_lines, best_count

def _grid_search(deadline):
    """
    Try various grid configurations and optimize.
    """
    best_points = None
    best_lines = None
    best_count = 0
    
    grid_configs = [
        (range(0, 6), range(0, 9)),    # 5x9 = 45
        (range(0, 9), range(0, 6)),    # 9x5 = 45
        (range(0, 6), range(0, 10)),   # 6x10 = 60
        (range(0, 10), range(0, 6)),   # 10x6 = 60
        (range(0, 7), range(0, 8)),    # 7x8 = 56
        (range(0, 8), range(0, 7)),    # 8x7 = 56
        (range(0, 9), range(0, 7)),    # 9x7 = 63
        (range(0, 7), range(0, 9)),    # 7x9 = 63
    ]
    
    for x_range, y_range in grid_configs:
        if time.time() > deadline:
            break
        
        xs = list(x_range)
        ys = list(y_range)
        grid_points = [(x, y) for x in xs for y in ys]
        
        if len(grid_points) < 54:
            continue
        
        # Try multiple random selections
        for trial in range(5):
            if time.time() > deadline:
                break
            
            random.shuffle(grid_points)
            points = np.array(grid_points[:54], dtype=np.int64)
            
            # Use the new solve_one to get lines
            line_list, total = _solve_one(points.tolist())
            if len(line_list) < 27:
                continue
            
            lines = np.array(line_list[:27], dtype=np.int64)
            
            if not _is_valid(points, lines):
                continue
            
            # Local search
            p, l, c = _local_search(points, lines, deadline, max_iter=2000)
            
            if c > best_count:
                best_count = c
                best_points = p
                best_lines = l
    
    return best_points, best_lines, best_count

def _smart_remove_points(all_pts, n_pts=54):
    """
    For a grid of size > n_pts, find the best n_pts points to keep
    by brute forcing the removal of (size - n_pts) points if the number
    of removals is small (<= 3).
    """
    size = len(all_pts)
    k_remove = size - n_pts
    
    if k_remove == 0:
        return all_pts
    
    if k_remove > 3:
        rng = random.Random(12345)
        return rng.sample(all_pts, n_pts)
    
    best_score = -1
    best_subset = None
    indices = list(range(size))
    
    for remove_combo in combinations(indices, k_remove):
        remove_set = set(remove_combo)
        pts = [all_pts[i] for i in indices if i not in remove_set]
        
        lines, total = _solve_one(pts)
        if total > best_score:
            best_score = total
            best_subset = pts
            
        if _time_left() < 5:
            break
    
    if best_subset is not None:
        return best_subset
    return all_pts[:n_pts]

def _run_search(deadline):
    """
    Main search routine combining grid search and smart removal.
    """
    best_points = None
    best_lines = None
    best_count = 0
    
    # Seed baseline first
    base_points, base_lines = _szemeredi_baseline()
    base_count = _count_incidences_fast(base_points, base_lines)
    best_points = base_points.copy()
    best_lines = base_lines.copy()
    best_count = base_count
    
    # Focus on grids where smart removal is feasible (size <= 57)
    # Prioritize the known-good 6x9 and 9x6 grids first
    configs = [
        (6, 9), (9, 6), # Size 54, full - known good
        (7, 8), (8, 7), # Size 56, remove 2
        (5, 11), (11, 5), # Size 55, remove 1
        (4, 14), (14, 4), # Size 56, remove 2
        (3, 18), (18, 3), # Size 54, full
        (2, 27), (27, 2), # Size 54, full
        (1, 54), (54, 1), # Size 54, full
    ]
    
    for (w, h) in configs:
        if time.time() > deadline:
            break
        
        all_pts = [(x, y) for x in range(w) for y in range(h)]
        size = len(all_pts)
        
        if size == 54:
            points = all_pts
            lines_list, total = _solve_one(points)
            if len(lines_list) >= 27 and total > best_count:
                lines_arr = np.array(lines_list[:27], dtype=np.int64)
                points_arr = np.array(points, dtype=np.int64)
                if _is_valid(points_arr, lines_arr):
                    best_count = total
                    best_points = points_arr
                    best_lines = lines_arr
        elif size > 54:
            if size - 54 <= 3:
                points = _smart_remove_points(all_pts, 54)
                lines_list, total = _solve_one(points)
                if len(lines_list) >= 27 and total > best_count:
                    lines_arr = np.array(lines_list[:27], dtype=np.int64)
                    points_arr = np.array(points, dtype=np.int64)
                    if _is_valid(points_arr, lines_arr):
                        best_count = total
                        best_points = points_arr
                        best_lines = lines_arr
            else:
                # Random removal, more trials
                rng = random.Random(12345)
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    points = rng.sample(all_pts, 54)
                    lines_list, total = _solve_one(points)
                    if len(lines_list) >= 27 and total > best_count:
                        lines_arr = np.array(lines_list[:27], dtype=np.int64)
                        points_arr = np.array(points, dtype=np.int64)
                        if _is_valid(points_arr, lines_arr):
                            best_count = total
                            best_points = points_arr
                            best_lines = lines_arr
    
    # Additional local search from best found - use remaining time
    if time.time() < deadline - 30:
        p, l, c = _local_search(best_points, best_lines, deadline, max_iter=20000)
        if c > best_count:
            best_points = p
            best_lines = l
            best_count = c
    
    return best_points, best_lines, best_count

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Returns (points, lines) maximizing incidences.
    """
    random.seed(42)
    np.random.seed(42)
    
    deadline = _T0 + 590  # 590s to be safe (630s kill, 40s margin)
    
    # Run the main search
    best_points, best_lines, best_count = _run_search(deadline)
    
    # Final validation
    if not _is_valid(best_points, best_lines):
        # Fall back to baseline
        best_points, best_lines = _szemeredi_baseline()
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
