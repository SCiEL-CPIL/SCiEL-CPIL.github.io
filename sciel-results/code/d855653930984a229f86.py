import numpy as np
import time
from math import gcd
from itertools import combinations
import random

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

def _compute_incidence_matrix(points, lines):
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    # vals[i, j] = A_i * x_j + B_i * y_j + C_i
    vals = lns[:, 0, None] * pts[None, :, 0] + lns[:, 1, None] * pts[None, :, 1] + lns[:, 2, None]
    return (vals == 0).astype(np.int64)

def _incidence_count(points, lines):
    return int(np.sum(_compute_incidence_matrix(points, lines)))

def _initial_grid_diagonal():
    w, h = 6, 9
    points = []
    for x in range(w):
        for y in range(h):
            points.append((x, y))
    
    pts = np.array(points, dtype=np.int64)
    
    lines = []
    for x in range(w):
        lines.append((1, 0, -x))
    for y in range(h):
        lines.append((0, 1, -y))
    
    diag_cands = []
    for c in range(-h+1, w):
        diag_cands.append((1, -1, -c))
    for c in range(0, w+h-1):
        diag_cands.append((1, 1, -c))
    
    scored = []
    for l in diag_cands:
        A, B, C = l
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        cnt = int(np.sum(vals == 0))
        scored.append((cnt, l))
    
    scored.sort(key=lambda x: -x[0])
    
    current_lines = set(lines)
    for cnt, l in scored:
        canon = _normalize_line(*l)
        if canon not in current_lines:
            lines.append(canon)
            current_lines.add(canon)
            if len(lines) == 27:
                break
    
    return np.array(points, dtype=np.int64), np.array(lines, dtype=np.int64)

def _greedy_select_lines(pts, n_lines=27):
    """Select top n_lines lines by incidence count, no duplicates."""
    n_pts = len(pts)
    cand_lines = set()
    
    # Systematic lines: small slopes and intercepts
    for A in range(-4, 5):
        for B in range(-4, 5):
            if A == 0 and B == 0: continue
            for C in range(-20, 21):
                c = _normalize_line(A, B, C)
                if c: cand_lines.add(c)
    
    # Secants of all point pairs
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            p1 = pts[i]
            p2 = pts[j]
            A = int(p2[1]) - int(p1[1])
            B = int(p1[0]) - int(p2[0])
            C = -(A * int(p1[0]) + B * int(p1[1]))
            c = _normalize_line(A, B, C)
            if c: cand_lines.add(c)
    
    cand_lines_list = list(cand_lines)
    if not cand_lines_list:
        return np.zeros((n_lines, 3), dtype=np.int64)
        
    # Vectorized scoring
    # Create arrays for A, B, C
    A_arr = np.array([l[0] for l in cand_lines_list], dtype=np.int64)
    B_arr = np.array([l[1] for l in cand_lines_list], dtype=np.int64)
    C_arr = np.array([l[2] for l in cand_lines_list], dtype=np.int64)
    
    # Compute vals: (N_cands, N_pts)
    # A_arr[:, None] * pts[None, :, 0] + ...
    vals = A_arr[:, None] * pts[None, :, 0] + B_arr[:, None] * pts[None, :, 1] + C_arr[:, None]
    counts = np.sum(vals == 0, axis=1)
    
    # Sort indices by count descending
    sorted_indices = np.argsort(-counts)
    
    selected = []
    seen = set()
    for idx in sorted_indices:
        line = cand_lines_list[idx]
        if line not in seen:
            selected.append(line)
            seen.add(line)
            if len(selected) == n_lines:
                break
    
    # If we didn't find enough, pad (should not happen with this many candidates)
    if len(selected) < n_lines:
        for line in cand_lines_list:
            if line not in seen:
                selected.append(line)
                seen.add(line)
                if len(selected) == n_lines:
                    break
    
    return np.array(selected[:n_lines], dtype=np.int64)

def _local_search(points, lines, deadline):
    import random
    rng = random.Random(123)
    
    pts = points.copy()
    lns = lines.copy()
    n_pts, n_lines = 54, 27
    
    inc_matrix = _compute_incidence_matrix(pts, lns)
    current_score = int(np.sum(inc_matrix))
    best_score = current_score
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    # Pre-generate static candidate lines (secants of initial points + systematic)
    cand_lines = set()
    for A in range(-4, 5):
        for B in range(-4, 5):
            if A == 0 and B == 0: continue
            for C in range(-20, 21):
                c = _normalize_line(A, B, C)
                if c: cand_lines.add(c)
    
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            p1 = pts[i]
            p2 = pts[j]
            A = int(p2[1]) - int(p1[1])
            B = int(p1[0]) - int(p2[0])
            C = -(A * int(p1[0]) + B * int(p1[1]))
            c = _normalize_line(A, B, C)
            if c: cand_lines.add(c)
    
    cand_lines_list = list(cand_lines)
    cand_lines_set = set(cand_lines_list)
    
    T = 8.0
    T_min = 0.01
    cooling = 0.9998
    
    def compute_greedy_score(pts_arr):
        lns_sel = _greedy_select_lines(pts_arr, n_lines)
        return _incidence_count(pts_arr, lns_sel)
    
    last_greedy_check = time.time()
    greedy_interval = 2.0
    
    # Pre-compute current line canonicals for fast duplicate check
    current_line_canons = [_normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2])) for j in range(n_lines)]
    
    while time.time() < deadline - 10:
        if rng.random() < 0.7:
            # Move point
            i = rng.randint(0, n_pts - 1)
            old_pos = tuple(pts[i])
            
            # Sample a random neighbor
            dx = rng.randint(-3, 3)
            dy = rng.randint(-3, 3)
            new_pos = (int(pts[i, 0]) + dx, int(pts[i, 1]) + dy)
            
            if new_pos == old_pos:
                continue
            
            # Check duplicate
            dup = False
            for k in range(n_pts):
                if k != i and tuple(pts[k]) == new_pos:
                    dup = True
                    break
            if dup:
                continue
            
            # Quick delta using current lines
            A = lns[:, 0]
            B = lns[:, 1]
            C = lns[:, 2]
            new_vals = A * new_pos[0] + B * new_pos[1] + C
            new_inc = (new_vals == 0).astype(np.int64)
            old_inc = inc_matrix[:, i]
            delta = int(np.sum(new_inc)) - int(np.sum(old_inc))
            
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                pts[i] = new_pos
                inc_matrix[:, i] = new_inc
                current_score += delta
                
                # Periodic greedy re-evaluation
                if time.time() - last_greedy_check > greedy_interval:
                    greedy_score = compute_greedy_score(pts)
                    if greedy_score > best_score:
                        best_score = greedy_score
                        best_pts = pts.copy()
                        best_lns = _greedy_select_lines(pts, n_lines)
                    last_greedy_check = time.time()
        else:
            # Move line
            j = rng.randint(0, n_lines - 1)
            old_canon = current_line_canons[j]
            new_line = cand_lines_list[rng.randint(0, len(cand_lines_list)-1)]
            
            if new_line == old_canon:
                continue
            
            # Check duplicate using set
            if new_line in current_line_canons:
                continue
            
            A, B, C = new_line
            new_vals = A * pts[:, 0] + B * pts[:, 1] + C
            new_inc = (new_vals == 0).astype(np.int64)
            old_inc = inc_matrix[j, :]
            delta = int(np.sum(new_inc)) - int(np.sum(old_inc))
            
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                lns[j] = new_line
                inc_matrix[j, :] = new_inc
                current_score += delta
                current_line_canons[j] = new_line
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final greedy line selection
    final_lns = _greedy_select_lines(best_pts, n_lines)
    final_score = _incidence_count(best_pts, final_lns)
    
    if final_score >= best_score:
        return best_pts, final_lns
    else:
        return best_pts, best_lns

def _refine_points(pts, lns, deadline):
    n_pts, n_lines = 54, 27
    inc_matrix = _compute_incidence_matrix(pts, lns)
    current_score = int(np.sum(inc_matrix))
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(n_pts):
            best_pos = tuple(pts[i])
            best_delta = 0
            best_new_inc = None
            old_sum = int(np.sum(inc_matrix[:, i]))
            
            # Try nearby positions
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    new_pos = (int(pts[i, 0]) + dx, int(pts[i, 1]) + dy)
                    if new_pos == best_pos:
                        continue
                    dup = False
                    for k in range(n_pts):
                        if k != i and tuple(pts[k]) == new_pos:
                            dup = True
                            break
                    if dup:
                        continue
                    
                    A = lns[:, 0]
                    B = lns[:, 1]
                    C = lns[:, 2]
                    new_vals = A * new_pos[0] + B * new_pos[1] + C
                    new_inc = (new_vals == 0).astype(np.int64)
                    new_sum = int(np.sum(new_inc))
                    delta = new_sum - old_sum
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_pos = new_pos
                        best_new_inc = new_inc
            
            if best_delta > 0:
                pts[i] = best_pos
                inc_matrix[:, i] = best_new_inc
                current_score += best_delta
                improved = True
    
    return pts

def _refine_lines(pts, lns, deadline):
    n_pts, n_lines = 54, 27
    inc_matrix = _compute_incidence_matrix(pts, lns)
    current_score = int(np.sum(inc_matrix))
    improved = True
    
    # Pre-generate a larger set of candidate lines
    cand_lines = set()
    for A in range(-10, 11):
        for B in range(-10, 11):
            if A == 0 and B == 0: continue
            for C in range(-30, 31):
                c = _normalize_line(A, B, C)
                if c: cand_lines.add(c)
    # Secants
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            p1 = pts[i]
            p2 = pts[j]
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = -(A * p1[0] + B * p1[1])
            c = _normalize_line(A, B, C)
            if c: cand_lines.add(c)
            
    cand_lines_list = list(cand_lines)

    while improved and time.time() < deadline - 10:
        improved = False
        for j in range(n_lines):
            best_line = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            best_delta = 0
            best_new_inc = None
            old_sum = int(np.sum(inc_matrix[j, :]))
            
            for new_line in cand_lines_list:
                if new_line == best_line:
                    continue
                dup = False
                for k in range(n_lines):
                    if k != j and _normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2])) == new_line:
                        dup = True
                        break
                if dup:
                    continue
                
                A, B, C = new_line
                new_vals = A * pts[:, 0] + B * pts[:, 1] + C
                new_inc = (new_vals == 0).astype(np.int64)
                new_sum = int(np.sum(new_inc))
                delta = new_sum - old_sum
                
                if delta > best_delta:
                    best_delta = delta
                    best_line = new_line
                    best_new_inc = new_inc
            
            if best_delta > 0:
                lns[j] = best_line
                inc_matrix[j, :] = best_new_inc
                current_score += best_delta
                improved = True
    
    return lns

def _multi_start_local_search(deadline):
    """Run local search from multiple starting configurations."""
    best_pts = None
    best_lns = None
    best_score = -1
    
    # Primary start: 6x9 grid
    pts, lns = _initial_grid_diagonal()
    
    # Quick refine
    if time.time() < deadline - 30:
        pts = _refine_points(pts, lns, deadline - 30)
    lns = _greedy_select_lines(pts, 27)
    score = _incidence_count(pts, lns)
    
    if score > best_score:
        best_score = score
        best_pts = pts.copy()
        best_lns = lns.copy()
    
    # Local search from this config
    if time.time() < deadline - 10:
        sa_pts, sa_lns = _local_search(pts, lns, deadline - 10)
        sa_score = _incidence_count(sa_pts, sa_lns)
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()
            best_lns = sa_lns.copy()
    
    return best_pts, best_lns, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580 # Safety margin
    
    # Multi-start local search with greedy line selection
    pts, lns, score = _multi_start_local_search(deadline)
    
    # Final refinement if time permits
    if time.time() < deadline - 15:
        pts = _refine_points(pts, lns, deadline - 15)
        lns = _greedy_select_lines(pts, 27)
    
    if time.time() < deadline - 5:
        lns = _refine_lines(pts, lns, deadline - 5)
    
    # Validation
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
    if not _is_valid(pts, lns):
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for x in range(6):
            lines.append((1, 0, -x))
        for y in range(9):
            lines.append((0, 1, -y))
        for c in range(0, 12):
            lines.append((1, 1, -c))
        lns = np.array(lines, dtype=np.int64)
    
    return pts, lns
