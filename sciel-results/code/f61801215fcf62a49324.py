import numpy as np
import time
from math import gcd
from collections import defaultdict
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

def _incidence_count(points, lines):
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    vals = lns[:, 0, None] * pts[None, :, 0] + lns[:, 1, None] * pts[None, :, 1] + lns[:, 2, None]
    return int(np.sum(vals == 0))

def _generate_grid_initialization():
    w, h = 6, 9
    points = []
    for x in range(w):
        for y in range(h):
            points.append((x, y))
            
    lines = []
    # Verticals
    for x in range(w):
        lines.append((1, 0, -x))
    # Horizontals
    for y in range(h):
        lines.append((0, 1, -y))
        
    # Diagonals with slope 1 (x - y = c)
    for c in range(-h+1, w):
        y_min = max(0, -c)
        y_max = min(h, w-c)
        if y_max > y_min:
            lines.append((1, -1, -c))
            
    # Diagonals with slope -1 (x + y = c)
    for c in range(0, w+h-1):
        y_min = max(0, c-w+1)
        y_max = min(h, c+1)
        if y_max > y_min:
            lines.append((1, 1, -c))
            
    pts_arr = np.array(points)
    line_lengths = []
    for l in lines:
        A, B, C = l
        vals = A * pts_arr[:, 0] + B * pts_arr[:, 1] + C
        cnt = int(np.sum(vals == 0))
        line_lengths.append((cnt, l))
        
    line_lengths.sort(key=lambda x: -x[0])
    
    selected_lines = []
    seen = set()
    for cnt, l in line_lengths:
        canon = _normalize_line(l[0], l[1], l[2])
        if canon not in seen:
            seen.add(canon)
            selected_lines.append(canon)
            if len(selected_lines) == 27:
                break
                
    return np.array(points, dtype=np.int64), np.array(selected_lines, dtype=np.int64)

def _local_search(pts, lns, deadline):
    rng = random.Random(42)
    pts = pts.copy()
    lns = lns.copy()
    
    current_score = _incidence_count(pts, lns)
    best_score = current_score
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    n_pts = 54
    n_lines = 27
    
    # Precompute candidate lines pool for random replacement
    # Larger coefficient range
    cand_pool = set()
    for A in range(-5, 6):
        for B in range(-5, 6):
            if A == 0 and B == 0:
                continue
            for C in range(-30, 31):
                c = _normalize_line(A, B, C)
                if c:
                    cand_pool.add(c)
    cand_pool = list(cand_pool)
    
    T_start = 10.0
    T_end = 0.05
    max_iters = 500000
    iters_done = 0
    
    while time.time() < deadline - 15:
        iters_done += 1
        progress = min(iters_done / max_iters, 1.0)
        T = T_start * (T_end / T_start) ** progress
        
        if rng.random() < 0.6:
            # Move a point
            i = rng.randint(0, n_pts - 1)
            old_pos = tuple(pts[i])
            ox, oy = old_pos
            
            # Try small moves
            moved = False
            for _ in range(15):
                dx = rng.randint(-3, 3)
                dy = rng.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                
                # Check duplicates
                is_dup = False
                for k in range(n_pts):
                    if k != i and pts[k, 0] == nx and pts[k, 1] == ny:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                # Vectorized delta calculation
                # was: lns[:,0]*ox + lns[:,1]*oy + lns[:,2] == 0
                # now: lns[:,0]*nx + lns[:,1]*ny + lns[:,2] == 0
                old_vals = lns[:, 0] * ox + lns[:, 1] * oy + lns[:, 2]
                new_vals = lns[:, 0] * nx + lns[:, 1] * ny + lns[:, 2]
                delta = int(np.sum(new_vals == 0)) - int(np.sum(old_vals == 0))
                
                if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                    pts[i] = [nx, ny]
                    current_score += delta
                    moved = True
                    break
            
            if not moved:
                continue
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                best_lns = lns.copy()
                
        else:
            # Replace a line
            j = rng.randint(0, n_lines - 1)
            old_line = tuple(lns[j])
            
            # Generate candidate: either from pool or defined by 2 random points
            if rng.random() < 0.5:
                new_line = cand_pool[rng.randint(0, len(cand_pool)-1)]
            else:
                # Pick 2 distinct points
                i1 = rng.randint(0, n_pts - 1)
                i2 = rng.randint(0, n_pts - 1)
                if i1 == i2:
                    continue
                x1, y1 = pts[i1]
                x2, y2 = pts[i2]
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                new_line = _normalize_line(A, B, C)
                if new_line is None:
                    continue
            
            if new_line == old_line:
                continue
                
            # Check duplicates
            is_dup = False
            for k in range(n_lines):
                if k != j:
                    if tuple(lns[k]) == new_line:
                        is_dup = True
                        break
            if is_dup:
                continue
            
            # Vectorized delta calculation
            old_A, old_B, old_C = old_line
            A, B, C = new_line
            old_vals = old_A * pts[:, 0] + old_B * pts[:, 1] + old_C
            new_vals = A * pts[:, 0] + B * pts[:, 1] + C
            delta = int(np.sum(new_vals == 0)) - int(np.sum(old_vals == 0))
                
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                lns[j] = new_line
                current_score += delta
                
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                best_lns = lns.copy()
                
    return best_pts, best_lns, best_score

def _polish(pts, lns, deadline):
    """Final polish pass with deterministic hill climbing"""
    pts = pts.copy()
    lns = lns.copy()
    n_pts = 54
    n_lines = 27
    
    current_score = _incidence_count(pts, lns)
    best_score = current_score
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try moving each point to best position
        for i in range(n_pts):
            ox, oy = pts[i]
            best_delta = 0
            best_pos = None
            
            # Try all positions in a reasonable range
            x_range = range(int(min(pts[:, 0])) - 5, int(max(pts[:, 0])) + 6)
            y_range = range(int(min(pts[:, 1])) - 5, int(max(pts[:, 1])) + 6)
            
            for nx in x_range:
                for ny in y_range:
                    if nx == ox and ny == oy:
                        continue
                    # Check duplicates
                    is_dup = False
                    for k in range(n_pts):
                        if k != i and pts[k, 0] == nx and pts[k, 1] == ny:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    old_vals = lns[:, 0] * ox + lns[:, 1] * oy + lns[:, 2]
                    new_vals = lns[:, 0] * nx + lns[:, 1] * ny + lns[:, 2]
                    delta = int(np.sum(new_vals == 0)) - int(np.sum(old_vals == 0))
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_pos = (nx, ny)
            
            if best_pos is not None:
                pts[i] = best_pos
                current_score += best_delta
                improved = True
        
        # Try replacing each line with best from candidate pool
        for j in range(n_lines):
            old_line = tuple(lns[j])
            old_A, old_B, old_C = old_line
            best_delta = 0
            best_line = None
            
            # Generate candidates from pairs of points
            seen_cands = set()
            for i1 in range(n_pts):
                for i2 in range(i1 + 1, n_pts):
                    x1, y1 = pts[i1]
                    x2, y2 = pts[i2]
                    A = y2 - y1
                    B = x1 - x2
                    C = x2 * y1 - x1 * y2
                    cand = _normalize_line(A, B, C)
                    if cand is None or cand == old_line or cand in seen_cands:
                        continue
                    seen_cands.add(cand)
                    
                    # Check duplicates with other lines
                    is_dup = False
                    for k in range(n_lines):
                        if k != j and tuple(lns[k]) == cand:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    new_vals = cand[0] * pts[:, 0] + cand[1] * pts[:, 1] + cand[2]
                    old_vals = old_A * pts[:, 0] + old_B * pts[:, 1] + old_C
                    delta = int(np.sum(new_vals == 0)) - int(np.sum(old_vals == 0))
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_line = cand
            
            if best_line is not None:
                lns[j] = best_line
                current_score += best_delta
                improved = True
        
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
            best_lns = lns.copy()
    
    return best_pts, best_lns, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # 1. Initialize
    pts, lns = _generate_grid_initialization()
    
    # 2. Local Search
    pts, lns, score = _local_search(pts, lns, deadline)
    
    # 3. Polish pass
    if time.time() < deadline - 20:
        pts, lns, score = _polish(pts, lns, deadline)
    
    # 4. Final Validation and Cleanup
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
    if not _is_valid(pts, lns):
        # Fallback
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for x in range(6):
            lines.append((1, 0, -x))
        for y in range(9):
            lines.append((0, 1, -y))
        for k in range(12):
            lines.append((1, 1, -k))
        lns = np.array(lines, dtype=np.int64)
        
    return pts, lns
