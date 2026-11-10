import numpy as np
import time
import math
import random
from itertools import combinations

def _normalize_line(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = math.gcd(abs(A), abs(B))
    g = math.gcd(g, abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidence(points, lines):
    """Count incidences between points (N,2) and lines (M,3)."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0]
    py = points[:, 1]
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
    return int(np.sum(vals == 0))

def _best_lines_for_points(points, k=27):
    """
    Find the k lines with the highest incidence count for the given points.
    Uses efficient vectorized counting for all candidate lines.
    """
    n = len(points)
    if n < 2:
        return np.zeros((k, 3), dtype=np.int64)
    
    xs = points[:, 0]
    ys = points[:, 1]
    
    seen = set()
    line_counts = {}
    
    # Generate candidate lines from all pairs
    candidate_lines = []
    for i in range(n):
        x1, y1 = int(xs[i]), int(ys[i])
        for j in range(i + 1, n):
            x2, y2 = int(xs[j]), int(ys[j])
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _normalize_line(A, B, C)
            if key not in seen:
                seen.add(key)
                candidate_lines.append(key)
    
    # Add generic lines (horizontal, vertical, diagonal)
    for c in range(-10, 11):
        key = _normalize_line(0, 1, -c)
        if key not in seen:
            seen.add(key)
            candidate_lines.append(key)
        key = _normalize_line(1, 0, -c)
        if key not in seen:
            seen.add(key)
            candidate_lines.append(key)
        key = _normalize_line(1, -1, -c)  # y = x - c
        if key not in seen:
            seen.add(key)
            candidate_lines.append(key)
        key = _normalize_line(1, 1, -c)   # y = -x - c
        if key not in seen:
            seen.add(key)
            candidate_lines.append(key)
    
    # If we still don't have enough, add more
    if len(candidate_lines) < k:
        c = -100
        while len(candidate_lines) < k:
            key = _normalize_line(0, 1, -c)
            if key not in seen:
                seen.add(key)
                candidate_lines.append(key)
            c -= 1
            if c < -2000:
                break
    
    # Vectorized counting for all candidate lines
    if candidate_lines:
        lines_arr = np.array(candidate_lines, dtype=np.int64)
        A = lines_arr[:, 0]
        B = lines_arr[:, 1]
        C = lines_arr[:, 2]
        vals = A[:, None] * xs[None, :] + B[:, None] * ys[None, :] + C[:, None]
        counts = np.sum(vals == 0, axis=1)
        
        for idx, cnt in enumerate(counts):
            line_counts[candidate_lines[idx]] = int(cnt)
    
    # Sort lines by count
    sorted_lines = sorted(line_counts.items(), key=lambda x: x[1], reverse=True)
    top_lines = [item[0] for item in sorted_lines[:k]]
    
    # Pad if necessary
    while len(top_lines) < k:
        top_lines.append((0, 1, -1000 - len(top_lines)))
        
    return np.array(top_lines[:k], dtype=np.int64)

def _evaluate_configuration(points):
    """Evaluate a point configuration and return (points, lines, score)."""
    lines = _best_lines_for_points(points, 27)
    score = _count_incidence(points, lines)
    return points, lines, score

def _generate_grid_points(k, m, dx=1, dy=1, ox=0, oy=0):
    """Generate a k x m grid of points."""
    xs = np.arange(k) * dx + ox
    ys = np.arange(m) * dy + oy
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _generate_punctured_grid(k, m, points_to_remove, dx=1, dy=1):
    """Generate a grid with specific points removed."""
    full_grid = _generate_grid_points(k, m, dx, dy)
    mask = np.ones(len(full_grid), dtype=bool)
    for (px, py) in points_to_remove:
        match = (full_grid[:, 0] == px) & (full_grid[:, 1] == py)
        if np.sum(match) > 0:
            mask[np.where(match)[0][0]] = False
    return full_grid[mask]

def _search_punctured_grid(k, m, n_remove, time_limit):
    """
    Search for the best n_remove points to remove from a k x m grid.
    Returns (points, lines, score).
    """
    full_grid = _generate_grid_points(k, m)
    n_pts = k * m
    
    if n_pts < n_remove:
        return None
    
    # Total number of ways to choose n_remove points
    total_ways = math.comb(n_pts, n_remove)
    
    if total_ways > 500000:  # Too many combinations, use sampling
        best_pts = None
        best_lines = None
        best_score = -1
        
        start = time.time()
        n_samples = min(50000, total_ways)
        
        for _ in range(n_samples):
            if time.time() - start > time_limit:
                break
                
            remove_idx = random.sample(range(n_pts), n_remove)
            mask = np.ones(n_pts, dtype=bool)
            mask[remove_idx] = False
            pts = full_grid[mask]
            
            if len(pts) != 54:
                continue
                
            p, l, s = _evaluate_configuration(pts)
            if s > best_score:
                best_score = s
                best_pts = p.copy()
                best_lines = l.copy()
        
        if best_pts is not None:
            return best_pts, best_lines, best_score
        return None
    
    # Exhaustive search if feasible
    best_pts = None
    best_lines = None
    best_score = -1
    
    start = time.time()
    
    for remove_idx in combinations(range(n_pts), n_remove):
        if time.time() - start > time_limit:
            break
            
        mask = np.ones(n_pts, dtype=bool)
        mask[list(remove_idx)] = False
        pts = full_grid[mask]
        
        if len(pts) != 54:
            continue
        
        p, l, s = _evaluate_configuration(pts)
        if s > best_score:
            best_score = s
            best_pts = p.copy()
            best_lines = l.copy()
    
    if best_pts is not None:
        return best_pts, best_lines, best_score
    return None

def _local_search(points, lines, score, time_limit):
    """
    Local search to improve the configuration.
    Tries to move individual points to improve incidence.
    Uses faster evaluation by checking which lines the point is on.
    """
    points = points.copy()
    lines = lines.copy()
    current_score = score
    
    start = time.time()
    no_improve = 0
    max_no_improve = 30
    
    # Build a set of existing points for fast lookup
    point_set = set(map(tuple, points.tolist()))
    
    while time.time() - start < time_limit:
        # Pick a random point to move
        i = random.randint(0, 53)
        old_x, old_y = int(points[i, 0]), int(points[i, 1])
        improved = False
        
        # Try multiple moves
        for attempt in range(20):
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            
            new_x = old_x + dx
            new_y = old_y + dy
            
            # Check uniqueness
            if (new_x, new_y) in point_set:
                continue
            
            # Remove old point from set, add new
            point_set.discard((old_x, old_y))
            point_set.add((new_x, new_y))
            
            new_points = points.copy()
            new_points[i, 0] = new_x
            new_points[i, 1] = new_y
            
            # Recompute best lines
            new_lines = _best_lines_for_points(new_points, 27)
            new_score = _count_incidence(new_points, new_lines)
            
            if new_score > current_score:
                points = new_points
                lines = new_lines
                current_score = new_score
                improved = True
                no_improve = 0
                break
            elif new_score == current_score and random.random() < 0.2:
                # Accept neutral moves to escape local minima
                points = new_points
                lines = new_lines
                no_improve += 1
            else:
                # Revert
                point_set.discard((new_x, new_y))
                point_set.add((old_x, old_y))
        
        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
    
    return points, lines, current_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # 50s safety margin before 630s limit
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Phase 1: Evaluate various grid-based configurations
    
    # 6x9 grid (54 points)
    pts = _generate_grid_points(6, 9)
    if len(pts) == 54:
        p, l, s = _evaluate_configuration(pts)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            best_lines = l.copy()
    
    # 9x6 grid
    pts = _generate_grid_points(9, 6)
    if len(pts) == 54:
        p, l, s = _evaluate_configuration(pts)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            best_lines = l.copy()
    
    # 3x18 grid
    pts = _generate_grid_points(3, 18)
    if len(pts) == 54:
        p, l, s = _evaluate_configuration(pts)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            best_lines = l.copy()
    
    # 18x3 grid
    pts = _generate_grid_points(18, 3)
    if len(pts) == 54:
        p, l, s = _evaluate_configuration(pts)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            best_lines = l.copy()
    
    # 2x27 grid
    pts = _generate_grid_points(2, 27)
    if len(pts) == 54:
        p, l, s = _evaluate_configuration(pts)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            best_lines = l.copy()
    
    # 27x2 grid
    pts = _generate_grid_points(27, 2)
    if len(pts) == 54:
        p, l, s = _evaluate_configuration(pts)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            best_lines = l.copy()
    
    # Phase 2: Search punctured grids (remove points from larger grids)
    
    # 7x8 = 56, remove 2
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        result = _search_punctured_grid(7, 8, 2, min(25, remaining_time - 5))
        if result is not None:
            p, l, s = result
            if s > best_score:
                best_score = s
                best_points = p.copy()
                best_lines = l.copy()
    
    # 5x11 = 55, remove 1
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        result = _search_punctured_grid(5, 11, 1, min(15, remaining_time - 5))
        if result is not None:
            p, l, s = result
            if s > best_score:
                best_score = s
                best_points = p.copy()
                best_lines = l.copy()
    
    # 6x10 = 60, remove 6
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        result = _search_punctured_grid(6, 10, 6, min(10, remaining_time - 5))
        if result is not None:
            p, l, s = result
            if s > best_score:
                best_score = s
                best_points = p.copy()
                best_lines = l.copy()
    
    # 9x7 = 63, remove 9
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        result = _search_punctured_grid(9, 7, 9, min(5, remaining_time - 5))
        if result is not None:
            p, l, s = result
            if s > best_score:
                best_score = s
                best_points = p.copy()
                best_lines = l.copy()
    
    # Phase 3: Local search from the best configuration found
    if best_points is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 30:
            p, l, s = _local_search(best_points, best_lines, best_score, min(remaining_time - 10, 120))
            if s > best_score:
                best_score = s
                best_points = p.copy()
                best_lines = l.copy()
        
        # Phase 3b: Another round of local search with different parameters
        remaining_time = deadline - time.time()
        if remaining_time > 20:
            p, l, s = _local_search(best_points, best_lines, best_score, min(remaining_time - 5, 60))
            if s > best_score:
                best_score = s
                best_points = p.copy()
                best_lines = l.copy()
    
    # Fallback if no valid configuration found
    if best_points is None:
        best_points = _generate_grid_points(6, 9)
        best_lines = _best_lines_for_points(best_points, 27)
        best_score = _count_incidence(best_points, best_lines)
    
    # Final validation and formatting
    final_points = np.array(best_points, dtype=np.int64)
    final_lines = np.array(best_lines, dtype=np.int64)
    
    # Ensure shapes
    if final_points.shape != (54, 2):
        final_points = _generate_grid_points(6, 9)
        final_lines = _best_lines_for_points(final_points, 27)
        
    if final_lines.shape != (27, 3):
        final_lines = final_lines[:27]
        if len(final_lines) < 27:
            pad = np.zeros((27 - len(final_lines), 3), dtype=np.int64)
            pad[:, 1] = 1
            pad[:, 2] = -1000
            final_lines = np.vstack([final_lines, pad])
    
    # Double check uniqueness of points
    unique_pts = np.unique(final_points, axis=0)
    if len(unique_pts) < 54:
        added = 0
        offset = 100
        while len(unique_pts) < 54:
            pt = np.array([offset + added, offset * 2 + added], dtype=np.int64)
            if pt not in unique_pts:
                unique_pts = np.vstack([unique_pts, pt[None, :]])
            added += 1
        final_points = unique_pts[:54]
        final_lines = _best_lines_for_points(final_points, 27)
    
    return final_points.astype(np.int64), final_lines.astype(np.int64)
