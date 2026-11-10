# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize line coefficients so first nonzero is positive and gcd divides all."""
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count exact incidences between points and lines using int64 arithmetic."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    total = 0
    for i in range(len(lns)):
        A, B, C = lns[i]
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        total += np.sum(vals == 0)
    return total

def _generate_candidate_lines(points):
    """Generate candidate lines from pairs of points, normalized."""
    n = len(points)
    candidate_set = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            norm = _normalize_line(A, B, C)
            if norm is not None:
                candidate_set[norm] = norm
    return list(candidate_set.values())

def _compute_line_weights(points, lines):
    """Compute incidence count for each line."""
    pts = np.asarray(points, dtype=np.int64)
    weights = np.zeros(len(lines), dtype=np.int64)
    for i in range(len(lines)):
        A, B, C = lines[i]
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        weights[i] = np.sum(vals == 0)
    return weights

def _find_best_lines_for_points(points, num_lines):
    """
    Find the best num_lines lines that maximize incidences with given points.
    This is done once to set up a good line set.
    """
    n = len(points)
    # Collect all lines determined by pairs of points
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            line = _normalize_line(points[j][1] - points[i][1], points[i][0] - points[j][0], points[j][0]*points[i][1] - points[i][0]*points[j][1])
            if line is not None:
                line_counts[line] = line_counts.get(line, 0) + 1
    
    # Sort by count descending
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    
    # Greedily pick top lines
    selected = []
    selected_set = set()
    for line, count in sorted_lines:
        if len(selected) >= num_lines:
            break
        if line not in selected_set:
            selected.append(line)
            selected_set.add(line)
            
    # If we don't have enough, add arbitrary distinct lines
    if len(selected) < num_lines:
        existing_norms = set(selected)
        counter = 100
        while len(selected) < num_lines:
            test_line = _normalize_line(1, 1, -counter)
            if test_line not in existing_norms:
                selected.append(test_line)
                existing_norms.add(test_line)
            counter += 1
            
    return np.array(selected, dtype=np.int64)

def _point_incidence_count(point, lines):
    """Count how many lines pass through a single point."""
    x, y = int(point[0]), int(point[1])
    count = 0
    for A, B, C in lines:
        if A * x + B * y + C == 0:
            count += 1
    return count

def _optimize_points_fixed_lines(points, lines, deadline):
    """
    Optimize point positions to maximize incidences with a fixed set of lines.
    Uses local search: move a point to a nearby integer coordinate if it increases incidence.
    """
    pts = points.copy()
    n = len(pts)
    m = len(lines)
    
    # Precompute line coefficients for speed
    lns = lines.astype(np.int64)
    
    # Initial incidence count per point
    point_counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        point_counts[i] = _point_incidence_count(pts[i], lns)
    
    total_inc = np.sum(point_counts)
    
    start_time = time.time()
    improved = True
    max_radius = 3
    
    while improved and (time.time() - start_time) < deadline - 10:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > deadline - 10:
                break
            
            old_x, old_y = int(pts[i, 0]), int(pts[i, 1])
            old_count = point_counts[i]
            
            best_new_x, best_new_y = old_x, old_y
            best_new_count = old_count
            
            for dx in range(-max_radius, max_radius + 1):
                for dy in range(-max_radius, max_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = old_x + dx, old_y + dy
                    
                    # Check if (new_x, new_y) is occupied by another point
                    is_occupied = False
                    for k in range(n):
                        if k != i and int(pts[k, 0]) == new_x and int(pts[k, 1]) == new_y:
                            is_occupied = True
                            break
                    if is_occupied:
                        continue
                    
                    # Calculate new incidence count for this point
                    new_count = 0
                    for j in range(m):
                        A, B, C = lns[j]
                        if A * new_x + B * new_y + C == 0:
                            new_count += 1
                    
                    if new_count > best_new_count:
                        best_new_count = new_count
                        best_new_x, best_new_y = new_x, new_y
            
            if best_new_count > old_count:
                pts[i, 0] = best_new_x
                pts[i, 1] = best_new_y
                point_counts[i] = best_new_count
                total_inc += best_new_count - old_count
                improved = True
                
    return pts, total_inc

def _line_swap_local_search(points, lines, deadline):
    """
    Line-swap local search using bipartite overlap structure.
    For each pair of lines, compute their shared points. If two lines share many points,
    one may be redundant. Try replacing the weaker line with a new candidate line that
    covers more uncovered points.
    """
    pts = np.asarray(points, dtype=np.int64)
    lns = lines.astype(np.int64)
    n_pts = len(pts)
    n_lines = len(lns)
    
    # Precompute which points lie on each line
    line_point_masks = []
    for i in range(n_lines):
        A, B, C = lns[i]
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        line_point_masks.append(vals == 0)
    
    # Compute pairwise overlap between lines
    overlap = np.zeros((n_lines, n_lines), dtype=np.int64)
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            overlap[i, j] = np.sum(line_point_masks[i] & line_point_masks[j])
            overlap[j, i] = overlap[i, j]
    
    # Compute current total incidence
    total_inc = sum(np.sum(mask) for mask in line_point_masks)
    
    # Generate candidate lines from point pairs
    candidate_lines = _generate_candidate_lines(pts)
    
    # For each candidate, precompute its point mask
    candidate_masks = []
    for cl in candidate_lines:
        A, B, C = cl
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        candidate_masks.append(vals == 0)
    
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < deadline - 10:
        improved = False
        
        # Find the pair of lines with highest overlap (most redundant)
        best_pair = (-1, -1)
        best_overlap = 0
        for i in range(n_lines):
            for j in range(i + 1, n_lines):
                if overlap[i, j] > best_overlap:
                    best_overlap = overlap[i, j]
                    best_pair = (i, j)
        
        if best_pair[0] == -1 or best_overlap < 3:
            # No highly redundant pairs, try to improve individual lines
            # For each line, find a candidate that covers more unique points
            for i in range(n_lines):
                if (time.time() - start_time) > deadline - 10:
                    break
                
                current_mask = line_point_masks[i]
                current_count = np.sum(current_mask)
                
                # Find points not covered by any other line (or by this line)
                # A point is "well-covered" if it's on many lines
                point_line_counts = np.zeros(n_pts, dtype=np.int64)
                for k in range(n_lines):
                    point_line_counts += line_point_masks[k].astype(np.int64)
                
                # Try to find a candidate that covers points with low coverage
                best_gain = 0
                best_candidate_idx = -1
                for ci, cm in enumerate(candidate_masks):
                    # Skip if this candidate is already in our line set
                    if any(np.array_equal(cm, line_point_masks[k]) for k in range(n_lines)):
                        continue
                    
                    # Gain = new points covered - points lost
                    # Points lost: those on current line but not on new line
                    # Points gained: those on new line but not on current line
                    # But we need to consider: does removing this line and adding new one improve total?
                    # Total change = (points on new line not on current) - (points on current not on new)
                    # But only if the new line doesn't duplicate another existing line
                    new_only = np.sum(cm & ~current_mask)
                    lost_only = np.sum(current_mask & ~cm)
                    gain = new_only - lost_only
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate_idx = ci
                
                if best_candidate_idx >= 0 and best_gain > 0:
                    # Replace line i with candidate
                    old_line = lns[i].copy()
                    new_line = np.array(candidate_lines[best_candidate_idx], dtype=np.int64)
                    lns[i] = new_line
                    line_point_masks[i] = candidate_masks[best_candidate_idx]
                    
                    # Update overlaps
                    for j in range(n_lines):
                        if j != i:
                            overlap[i, j] = np.sum(line_point_masks[i] & line_point_masks[j])
                            overlap[j, i] = overlap[i, j]
                    
                    total_inc += best_gain
                    improved = True
            
            continue
        
        # We have a redundant pair (i, j). Try replacing the weaker one.
        i, j = best_pair
        count_i = np.sum(line_point_masks[i])
        count_j = np.sum(line_point_masks[j])
        
        # Replace the line with fewer points
        if count_i <= count_j:
            replace_idx = i
            keep_idx = j
        else:
            replace_idx = j
            keep_idx = i
        
        current_mask = line_point_masks[replace_idx]
        current_count = np.sum(current_mask)
        
        # Find best replacement candidate
        best_gain = 0
        best_candidate_idx = -1
        for ci, cm in enumerate(candidate_masks):
            # Skip if this candidate is already in our line set
            if any(np.array_equal(cm, line_point_masks[k]) for k in range(n_lines)):
                continue
            
            new_only = np.sum(cm & ~current_mask)
            lost_only = np.sum(current_mask & ~cm)
            gain = new_only - lost_only
            
            if gain > best_gain:
                best_gain = gain
                best_candidate_idx = ci
        
        if best_candidate_idx >= 0 and best_gain > 0:
            new_line = np.array(candidate_lines[best_candidate_idx], dtype=np.int64)
            lns[replace_idx] = new_line
            line_point_masks[replace_idx] = candidate_masks[best_candidate_idx]
            
            # Update overlaps
            for k in range(n_lines):
                if k != replace_idx:
                    overlap[replace_idx, k] = np.sum(line_point_masks[replace_idx] & line_point_masks[k])
                    overlap[k, replace_idx] = overlap[replace_idx, k]
            
            total_inc += best_gain
            improved = True
    
    return lns, total_inc

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    Strategy:
    1. Start with 6x9 grid.
    2. Find best 27 lines for this grid.
    3. Optimize point positions to maximize incidences with these fixed lines.
    4. Line-swap local search to improve line selection using overlap structure.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # 1. Generate initial Points (6x9 grid)
    xs = np.arange(1, 7)  # 1 to 6
    ys = np.arange(1, 10) # 1 to 9
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Find best lines for this initial configuration
    lines = _find_best_lines_for_points(points, 27)
    
    # 3. Optimize points with fixed lines (use 60% of time)
    point_deadline = start_time + (deadline - start_time) * 0.6
    points, total_inc = _optimize_points_fixed_lines(points, lines, point_deadline)
    
    # 4. Line-swap local search (use remaining time)
    lines, total_inc = _line_swap_local_search(points, lines, deadline)
    
    # 5. Another round of point optimization with new lines
    remaining_deadline = deadline - 5
    points, total_inc = _optimize_points_fixed_lines(points, lines, remaining_deadline)
    
    # Final verification and cleanup
    unique_points = np.unique(points, axis=0)
    if len(unique_points) != len(points):
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = _find_best_lines_for_points(points, 27)
        
    valid_mask = ~((lines[:, 0] == 0) & (lines[:, 1] == 0))
    if not np.all(valid_mask):
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = _find_best_lines_for_points(points, 27)
        
    normalized_lines = [_normalize_line(int(l[0]), int(l[1]), int(l[2])) for l in lines]
    seen = set()
    unique_norms = []
    for nl in normalized_lines:
        if nl is not None and nl not in seen:
            seen.add(nl)
            unique_norms.append(nl)
            
    if len(unique_norms) != len(lines):
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = _find_best_lines_for_points(points, 27)
        
    return points, lines

# EVOLVE-BLOCK-END
