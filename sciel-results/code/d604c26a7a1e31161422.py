import numpy as np
from math import gcd
import time
import itertools

def _gcd3(a, b, c):
    return gcd(gcd(abs(a), abs(b)), abs(c))

def _normalize_line(A, B, C):
    g = _gcd3(A, B, C)
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    # Make first non-zero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidents(points, lines):
    # points: (N, 2) array
    # lines: (M, 3) array
    # Return number of incidences
    count = 0
    for p in points:
        x, y = p
        for l in lines:
            A, B, C = l
            if A * x + B * y + C == 0:
                count += 1
    return count

def _validate(points, lines):
    # Check shapes
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    # Check integer types (assuming they are int64)
    # Check for NaN/inf
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    # Check if entries are close to integers
    if not np.all(np.abs(points - np.round(points)) < 1e-6):
        return False
    if not np.all(np.abs(lines - np.round(lines)) < 1e-6):
        return False
    
    points = np.round(points).astype(np.int64)
    lines = np.round(lines).astype(np.int64)
    
    # Check repeated points
    unique_points = np.unique(points, axis=0)
    if len(unique_points) != 54:
        return False
        
    # Check line validity
    seen_lines = set()
    for l in lines:
        A, B, C = l
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm in seen_lines:
            return False
        seen_lines.add(norm)
        
    return True

def _generate_candidate_lines(points):
    """
    Generate all lines passing through at least 2 points in the grid.
    Returns a list of (line_tuple, count)
    """
    line_counts = {}
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            norm = _normalize_line(A, B, C)
            if norm not in line_counts:
                line_counts[norm] = 0
            # We need to count unique points on the line.
            # Since we generate from pairs, we just need to know if a point is on the line.
            # A better way: for each line, count points.
    # This O(N^2) generation of lines is fine.
    # But counting incidences for each line is O(N^2 * N) if done naively.
    # Let's just store the lines, then count incidences in a vectorized way or simple loop.
    
    unique_lines = list(line_counts.keys())
    # Filter out lines with < 2 points? 
    # Actually, any line defined by 2 points has at least 2.
    # We can just return unique_lines.
    return unique_lines

def _local_search(points, initial_lines, max_time=500):
    """
    Local search to improve the set of lines.
    """
    current_lines = list(initial_lines)
    current_score = _count_incidents(points, current_lines)
    
    # Generate all candidate lines
    all_candidates = _generate_candidate_lines(points)
    # Precompute incidence count for each candidate
    cand_scores = {}
    for l in all_candidates:
        # Count points on line l
        cnt = 0
        A, B, C = l
        for p in points:
            x, y = p
            if A * x + B * y + C == 0:
                cnt += 1
        cand_scores[l] = cnt
        
    # Sort candidates by score descending
    sorted_cands = sorted(all_candidates, key=lambda x: -cand_scores[x])
    
    # Start with top 27? Or the initial set?
    # The initial set is likely good.
    
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        # Try swapping each current line with a candidate not in current set
        current_set = set(current_lines)
        # Get candidates not in current set, sorted by score
        # To speed up, we can just iterate through sorted_cands
        for i in range(27):
            line_to_remove = current_lines[i]
            # Find best line to add
            best_add = None
            best_gain = 0
            # We can prune: if cand_score <= 1, gain is likely low
            for l_add in sorted_cands:
                if l_add in current_set:
                    continue
                # Calculate gain
                # Gain = (points on l_add) - (points on l_remove that are NOT on l_add)
                # This is complex. Just recompute score?
                # 27 * 54 is small.
                new_lines = current_lines.copy()
                new_lines[i] = l_add
                new_score = _count_incidents(points, new_lines)
                gain = new_score - current_score
                if gain > best_gain:
                    best_gain = gain
                    best_add = l_add
                if best_gain > 0:
                    # If we found a good gain, we might break early?
                    # No, we want the BEST swap.
                    pass
                    
            if best_add is not None and best_gain > 0:
                current_lines[i] = best_add
                current_score += best_gain
                improved = True
                current_set.add(best_add)
                current_set.remove(line_to_remove)
                
    return current_lines, current_score

def szemeredi_trotter_incidences():
    # Grid 6x9
    a, b = 6, 9
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Initial lines: 9 H, 6 V, 12 D
    lines = []
    for j in range(1, b + 1):
        lines.append((0, 1, -j))
    for i in range(1, a + 1):
        lines.append((1, 0, -i))
        
    # Top 12 diagonals
    diag_counts = {}
    for i in range(1, a + 1):
        for j in range(1, b + 1):
            c = j - i
            diag_counts[c] = diag_counts.get(c, 0) + 1
    sorted_diags = sorted(diag_counts.items(), key=lambda x: (-x[1], x[0]))
    selected_c = [c for c, cnt in sorted_diags[:12]]
    for c in selected_c:
        lines.append((1, -1, c))
        
    initial_lines = np.array(lines, dtype=np.int64)
    
    # Run local search
    # Time limit 500s to be safe
    best_lines, best_score = _local_search(points, initial_lines, max_time=500)
    
    best_lines = np.array(best_lines, dtype=np.int64)
    
    # Final validation
    if not _validate(points, best_lines):
        # Fallback to initial
        return points, initial_lines
        
    return points, best_lines
