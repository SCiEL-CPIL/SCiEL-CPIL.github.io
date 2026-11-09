# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd

def _canonical_line(A, B, C):
    """Return canonical form of line (A, B, C) with gcd normalized and first nonzero positive."""
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 1:
        A //= g
        B //= g
        C //= g
    # Make first nonzero entry positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidence(points, A, B, C):
    """Count points on line A*x + B*y + C = 0."""
    return int(np.sum(points[:, 0] * A + points[:, 1] * B + C == 0))

def _get_all_diagonal_lines(points):
    """
    Get all possible diagonal lines (slope 1 and slope -1) through the grid,
    sorted by number of incidences descending.
    Returns list of (incidence_count, A, B, C) tuples.
    """
    lines_dict = {}
    
    # Slope 1 lines: y - x = c, i.e., -x + y - c = 0
    for pt in points:
        c = pt[1] - pt[0]
        key = (-1, 1, -c)
        canon = _canonical_line(*key)
        if canon is not None and canon not in lines_dict:
            lines_dict[canon] = _count_incidence(points, *canon)
    
    # Slope -1 lines: x + y = c, i.e., x + y - c = 0
    for pt in points:
        c = pt[0] + pt[1]
        key = (1, 1, -c)
        canon = _canonical_line(*key)
        if canon is not None and canon not in lines_dict:
            lines_dict[canon] = _count_incidence(points, *canon)
    
    # Sort by incidence count descending, then by line tuple for determinism
    sorted_lines = sorted(lines_dict.items(), key=lambda x: (-x[1], x[0]))
    return [(cnt, A, B, C) for (A, B, C), cnt in sorted_lines]

def _total_incidence(points, lines):
    """Compute total number of point-line incidences."""
    total = 0
    for A, B, C in lines:
        total += _count_incidence(points, A, B, C)
    return total

def _select_best_lines(points, num_lines=27):
    """
    Select the best num_lines from all possible lines through pairs of points.
    Uses a greedy approach: first add all high-incidence lines, then fill remaining.
    """
    # Generate all candidate lines from pairs of points
    n = len(points)
    lines_set = set()
    lines_incidents = {}
    
    # Add horizontal, vertical, and diagonal lines first (they're likely to be good)
    # Horizontal: y = c
    for c in set(points[:, 1]):
        key = (0, 1, -int(c))
        canon = _canonical_line(*key)
        if canon is not None:
            lines_set.add(canon)
            lines_incidents[canon] = _count_incidence(points, *canon)
    
    # Vertical: x = c
    for c in set(points[:, 0]):
        key = (1, 0, -int(c))
        canon = _canonical_line(*key)
        if canon is not None:
            lines_set.add(canon)
            lines_incidents[canon] = _count_incidence(points, *canon)
    
    # Diagonals slope 1: y - x = c
    for pt in points:
        c = pt[1] - pt[0]
        key = (-1, 1, -int(c))
        canon = _canonical_line(*key)
        if canon is not None and canon not in lines_set:
            lines_set.add(canon)
            lines_incidents[canon] = _count_incidence(points, *canon)
    
    # Diagonals slope -1: x + y = c
    for pt in points:
        c = pt[0] + pt[1]
        key = (1, 1, -int(c))
        canon = _canonical_line(*key)
        if canon is not None and canon not in lines_set:
            lines_set.add(canon)
            lines_incidents[canon] = _count_incidence(points, *canon)
    
    # Now add all lines through pairs of points
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2)
            # A = y2 - y1, B = x1 - x2, C = x2*y1 - x1*y2
            A = int(y2 - y1)
            B = int(x1 - x2)
            C = int(x2 * y1 - x1 * y2)
            if A == 0 and B == 0:
                continue
            canon = _canonical_line(A, B, C)
            if canon is not None and canon not in lines_set:
                lines_set.add(canon)
                lines_incidents[canon] = _count_incidence(points, *canon)
    
    # Sort by incidence count descending
    sorted_lines = sorted(lines_incidents.items(), key=lambda x: (-x[1], x[0]))
    
    # Select top num_lines
    selected = [line for line, cnt in sorted_lines[:num_lines]]
    
    return selected

def _local_search(points, lines, max_iterations=5000):
    """
    Local search: try replacing one line at a time with a better one.
    """
    current_lines = list(lines)
    current_total = _total_incidence(points, current_lines)
    
    # Precompute incidences for all candidate lines
    n = len(points)
    all_candidate_lines = set()
    all_incidents = {}
    
    # Add lines through pairs
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = int(y2 - y1)
            B = int(x1 - x2)
            C = int(x2 * y1 - x1 * y2)
            if A == 0 and B == 0:
                continue
            canon = _canonical_line(A, B, C)
            if canon is not None:
                all_candidate_lines.add(canon)
    
    # Compute incidences for all candidates
    for line in all_candidate_lines:
        all_incidents[line] = _count_incidence(points, *line)
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for idx in range(len(current_lines)):
            current_line = current_lines[idx]
            current_inc = all_incidents.get(current_line, 0)
            
            # Try replacing with a better line not already in the set
            current_set = set(current_lines)
            best_gain = 0
            best_line = None
            for candidate in all_candidate_lines:
                if candidate in current_set:
                    continue
                cand_inc = all_incidents[candidate]
                gain = cand_inc - current_inc
                if gain > best_gain:
                    best_gain = gain
                    best_line = candidate
            
            if best_line is not None and best_gain > 0:
                current_lines[idx] = best_line
                current_total += best_gain
                improved = True
    
    return current_lines, current_total

def _validate(points, lines):
    """Validate the configuration meets all constraints."""
    # Check shapes
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    
    # Check for NaN/inf
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    
    # Check entries are close to integers
    if not np.allclose(points, np.round(points), atol=1e-6):
        return False
    if not np.allclose(lines, np.round(lines), atol=1e-6):
        return False
    
    # Check no repeated points
    if len(set(map(tuple, points.tolist()))) != 54:
        return False
    
    # Check no line with A=B=0
    if np.any((lines[:, 0] == 0) & (lines[:, 1] == 0)):
        return False
    
    # Check no duplicate lines (after canonicalization)
    canon_lines = set()
    for A, B, C in lines:
        canon = _canonical_line(int(A), int(B), int(C))
        if canon is None:
            return False
        if canon in canon_lines:
            return False
        canon_lines.add(canon)
    
    return True

def szemeredi_trotter_incidences():
    """
    Constructs a 9x6 grid of points and a set of 27 lines maximizing incidences.
    Uses local search to optimize line selection.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Points: 9x6 grid
    xs = np.arange(1, 10)  # 1 to 9
    ys = np.arange(1, 7)   # 1 to 6
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Initial line selection: horizontal, vertical, and best diagonals
    lines = []
    
    # 6 Horizontal lines: y = c -> 0x + 1y - c = 0
    for c in range(1, 7):
        lines.append((0, 1, -c))
    
    # 9 Vertical lines: x = c -> 1x + 0y - c = 0
    for c in range(1, 10):
        lines.append((1, 0, -c))
    
    # For the remaining 12 lines, select best diagonals
    # Get all diagonal lines sorted by incidence
    diag_lines = _get_all_diagonal_lines(points)
    
    # Select top 12 diagonal lines
    selected_diags = diag_lines[:12]
    for cnt, A, B, C in selected_diags:
        lines.append((A, B, C))
    
    lines = np.array(lines, dtype=np.int64)
    
    # Verify initial configuration
    initial_total = _total_incidence(points, lines)
    
    # Try local search to improve
    if time.time() < deadline - 60:  # Leave 60s buffer
        improved_lines, improved_total = _local_search(points, lines.tolist(), max_iterations=2000)
        if improved_total > initial_total:
            lines = np.array(improved_lines, dtype=np.int64)
            initial_total = improved_total
    
    # Final validation
    if not _validate(points, lines):
        # Fallback to known valid configuration
        lines = []
        for c in range(1, 7):
            lines.append((0, 1, -c))
        for c in range(1, 10):
            lines.append((1, 0, -c))
        for c in [0, -1, -2, -3]:
            lines.append((-1, 1, -c))
        for c in [7, 8, 9, 10]:
            lines.append((1, 1, -c))
        for c in [1, -4]:
            lines.append((-1, 1, -c))
        for c in [6, 11]:
            lines.append((1, 1, -c))
        lines = np.array(lines, dtype=np.int64)
    
    return points, lines

# EVOLVE-BLOCK-END
