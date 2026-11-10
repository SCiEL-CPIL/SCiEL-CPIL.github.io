# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
from itertools import combinations
import time

def _normalize_line(A, B, C):
    """Normalize line (A,B,C) so first nonzero of (A,B) is positive, and gcd(|A|,|B|,|C|)=1."""
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C)) if C != 0 else gcd(abs(A), abs(B))
    if g == 0:
        g = 1
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidence(point, A, B, C):
    """Check if point (x,y) lies on line Ax + By + C = 0."""
    return A * point[0] + B * point[1] + C == 0

def _line_intersection(l1, l2):
    """
    Compute intersection of two lines (A1,B1,C1) and (A2,B2,C2).
    Returns (x, y) as tuple of ints if integer intersection exists, else None.
    Line: A*x + B*y + C = 0
    """
    A1, B1, C1 = l1
    A2, B2, C2 = l2
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None  # parallel or same line
    # x = (B1*C2 - B2*C1) / det
    # y = (A2*C1 - A1*C2) / det
    num_x = B1 * C2 - B2 * C1
    num_y = A2 * C1 - A1 * C2
    if num_x % det != 0 or num_y % det != 0:
        return None
    x = num_x // det
    y = num_y // det
    return (x, y)

def _verify_result(points, lines):
    """Verify the result meets all validity constraints. Returns (valid, I)."""
    if not isinstance(points, np.ndarray) or points.shape != (54, 2):
        return False, 0
    if not isinstance(lines, np.ndarray) or lines.shape != (27, 3):
        return False, 0
    # Check for NaN/inf
    if not np.all(np.isfinite(points)):
        return False, 0
    if not np.all(np.isfinite(lines)):
        return False, 0
    # Check entries are close to integers
    if not np.allclose(points, np.round(points), atol=1e-6):
        return False, 0
    if not np.allclose(lines, np.round(lines), atol=1e-6):
        return False, 0
    # Round to int64
    points = np.round(points).astype(np.int64)
    lines = np.round(lines).astype(np.int64)
    # Check for repeated points
    if len(np.unique(points, axis=0)) != 54:
        return False, 0
    # Check lines: no A=B=0, no duplicate lines
    seen_lines = set()
    for i in range(27):
        A, B, C = lines[i]
        if A == 0 and B == 0:
            return False, 0
        norm = _normalize_line(int(A), int(B), int(C))
        if norm is None:
            return False, 0
        if norm in seen_lines:
            return False, 0
        seen_lines.add(norm)
    # Count incidences
    I = 0
    for p in points:
        for l in lines:
            if l[0] * p[0] + l[1] * p[1] + l[2] == 0:
                I += 1
    return True, I

def _build_line_families():
    """
    Build a diverse set of candidate lines.
    We want lines that intersect each other at many integer points.
    """
    lines = set()
    
    # Horizontal lines: y = c  =>  0*x + 1*y - c = 0
    for c in range(-10, 21):
        lines.add(_normalize_line(0, 1, -c))
    
    # Vertical lines: x = c  =>  1*x + 0*y - c = 0
    for c in range(-10, 21):
        lines.add(_normalize_line(1, 0, -c))
    
    # Diagonal lines with slope 1: y - x = c  =>  -1*x + 1*y - c = 0
    for c in range(-15, 16):
        lines.add(_normalize_line(-1, 1, -c))
    
    # Diagonal lines with slope -1: y + x = c  =>  1*x + 1*y - c = 0
    for c in range(-15, 16):
        lines.add(_normalize_line(1, 1, -c))
    
    # Slope 2 lines: y = 2x + c  =>  -2*x + 1*y - c = 0
    for c in range(-10, 11):
        lines.add(_normalize_line(-2, 1, -c))
    
    # Slope -2 lines: y = -2x + c  =>  2*x + 1*y - c = 0
    for c in range(-10, 11):
        lines.add(_normalize_line(2, 1, -c))
    
    # Slope 1/2 lines: 2y = x + c  =>  -1*x + 2*y - c = 0
    for c in range(-15, 16):
        lines.add(_normalize_line(-1, 2, -c))
    
    # Slope -1/2 lines: 2y = -x + c  =>  1*x + 2*y - c = 0
    for c in range(-15, 16):
        lines.add(_normalize_line(1, 2, -c))
    
    return list(lines)

def _compute_point_incidence_counts(points, lines):
    """For each point, count how many lines it lies on."""
    n_points = len(points)
    n_lines = len(lines)
    counts = np.zeros(n_points, dtype=np.int64)
    for i, p in enumerate(points):
        for j, l in enumerate(lines):
            if l[0] * p[0] + l[1] * p[1] + l[2] == 0:
                counts[i] += 1
    return counts

def _select_best_lines_for_points(points, all_lines, n_lines=27):
    """
    Given a set of points, select n_lines from all_lines to maximize total incidences.
    Uses greedy: at each step, pick the line with the most incidences with current points.
    Since each point-line pair is independent, greedy is optimal.
    """
    # Count incidences for each line with all points
    line_counts = {}
    for l in all_lines:
        count = 0
        for p in points:
            if l[0] * p[0] + l[1] * p[1] + l[2] == 0:
                count += 1
        line_counts[l] = count
    
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    selected = [l for l, c in sorted_lines[:n_lines]]
    return selected

def _greedy_point_selection(all_lines, n_points=54, max_iter=100):
    """
    Greedily select points to maximize incidences with all_lines.
    At each step, pick the point (from candidate intersections) that maximizes
    the number of lines it lies on among all_lines, with tie-breaking by total
    potential.
    """
    # First, compute all integer intersections of pairs of lines
    n_lines = len(all_lines)
    intersections = {}  # point -> set of line indices it lies on
    
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(all_lines[i], all_lines[j])
            if pt is not None:
                x, y = pt
                # Keep points within reasonable bounds
                if -20 <= x <= 20 and -20 <= y <= 20:
                    if pt not in intersections:
                        intersections[pt] = set()
                    intersections[pt].add(i)
                    intersections[pt].add(j)
    
    # For each intersection point, compute how many of all_lines it lies on
    # (it might lie on more lines than just the two that created it)
    point_line_counts = {}
    for pt, line_set in intersections.items():
        count = 0
        for idx in range(n_lines):
            A, B, C = all_lines[idx]
            if A * pt[0] + B * pt[1] + C == 0:
                count += 1
        point_line_counts[pt] = count
    
    # Sort candidate points by count descending
    sorted_points = sorted(point_line_counts.items(), key=lambda x: -x[1])
    
    # Greedily select points
    selected_points = []
    used_points = set()
    
    for pt, count in sorted_points:
        if len(selected_points) == n_points:
            break
        if pt not in used_points:
            selected_points.append(pt)
            used_points.add(pt)
    
    # If we don't have enough points, add more from a grid
    if len(selected_points) < n_points:
        # Add points from a grid in the region
        for x in range(-15, 16):
            for y in range(-15, 16):
                if len(selected_points) == n_points:
                    break
                pt = (x, y)
                if pt not in used_points:
                    selected_points.append(pt)
                    used_points.add(pt)
            if len(selected_points) == n_points:
                break
    
    return selected_points[:n_points]

def _local_search(points, all_lines, n_lines=27, n_points=54, max_time=300):
    """
    Local search to improve the point set.
    Try replacing individual points with better candidates.
    """
    start_time = time.time()
    points = list(points)
    
    # Precompute line coefficients as numpy arrays for speed
    lines_arr = np.array(all_lines, dtype=np.int64)
    
    def compute_I(pts):
        """Compute total incidences."""
        total = 0
        for p in pts:
            # Vectorized check
            vals = lines_arr[:, 0] * p[0] + lines_arr[:, 1] * p[1] + lines_arr[:, 2]
            total += np.sum(vals == 0)
        return total
    
    current_I = compute_I(points)
    
    # Generate candidate points: all integer intersections + grid points
    candidate_points = set()
    n_lines_total = len(all_lines)
    for i in range(n_lines_total):
        for j in range(i + 1, n_lines_total):
            pt = _line_intersection(all_lines[i], all_lines[j])
            if pt is not None:
                x, y = pt
                if -20 <= x <= 20 and -20 <= y <= 20:
                    candidate_points.add(pt)
    
    # Add grid points
    for x in range(-15, 16):
        for y in range(-15, 16):
            candidate_points.add((x, y))
    
    candidate_points = list(candidate_points)
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        for i in range(n_points):
            if (time.time() - start_time) > max_time - 10:
                break
            old_pt = points[i]
            old_I = current_I
            # Try replacing point i with each candidate
            best_new_pt = None
            best_new_I = old_I
            for new_pt in candidate_points:
                if new_pt == old_pt:
                    continue
                # Compute new I
                # Remove old point's contributions, add new point's contributions
                old_contrib = 0
                for l in all_lines:
                    if l[0] * old_pt[0] + l[1] * old_pt[1] + l[2] == 0:
                        old_contrib += 1
                new_contrib = 0
                for l in all_lines:
                    if l[0] * new_pt[0] + l[1] * new_pt[1] + l[2] == 0:
                        new_contrib += 1
                new_I = old_I - old_contrib + new_contrib
                if new_I > best_new_I:
                    best_new_I = new_I
                    best_new_pt = new_pt
            if best_new_pt is not None:
                points[i] = best_new_pt
                current_I = best_new_I
                improved = True
    
    return points, current_I

def _optimize_line_selection(points, all_lines, n_lines=27):
    """
    Given fixed points, select the best n_lines from all_lines.
    Greedy is optimal since incidences are independent per line.
    """
    line_counts = {}
    for l in all_lines:
        count = 0
        for p in points:
            if l[0] * p[0] + l[1] * p[1] + l[2] == 0:
                count += 1
        line_counts[l] = count
    
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    selected = [l for l, c in sorted_lines[:n_lines]]
    return selected

def _alternate_optimization(all_lines, n_points=54, n_lines=27, max_rounds=20, max_time=300):
    """
    Alternate between:
    1. Fix lines, optimize points
    2. Fix points, optimize lines
    """
    start_time = time.time()
    
    # Initialize with greedy point selection
    points = _greedy_point_selection(all_lines, n_points)
    points = [tuple(p) for p in points]
    
    # Initialize lines
    lines = _optimize_line_selection(points, all_lines, n_lines)
    
    best_I = 0
    best_points = list(points)
    best_lines = list(lines)
    
    for round_num in range(max_rounds):
        if (time.time() - start_time) > max_time - 30:
            break
        
        # Step 1: Fix lines, optimize points via local search
        points, I = _local_search(points, lines, n_lines, n_points, max_time=30)
        
        # Step 2: Fix points, optimize lines
        lines = _optimize_line_selection(points, all_lines, n_lines)
        
        # Recompute I
        I = 0
        for p in points:
            for l in lines:
                if l[0] * p[0] + l[1] * p[1] + l[2] == 0:
                    I += 1
        
        if I > best_I:
            best_I = I
            best_points = list(points)
            best_lines = list(lines)
    
    return best_points, best_lines, best_I

def _build_diverse_line_set():
    """
    Build a diverse set of lines with various slopes and intercepts.
    Focus on lines that create many integer intersections.
    """
    lines = set()
    
    # Horizontal lines
    for c in range(-12, 13):
        lines.add(_normalize_line(0, 1, -c))
    
    # Vertical lines
    for c in range(-12, 13):
        lines.add(_normalize_line(1, 0, -c))
    
    # Slope 1: y - x = c
    for c in range(-18, 19):
        lines.add(_normalize_line(-1, 1, -c))
    
    # Slope -1: y + x = c
    for c in range(-18, 19):
        lines.add(_normalize_line(1, 1, -c))
    
    # Slope 2: y = 2x + c
    for c in range(-12, 13):
        lines.add(_normalize_line(-2, 1, -c))
    
    # Slope -2: y = -2x + c
    for c in range(-12, 13):
        lines.add(_normalize_line(2, 1, -c))
    
    # Slope 1/2: 2y = x + c
    for c in range(-18, 19):
        lines.add(_normalize_line(-1, 2, -c))
    
    # Slope -1/2: 2y = -x + c
    for c in range(-18, 19):
        lines.add(_normalize_line(1, 2, -c))
    
    # Slope 3: y = 3x + c
    for c in range(-10, 11):
        lines.add(_normalize_line(-3, 1, -c))
    
    # Slope -3: y = -3x + c
    for c in range(-10, 11):
        lines.add(_normalize_line(3, 1, -c))
    
    # Slope 1/3: 3y = x + c
    for c in range(-12, 13):
        lines.add(_normalize_line(-1, 3, -c))
    
    # Slope -1/3: 3y = -x + c
    for c in range(-12, 13):
        lines.add(_normalize_line(1, 3, -c))
    
    # Slope 3/2: 2y = 3x + c
    for c in range(-10, 11):
        lines.add(_normalize_line(-3, 2, -c))
    
    # Slope -3/2: 2y = -3x + c
    for c in range(-10, 11):
        lines.add(_normalize_line(3, 2, -c))
    
    # Slope 2/3: 3y = 2x + c
    for c in range(-12, 13):
        lines.add(_normalize_line(-2, 3, -c))
    
    # Slope -2/3: 3y = -2x + c
    for c in range(-12, 13):
        lines.add(_normalize_line(2, 3, -c))
    
    return list(lines)

def szemeredi_trotter_incidences():
    """
    Constructs 54 points and 27 lines to maximize point-line incidences.
    Uses line-first construction with alternating optimization.
    """
    start_time = time.time()
    deadline = 600  # safety margin
    
    # Build a diverse set of candidate lines
    all_lines = _build_diverse_line_set()
    
    # Run alternating optimization
    points, lines, I = _alternate_optimization(all_lines, 54, 27, max_rounds=15, max_time=400)
    
    # Convert to numpy arrays
    points_arr = np.array(points, dtype=np.int64)
    lines_arr = np.array(lines, dtype=np.int64)
    
    # Verify result
    valid, I_check = _verify_result(points_arr, lines_arr)
    
    if not valid:
        # Fallback to a known valid configuration
        # 6x9 grid with best 27 lines
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_arr = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Generate all lines through pairs of points
        candidate_lines = set()
        for i in range(54):
            for j in range(i + 1, 54):
                x1, y1 = points_arr[i]
                x2, y2 = points_arr[j]
                A = y2 - y1
                B = -(x2 - x1)
                C = (x2 - x1) * y1 - (y2 - y1) * x2
                normalized = _normalize_line(A, B, C)
                if normalized is not None:
                    candidate_lines.add(normalized)
        
        # Count incidences for each line
        line_incidents = {}
        for line in candidate_lines:
            A, B, C = line
            count = 0
            for p in points_arr:
                if A * p[0] + B * p[1] + C == 0:
                    count += 1
            line_incidents[line] = count
        
        sorted_lines = sorted(line_incidents.items(), key=lambda x: -x[1])
        selected_lines = [l for l, c in sorted_lines[:27]]
        lines_arr = np.array(selected_lines, dtype=np.int64)
    
    # Final check
    valid, I_final = _verify_result(points_arr, lines_arr)
    if not valid:
        # Absolute fallback: simple grid
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_arr = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_arr = np.zeros((27, 3), dtype=np.int64)
        for i in range(27):
            if i < 9:
                lines_arr[i] = [0, 1, -(i + 1)]  # y = i+1
            elif i < 18:
                lines_arr[i] = [1, 0, -(i - 8)]  # x = i-8+1
            else:
                lines_arr[i] = [1, 1, -(i - 17)]  # x + y = i-17+1
    
    return points_arr, lines_arr

# EVOLVE-BLOCK-END
