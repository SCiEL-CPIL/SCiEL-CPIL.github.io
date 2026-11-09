# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from math import gcd
from collections import defaultdict

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) for A*x + B*y + C = 0."""
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    # Make first non-zero entry positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _compute_top27(points):
    """
    Compute the top 27 lines by incidence count for a set of points.
    Uses vectorized operations for speed.
    Returns (score, lines_array) where score is the sum of incidences of the top 27 lines.
    """
    n = len(points)
    if n < 2:
        return 0, np.zeros((27, 3), dtype=np.int64)
    
    pts = points.astype(np.int64)
    
    # Step 1: Find all unique lines defined by pairs of points
    line_set = set()
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            A = y2 - y1
            B = -(x2 - x1)
            C = x2 * y1 - x1 * y2
            key = _normalize_line(A, B, C)
            if key is not None:
                line_set.add(key)
                
    # Step 2: Count incidences for each line using vectorized ops
    counts = []
    pts_array = pts  # (n, 2)
    for (A, B, C) in line_set:
        # Vectorized: A*x + B*y + C == 0 for all points
        vals = A * pts_array[:, 0] + B * pts_array[:, 1] + C
        count = np.sum(vals == 0)
        if count >= 2:  # Only consider lines with at least 2 points
            counts.append((count, (A, B, C)))
        
    # Step 3: Sort by count descending, then by line tuple for determinism
    counts.sort(key=lambda x: (-x[0], x[1]))
    
    # Step 4: Select top 27
    if len(counts) >= 27:
        selected = counts[:27]
    else:
        selected = counts
        
    score = sum(c for c, _ in selected)
    lines = np.array([l for _, l in selected], dtype=np.int64)
    
    return score, lines

def _generate_grid(a, b):
    """Generate an a x b grid of points starting at (1,1)."""
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _generate_hexagonal_grid(rows, cols):
    """Generate a hexagonal-like grid pattern."""
    pts = []
    for i in range(rows):
        for j in range(cols):
            x = j + (i % 2) * 0.5
            y = i
            pts.append((int(x * 2), int(y * 2)))  # Scale to integers
    pts = np.array(pts, dtype=np.int64)
    # Remove duplicates
    unique = np.unique(pts, axis=0)
    return unique

def _remove_points(points, indices):
    """Remove points at given indices."""
    mask = np.ones(len(points), dtype=bool)
    mask[list(indices)] = False
    return points[mask]

def _add_points(points, new_pts):
    """Add new points to the array."""
    new_arr = np.array(new_pts, dtype=np.int64)
    return np.vstack([points, new_arr])

def _check_valid_points(points):
    """Basic validity check for points."""
    if len(points) != 54:
        return False
    if points.shape != (54, 2):
        return False
    # Check for duplicates
    pt_set = set(map(tuple, points))
    if len(pt_set) != 54:
        return False
    return True

def _compute_line_incidence_map(points):
    """
    Compute a map from each point index to the set of lines (normalized) passing through it.
    Returns: point_lines: list of sets, each set contains normalized line tuples
    """
    n = len(points)
    pts = points.astype(np.int64)
    point_lines = [set() for _ in range(n)]
    
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            A = y2 - y1
            B = -(x2 - x1)
            C = x2 * y1 - x1 * y2
            key = _normalize_line(A, B, C)
            if key is not None:
                point_lines[i].add(key)
                point_lines[j].add(key)
    
    return point_lines

def _compute_top27_with_map(points, point_lines=None):
    """
    Compute top 27 lines, optionally using a precomputed point-to-lines map.
    Returns (score, lines_array, line_counts_dict)
    """
    n = len(points)
    if n < 2:
        return 0, np.zeros((27, 3), dtype=np.int64), {}
    
    pts = points.astype(np.int64)
    
    # Build line -> set of point indices
    line_points = defaultdict(set)
    for i in range(n):
        if point_lines is not None:
            for line in point_lines[i]:
                line_points[line].add(i)
        else:
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                A = y2 - y1
                B = -(x2 - x1)
                C = x2 * y1 - x1 * y2
                key = _normalize_line(A, B, C)
                if key is not None:
                    line_points[key].add(i)
                    line_points[key].add(j)
    
    # Count incidences
    counts = []
    for line, pt_indices in line_points.items():
        count = len(pt_indices)
        if count >= 2:
            counts.append((count, line))
    
    # Sort and select top 27
    counts.sort(key=lambda x: (-x[0], x[1]))
    if len(counts) >= 27:
        selected = counts[:27]
    else:
        selected = counts
        
    score = sum(c for c, _ in selected)
    lines = np.array([l for _, l in selected], dtype=np.int64)
    
    return score, lines, dict(line_points)

def _local_search_incremental(points, deadline, best_score, best_points, best_lines):
    """
    Local search with incremental updates.
    When moving point i, only lines through i are affected.
    """
    n = len(points)
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
    
    current_points = points.copy()
    current_score, _, current_line_points = _compute_top27_with_map(current_points)
    
    if current_score > best_score:
        best_score = current_score
        best_points = current_points.copy()
        best_lines = _compute_top27(current_points)[1]
    
    improved = True
    passes = 0
    while improved and time.time() < deadline and passes < 50:
        improved = False
        passes += 1
        
        for idx in range(n):
            if time.time() > deadline:
                return best_score, best_points, best_lines
            
            for dx, dy in neighbors:
                if time.time() > deadline:
                    return best_score, best_points, best_lines
                    
                new_points = current_points.copy()
                new_points[idx] = current_points[idx] + np.array([dx, dy])
                
                if not _check_valid_points(new_points):
                    continue
                    
                # Incremental update: only lines through point idx change
                # Get lines that passed through idx in old config
                old_lines_through_idx = set()
                x_old, y_old = current_points[idx]
                for i in range(n):
                    if i == idx:
                        continue
                    x2, y2 = current_points[i]
                    A = y2 - y_old
                    B = -(x2 - x_old)
                    C = x2 * y_old - x_old * y2
                    key = _normalize_line(A, B, C)
                    if key is not None:
                        old_lines_through_idx.add(key)
                
                # New lines through idx
                new_lines_through_idx = set()
                x_new, y_new = new_points[idx]
                for i in range(n):
                    if i == idx:
                        continue
                    x2, y2 = new_points[i]
                    A = y2 - y_new
                    B = -(x2 - x_new)
                    C = x2 * y_new - x_new * y2
                    key = _normalize_line(A, B, C)
                    if key is not None:
                        new_lines_through_idx.add(key)
                
                # Lines that are added or removed
                added_lines = new_lines_through_idx - old_lines_through_idx
                removed_lines = old_lines_through_idx - new_lines_through_idx
                
                # For each affected line, update its count
                # Lines in added_lines: idx is now on them, so +1
                # Lines in removed_lines: idx is no longer on them, so -1
                # Lines in both: no change to count (still on line)
                
                # We need to recompute counts for affected lines
                # But this is complex. For simplicity, do full recompute if few changes.
                # Actually, let's just do full recompute - it's fast enough with vectorization.
                
                new_score, new_lines = _compute_top27(new_points)
                
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    best_lines = new_lines
                    current_points = new_points.copy()
                    current_score = new_score
                    current_line_points = _compute_top27_with_map(new_points)[2]
                    improved = True
                    break  # Restart for this point
    
    return best_score, best_points, best_lines

def _multi_point_local_search(points, deadline, best_score, best_points, best_lines):
    """
    Try moving pairs of points simultaneously.
    """
    n = len(points)
    small_moves = [(-1,0),(1,0),(0,-1),(0,1)]
    
    current_points = points.copy()
    current_score, _ = _compute_top27(current_points)
    
    if current_score > best_score:
        best_score = current_score
        best_points = current_points.copy()
        best_lines = _compute_top27(current_points)[1]
    
    # Try pairs of adjacent points
    for i in range(n):
        if time.time() > deadline:
            return best_score, best_points, best_lines
        for j in range(i+1, min(i+5, n)):  # Only nearby pairs
            for dx1, dy1 in small_moves:
                for dx2, dy2 in small_moves:
                    if time.time() > deadline:
                        return best_score, best_points, best_lines
                    
                    new_points = current_points.copy()
                    new_points[i] = current_points[i] + np.array([dx1, dy1])
                    new_points[j] = current_points[j] + np.array([dx2, dy2])
                    
                    if not _check_valid_points(new_points):
                        continue
                        
                    new_score, new_lines = _compute_top27(new_points)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
                        best_lines = new_lines
                        current_points = new_points.copy()
                        current_score = new_score
    
    return best_score, best_points, best_lines

def _simulated_annealing(points, deadline, best_score, best_points, best_lines):
    """
    Simulated annealing to escape local optima.
    """
    n = len(points)
    current = points.copy()
    current_score, _ = _compute_top27(current)
    
    if current_score > best_score:
        best_score = current_score
        best_points = current.copy()
        best_lines = _compute_top27(current)[1]
    
    T = 15.0
    T_min = 0.5
    cooling = 0.97
    
    while time.time() < deadline and T > T_min:
        # Pick random point and random move
        idx = np.random.randint(n)
        dx = np.random.randint(-4, 5)
        dy = np.random.randint(-4, 5)
        if dx == 0 and dy == 0:
            T *= cooling
            continue
        
        new_points = current.copy()
        new_points[idx] = current[idx] + np.array([dx, dy])
        
        if not _check_valid_points(new_points):
            T *= cooling
            continue
        
        new_score, new_lines = _compute_top27(new_points)
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current = new_points
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                best_lines = new_lines
        
        T *= cooling
    
    return best_score, best_points, best_lines

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    Uses a combination of grid configurations, local search, and simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 615  # Safety margin (630s limit)
    
    np.random.seed(42)
    
    best_score = -1
    best_points = None
    best_lines = None
    
    def update_best(pts, score, lines):
        nonlocal best_score, best_points, best_lines
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines

    # Phase 1: Systematic grid search with modifications
    grid_configs = [
        (6, 9, [], []),
        (9, 6, [], []),
        (3, 18, [], []),
        (18, 3, [], []),
        (2, 27, [], []),
        (27, 2, [], []),
        (5, 11, [0], []),
        (11, 5, [0], []),
        (7, 8, [0], []),
        (7, 8, [55], []),
        (7, 8, [0, 55], []),
        (4, 14, [0], []),
        (4, 14, [0, 55], []),
        (5, 10, [], [(1,11),(2,11),(3,11),(4,11)]),
        (5, 10, [], [(6,1),(6,2),(6,3),(6,4)]),
        (10, 5, [], [(11,1),(11,2),(11,3),(11,4)]),
        (10, 5, [], [(1,6),(2,6),(3,6),(4,6)]),
        (4, 12, [], [(1,13),(2,13),(3,13),(4,13),(5,13),(6,13)]),
        (4, 12, [], [(5,1),(5,2),(5,3),(5,4),(5,5),(5,6)]),
        (3, 17, [], [(4,1),(4,2),(4,3)]),
        (4, 13, [], [(1,14),(2,14),(3,14),(4,14)]),
        (13, 4, [], [(14,1),(14,2),(14,3),(14,4)]),
        (5, 12, [], [(1,13),(2,13),(3,13),(4,13),(5,13)]),
        (12, 5, [], [(13,1),(13,2),(13,3),(13,4),(13,5)]),
        (6, 10, [], [(1,11),(2,11),(3,11),(4,11),(5,11),(6,11)]),
        (10, 6, [], [(11,1),(11,2),(11,3),(11,4),(11,5),(11,6)]),
    ]
    
    for a, b, rem, add in grid_configs:
        if time.time() > deadline - 180:
            break
        pts = _generate_grid(a, b)
        if rem:
            pts = _remove_points(pts, rem)
        if add:
            pts = _add_points(pts, add)
        if len(pts) != 54:
            continue
        if not _check_valid_points(pts):
            continue
        score, lines = _compute_top27(pts)
        update_best(pts, score, lines)
        
    # Also try some non-rectangular configurations
    if time.time() < deadline - 180:
        # Diamond/rhombus patterns
        for scale in [1, 2, 3]:
            pts = []
            for i in range(-scale, scale+1):
                for j in range(-scale, scale+1):
                    if abs(i) + abs(j) <= scale + 1:
                        pts.append((i*scale, j*scale))
            pts = np.array(pts, dtype=np.int64)
            if len(pts) > 54:
                pts = pts[:54]
            elif len(pts) < 54:
                # Pad with nearby points
                extra = []
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        pt = (scale + dx, scale + dy)
                        if pt not in pts:
                            extra.append(pt)
                        if len(pts) + len(extra) >= 54:
                            break
                    if len(pts) + len(extra) >= 54:
                        break
                pts = np.vstack([pts, np.array(extra[:54-len(pts)], dtype=np.int64)]) if extra else pts
            
            if len(pts) == 54 and _check_valid_points(pts):
                score, lines = _compute_top27(pts)
                update_best(pts, score, lines)
    
    # Phase 2: Local search from best configuration
    if best_points is not None and time.time() < deadline - 90:
        best_score, best_points, best_lines = _local_search_incremental(
            best_points, deadline - 60, best_score, best_points, best_lines
        )
        
    # Phase 3: Multi-point local search
    if best_points is not None and time.time() < deadline - 45:
        best_score, best_points, best_lines = _multi_point_local_search(
            best_points, deadline - 30, best_score, best_points, best_lines
        )
        
    # Phase 4: Simulated Annealing to explore further
    if best_points is not None and time.time() < deadline - 15:
        best_score, best_points, best_lines = _simulated_annealing(
            best_points, deadline - 5, best_score, best_points, best_lines
        )
        
    # Final local search polish
    if best_points is not None and time.time() < deadline - 5:
        best_score, best_points, best_lines = _local_search_incremental(
            best_points, deadline - 2, best_score, best_points, best_lines
        )
        
    # Fallback if something went wrong
    if best_points is None:
        pts = _generate_grid(6, 9)
        score, lines = _compute_top27(pts)
        return pts, lines

    return best_points, best_lines

# EVOLVE-BLOCK-END
