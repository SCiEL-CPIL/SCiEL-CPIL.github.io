import numpy as np
import time
from math import gcd
from itertools import combinations

def _count_incidences(points, lines):
    """Count exact integer incidences."""
    n = len(points)
    m = len(lines)
    count = 0
    for i in range(m):
        A, B, C = lines[i]
        vals = A * points[:, 0] + B * points[:, 1] + C
        count += np.sum(vals == 0)
    return count

def _normalize_line(A, B, C):
    """Normalize a line so first nonzero entry is positive, divided by gcd."""
    g = 0
    for v in [abs(int(A)), abs(int(B)), abs(int(C))]:
        g = gcd(g, v)
    if g == 0:
        return (0, 0, 0)
    A, B, C = int(A) // g, int(B) // g, int(C) // g
    if A != 0:
        if A < 0:
            A, B, C = -A, -B, -C
    elif B != 0:
        if B < 0:
            A, B, C = -A, -B, -C
    else:
        if C < 0:
            A, B, C = -A, -B, -C
    return (A, B, C)

def _line_from_two_points(p1, p2):
    """Get normalized line through two points."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return _normalize_line(A, B, C)

def _find_best_lines_for_points(points, num_lines=27):
    """Find the best num_lines lines that maximize incidences with given points."""
    n = len(points)
    line_counts = {}
    
    # Generate all unique lines defined by pairs of points
    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_two_points(points[i], points[j])
            if line[0] == 0 and line[1] == 0:
                continue
            if line in line_counts:
                line_counts[line] += 1
            else:
                line_counts[line] = 1
                
    # Note: The count above is the number of *pairs* defining the line.
    # But we need the number of *points* on the line.
    # For a line with k points, there are k*(k-1)/2 pairs.
    # So count_pairs = k*(k-1)/2. We can invert this or just recount.
    # Recounting is safer and fast enough for ~1500 lines.
    
    unique_lines = list(line_counts.keys())
    
    # Recount actual point incidences
    final_counts = {}
    for line in unique_lines:
        A, B, C = line
        vals = A * points[:, 0] + B * points[:, 1] + C
        cnt = np.sum(vals == 0)
        final_counts[line] = cnt
        
    # Sort by count descending
    sorted_lines = sorted(final_counts.items(), key=lambda x: -x[1])
    
    selected = []
    for line, count in sorted_lines:
        if len(selected) >= num_lines:
            break
        selected.append(line)
        
    # If we don't have enough, add dummy lines (should not happen with 54 points)
    # 54 points define at least 54-1 = 53 lines? No, min lines is 1 if collinear.
    # But generally we have many.
    while len(selected) < num_lines:
        selected.append((1, 1, -(10**6 + len(selected))))
        
    return np.array(selected, dtype=np.int64)

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    # 1. Initial Grid 6x9
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = _find_best_lines_for_points(points, 27)
    best_inc = _count_incidences(points, lines)
    current_points = points.copy()
    
    # 2. Simulated Annealing Local Search
    rng = np.random.default_rng(42)
    T = 10.0
    cooling = 0.99
    
    for it in range(20000):
        if time.time() > deadline:
            break
            
        # Decay temperature
        if it % 100 == 0:
            T *= cooling
            
        # Pick a random point
        idx = rng.integers(0, len(points))
        old_x, old_y = int(current_points[idx, 0]), int(current_points[idx, 1])
        
        # Random move
        # Bias towards small moves, but allow larger ones occasionally
        if rng.random() < 0.8:
            dx = rng.integers(-2, 3)
            dy = rng.integers(-2, 3)
        else:
            dx = rng.integers(-5, 6)
            dy = rng.integers(-5, 6)
            
        new_x, new_y = old_x + dx, old_y + dy
        
        # Check collision
        # Vectorized check is fast
        new_pt = np.array([new_x, new_y])
        if np.any(np.all(current_points == new_pt, axis=1)):
            continue
            
        # Apply move
        current_points[idx, 0] = new_x
        current_points[idx, 1] = new_y
        
        # Re-optimize lines
        new_lines = _find_best_lines_for_points(current_points, 27)
        new_inc = _count_incidences(current_points, new_lines)
        
        delta = new_inc - best_inc
        
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
            best_inc = new_inc
            points = current_points.copy()
            lines = new_lines
        else:
            # Revert
            current_points[idx, 0] = old_x
            current_points[idx, 1] = old_y
            
    # Final validation and cleanup
    # Ensure no duplicate points
    unique_points = np.unique(points, axis=0)
    if len(unique_points) != len(points):
        # Fallback to grid
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = _find_best_lines_for_points(points, 27)
        
    # Ensure lines are valid and unique
    norm_lines = [_normalize_line(*l) for l in lines]
    seen = set()
    unique_norm = []
    for nl in norm_lines:
        if nl not in seen:
            seen.add(nl)
            unique_norm.append(nl)
            
    if len(unique_norm) < 27:
         # Fill with dummy lines if needed (unlikely)
         while len(unique_norm) < 27:
             unique_norm.append((1, 1, -(10**6 + len(unique_norm))))
             
    lines = np.array(unique_norm[:27], dtype=np.int64)
    
    return points, lines
