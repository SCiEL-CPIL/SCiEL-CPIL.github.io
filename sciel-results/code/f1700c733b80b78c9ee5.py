import numpy as np
from itertools import product
import time

def _normalize_line(A, B, C):
    """Normalize line so first nonzero of (A,B,C) is positive, divided by gcd."""
    from math import gcd
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_from_points(p1, p2):
    """Get line through two points (x1,y1) and (x2,y2) as (A,B,C)."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return _normalize_line(A, B, C)

def _count_incidences(points, lines):
    """Count exact incidences."""
    count = 0
    for (x, y) in points:
        for (A, B, C) in lines:
            if A * x + B * y + C == 0:
                count += 1
    return count

def _pick_top_k_lines(points, k):
    """Pick k lines that maximize incidences with given points."""
    from collections import Counter
    # Generate all lines through pairs of points
    line_counts = Counter()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_points(points[i], points[j])
            line_counts[line] += 1
    
    # Sort by count, take top k
    top_lines = line_counts.most_common(k)
    return [line for line, count in top_lines]

def _evaluate_grid_config(xs, ys, line_specs):
    """
    Evaluate a grid configuration.
    xs: list of x-coordinates
    ys: list of y-coordinates  
    line_specs: list of (slope_num, slope_den, intercept_offset) for lines y = (num/den)*x + offset
    Returns number of incidences.
    """
    points = [(x, y) for x in xs for y in ys]
    
    lines = []
    seen = set()
    for (num, den, offset) in line_specs:
        # Line: den*y = num*x + den*offset => num*x - den*y + den*offset = 0
        A = num
        B = -den
        C = den * offset
        norm = _normalize_line(A, B, C)
        if norm not in seen:
            seen.add(norm)
            lines.append(norm)
    
    # Pad with additional lines if needed
    if len(lines) < 27:
        # Add more lines from the point set
        from collections import Counter
        line_counts = Counter()
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                line = _line_from_points(points[i], points[j])
                if line not in seen:
                    line_counts[line] += 1
        for line, count in line_counts.most_common(27 - len(lines)):
            if line not in seen:
                seen.add(line)
                lines.append(line)
    
    if len(lines) > 27:
        lines = lines[:27]
    
    return _count_incidences(points, lines), points, lines

def _search_structured():
    """Search over structured grid configurations."""
    best_count = 0
    best_points = None
    best_lines = None
    
    # Try various grid shapes
    grid_shapes = [
        (1, 54), (2, 27), (3, 18), (6, 9), (9, 6), (18, 3), (27, 2), (54, 1)
    ]
    
    for (nx, ny) in grid_shapes:
        # Try different x and y coordinate sets
        x_options = [
            list(range(nx)),
            list(range(1, nx + 1)),
        ]
        y_options = [
            list(range(ny)),
            list(range(1, ny + 1)),
        ]
        
        for xs in x_options:
            for ys in y_options:
                # Generate candidate lines: all lines with small slopes
                line_specs = []
                # Horizontal lines
                for y in ys:
                    line_specs.append((0, 1, y))
                # Vertical lines (represented differently)
                # For now, use slope-based lines
                for num in range(-5, 6):
                    for den in [1, 2, 3]:
                        for offset in range(-10, 11):
                            line_specs.append((num, den, offset))
                
                count, points, lines = _evaluate_grid_config(xs, ys, line_specs)
                if count > best_count:
                    best_count = count
                    best_points = points
                    best_lines = lines
    
    return best_count, best_points, best_lines

def _search_pair_lines(points):
    """Given fixed points, find best 27 lines."""
    from collections import Counter
    line_counts = Counter()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_points(points[i], points[j])
            line_counts[line] += 1
    
    top_lines = line_counts.most_common(27)
    lines = [line for line, count in top_lines]
    return _count_incidences(points, lines), points, lines

def _try_grid(nx, ny, x_start=0, y_start=0):
    """Try a rectangular grid."""
    xs = list(range(x_start, x_start + nx))
    ys = list(range(y_start, y_start + ny))
    points = [(x, y) for x in xs for y in ys]
    return _search_pair_lines(points)

def _try_double_grid(nx1, ny1, nx2, ny2, offset_x=0, offset_y=0):
    """Try a union of two grids."""
    points1 = [(x, y) for x in range(nx1) for y in range(ny1)]
    points2 = [(x + offset_x, y + offset_y) for x in range(nx2) for y in range(ny2)]
    
    # Remove duplicates
    seen = set()
    points = []
    for p in points1 + points2:
        if p not in seen:
            seen.add(p)
            points.append(p)
    
    if len(points) != 54:
        return 0, None, None
    
    return _search_pair_lines(points)

def _try_line_arrangement():
    """Try arrangements based on line structure first."""
    best_count = 0
    best_points = None
    best_lines = None
    
    # Strategy: choose 27 lines, then find 54 points that maximize incidences
    # Start with simple line families
    
    # Family 1: horizontal + vertical + diagonal
    for nh in range(1, 15):  # number of horizontal lines
        for nv in range(1, 15):  # number of vertical lines
            nd = 27 - nh - nv
            if nd < 0:
                continue
            
            # Place lines
            lines = []
            # Horizontal: y = b
            for b in range(nh):
                lines.append(_normalize_line(0, 1, -b))
            # Vertical: x = a
            for a in range(nv):
                lines.append(_normalize_line(1, 0, -a))
            # Diagonal: y = x + c
            for c in range(nd):
                lines.append(_normalize_line(1, -1, c))
            
            # Find best 54 points: intersections of these lines
            intersections = set()
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    A1, B1, C1 = lines[i]
                    A2, B2, C2 = lines[j]
                    det = A1 * B2 - A2 * B1
                    if det == 0:
                        continue
                    x = (B1 * C2 - B2 * C1) / det
                    y = (A2 * C1 - A1 * C2) / det
                    if abs(x - round(x)) < 1e-9 and abs(y - round(y)) < 1e-9:
                        intersections.add((int(round(x)), int(round(y))))
            
            # Pick 54 points from intersections
            if len(intersections) >= 54:
                # Score each point by how many lines it's on
                point_scores = {}
                for p in intersections:
                    score = 0
                    for (A, B, C) in lines:
                        if A * p[0] + B * p[1] + C == 0:
                            score += 1
                    point_scores[p] = score
                
                # Take top 54
                sorted_points = sorted(point_scores.items(), key=lambda x: -x[1])[:54]
                points = [p for p, s in sorted_points]
                
                count = _count_incidences(points, lines)
                if count > best_count:
                    best_count = count
                    best_points = points
                    best_lines = lines
    
    return best_count, best_points, best_lines

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization with 54 points and 27 lines.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin
    
    best_count = 0
    best_points = None
    best_lines = None
    
    # Method 1: Structured grid search
    count, points, lines = _search_structured()
    if count > best_count and points is not None:
        best_count = count
        best_points = points
        best_lines = lines
    
    if time.time() > deadline:
        return np.array(best_points, dtype=np.int64), np.array(best_lines, dtype=np.int64)
    
    # Method 2: Try specific grid shapes with pair-line optimization
    grid_configs = [
        (3, 18), (6, 9), (9, 6), (18, 3), (2, 27), (27, 2), (1, 54), (54, 1)
    ]
    
    for (nx, ny) in grid_configs:
        for x_start in range(-2, 3):
            for y_start in range(-2, 3):
                if time.time() > deadline:
                    break
                count, points, lines = _try_grid(nx, ny, x_start, y_start)
                if count > best_count and points is not None:
                    best_count = count
                    best_points = points
                    best_lines = lines
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        return np.array(best_points, dtype=np.int64), np.array(best_lines, dtype=np.int64)
    
    # Method 3: Line arrangement search
    count, points, lines = _try_line_arrangement()
    if count > best_count and points is not None:
        best_count = count
        best_points = points
        best_lines = lines
    
    if time.time() > deadline:
        return np.array(best_points, dtype=np.int64), np.array(best_lines, dtype=np.int64)
    
    # Method 4: Try double grids
    double_grid_configs = [
        (3, 9, 3, 9, 0, 0),
        (3, 9, 3, 9, 3, 0),
        (3, 9, 3, 9, 0, 9),
        (6, 6, 6, 6, 0, 0),
        (6, 6, 6, 6, 6, 0),
        (6, 6, 6, 6, 0, 6),
        (2, 27, 2, 27, 0, 0),
        (2, 27, 2, 27, 2, 0),
        (9, 6, 9, 6, 0, 0),
        (9, 6, 9, 6, 9, 0),
    ]
    
    for config in double_grid_configs:
        if time.time() > deadline:
            break
        count, points, lines = _try_double_grid(*config)
        if count > best_count and points is not None:
            best_count = count
            best_points = points
            best_lines = lines
    
    if time.time() > deadline:
        return np.array(best_points, dtype=np.int64), np.array(best_lines, dtype=np.int64)
    
    # Final: ensure we have valid output
    if best_points is None:
        # Fallback to basic grid
        xs = list(range(3))
        ys = list(range(18))
        best_points = [(x, y) for x in xs for y in ys]
        best_count, best_points, best_lines = _search_pair_lines(best_points)
    
    return np.array(best_points, dtype=np.int64), np.array(best_lines, dtype=np.int64)
