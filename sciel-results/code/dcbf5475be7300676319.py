# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _gcd3(a, b, c):
    """Compute gcd of three integers."""
    g = math.gcd(abs(int(a)), abs(int(b)))
    g = math.gcd(g, abs(int(c)))
    return g if g != 0 else 1


def _normalize_line(A, B, C):
    """Normalize a line so it's unique."""
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    if g != 0:
        A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _find_best_lines_for_points(points, K, deadline):
    """
    Find the 27 lines with highest incidence for given points.
    K is max absolute value for A and B coefficients.
    Returns (lines_array, total_incidence) or (None, 0).
    """
    n_pts = len(points)
    px = points[:, 0].astype(np.int64)
    py = points[:, 1].astype(np.int64)
    
    candidates = []  # List of (count, normalized_line_tuple)
    seen = set()
    
    for A in range(-K, K + 1):
        if time.time() > deadline:
            break
        for B in range(-K, K + 1):
            if time.time() > deadline:
                break
            if A == 0 and B == 0:
                continue
            if math.gcd(abs(A), abs(B)) != 1:
                continue
            
            # Compute V = A*x + B*y for all points
            V = A * px + B * py
            
            # Find unique values and their counts
            unique_vals, counts = np.unique(V, return_counts=True)
            
            for val, count in zip(unique_vals, counts):
                if count == 0:
                    continue
                C = -int(val)
                line_norm = _normalize_line(A, B, C)
                if line_norm not in seen:
                    seen.add(line_norm)
                    candidates.append((count, line_norm))
    
    if not candidates:
        return None, 0
    
    # Sort by count descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 27 unique lines
    selected_lines = []
    seen_sel = set()
    
    for count, line_norm in candidates:
        if len(selected_lines) >= 27:
            break
        if line_norm not in seen_sel:
            seen_sel.add(line_norm)
            selected_lines.append(line_norm)
    
    if len(selected_lines) < 27:
        # Fallback: fill with any valid distinct lines
        for A in range(-30, 31):
            for B in range(-30, 31):
                if A == 0 and B == 0:
                    continue
                if math.gcd(abs(A), abs(B)) != 1:
                    continue
                for C in range(-200, 201):
                    norm = _normalize_line(A, B, C)
                    if norm not in seen_sel:
                        seen_sel.add(norm)
                        selected_lines.append(norm)
                        if len(selected_lines) >= 27:
                            break
                if len(selected_lines) >= 27:
                    break
            if len(selected_lines) >= 27:
                break
    
    if len(selected_lines) < 27:
        return None, 0
    
    lines_array = np.array(selected_lines, dtype=np.int64)
    
    # Calculate total incidence
    total_incidence = 0
    for line in selected_lines:
        A, B, C = line
        val = A * px + B * py + C
        total_incidence += int(np.sum(val == 0))
    
    return lines_array, total_incidence


def _generate_grid_points(w, h, n=54, offset_x=0, offset_y=0):
    """
    Generate a w x h grid of points, taking first n if w*h > n.
    Points are (x, y) where x in [offset_x, offset_x+w) and y in [offset_y, offset_y+h).
    """
    xs = np.arange(offset_x, offset_x + w, dtype=np.int64)
    ys = np.arange(offset_y, offset_y + h, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    if len(points) > n:
        points = points[:n]
    
    return points


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for n=54, m=27.
    Searches over multiple grid configurations and finds optimal lines.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s limit
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Candidate grid configurations (w, h) with w*h >= 54
    # Include various aspect ratios to explore different incidence structures
    grid_configs = [
        (6, 9), (9, 6),
        (7, 8), (8, 7),
        (5, 11), (11, 5),
        (4, 14), (14, 4),
        (3, 18), (18, 3),
        (2, 27), (27, 2),
        (5, 12), (12, 5),  # 60 points, take 54
        (6, 10), (10, 6),  # 60 points, take 54
        (7, 9), (9, 7),    # 63 points, take 54
        (6, 11), (11, 6),  # 66 points, take 54
    ]
    
    K = 40  # Max slope component magnitude
    
    for w, h in grid_configs:
        if time.time() > deadline:
            break
        
        # Try grid with different offsets to break symmetry
        for ox in [0, -w // 2]:
            for oy in [0, -h // 2]:
                if time.time() > deadline:
                    break
                
                points = _generate_grid_points(w, h, 54, ox, oy)
                
                # Verify we have exactly 54 distinct points
                if len(points) != 54:
                    continue
                # Check for duplicates
                unique_pts = np.unique(points, axis=0)
                if len(unique_pts) != 54:
                    continue
                
                lines, score = _find_best_lines_for_points(points, K, deadline)
                
                if lines is not None and score > best_score:
                    best_score = score
                    best_points = points.copy()
                    best_lines = lines.copy()
    
    # If no grid worked, use fallback 6x9 grid with hardcoded lines
    if best_points is None:
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        lines = []
        for i in range(1, 7):
            lines.append((1, 0, -i))
        for j in range(1, 10):
            lines.append((0, 1, -j))
        c_vals = [0, 1, 2, 3, -1, 4, -2, 5, -3, 6, -4, 7]
        for c in c_vals:
            lines.append((1, -1, c))
        best_lines = np.array(lines, dtype=np.int64)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
