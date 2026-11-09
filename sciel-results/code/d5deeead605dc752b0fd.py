# EVOLVE-BLOCK-START
import numpy as np
import time

def _gcd3(a, b, c):
    """Compute gcd of three non-negative integers."""
    from math import gcd
    return gcd(gcd(abs(a), abs(b)), abs(c))

def _normalize_line(A, B, C):
    """Normalize a line (A,B,C) so that gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    g = _gcd3(A, B, C)
    if g > 0:
        A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidents(points, lines):
    """Count point-line incidences exactly using integer arithmetic."""
    # points: (n, 2) int64 array
    # lines: (m, 3) int64 array
    # For each line (A, B, C), count points where A*x + B*y + C == 0
    count = 0
    for i in range(lines.shape[0]):
        A, B, C = int(lines[i, 0]), int(lines[i, 1]), int(lines[i, 2])
        if A == 0 and B == 0:
            continue
        # Compute A*x + B*y + C for all points
        vals = A * points[:, 0] + B * points[:, 1] + C
        count += int(np.sum(vals == 0))
    return count

def _validate(points, lines):
    """Check if points and lines satisfy all constraints. Return True if valid."""
    # Check shapes
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    
    # Check for NaN/inf
    if not np.all(np.isfinite(points.astype(np.float64))):
        return False
    if not np.all(np.isfinite(lines.astype(np.float64))):
        return False
    
    # Check entries are close to integers
    if not np.allclose(points, np.round(points), atol=1e-6):
        return False
    if not np.allclose(lines, np.round(lines), atol=1e-6):
        return False
    
    # Round to int64
    points = np.round(points).astype(np.int64)
    lines = np.round(lines).astype(np.int64)
    
    # Check for repeated points
    unique_points = np.unique(points, axis=0)
    if unique_points.shape[0] != 54:
        return False
    
    # Check lines: no A=B=0, no duplicate lines
    seen_lines = set()
    for i in range(lines.shape[0]):
        A, B, C = int(lines[i, 0]), int(lines[i, 1]), int(lines[i, 2])
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm in seen_lines:
            return False
        seen_lines.add(norm)
    
    return True

def _build_grid_and_lines(gx, gy, n_diag=12):
    """Build a gx x gy grid with all vertical/horizontal lines and best diagonal lines."""
    # Points: gx x gy grid
    xs = np.arange(1, gx + 1)
    ys = np.arange(1, gy + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    # Vertical lines: x = i -> (1, 0, -i)
    for i in range(1, gx + 1):
        lines.append((1, 0, -i))
    
    # Horizontal lines: y = j -> (0, 1, -j)
    for j in range(1, gy + 1):
        lines.append((0, 1, -j))
    
    # Diagonal lines: slope 1 (y = x + c) and slope -1 (y = -x + c)
    # For slope 1: line is (1, -1, c), passes through (x, y) if y - x = c
    # For slope -1: line is (1, 1, -c), passes through (x, y) if x + y = c
    
    # Compute all possible slope-1 lines and their incidence counts
    slope1_lines = []
    for c in range(-(gy - 1), gx):  # c = y - x, range is -(gy-1) to gx-1
        # Count points where y - x = c
        count = 0
        for x in range(1, gx + 1):
            y = x + c
            if 1 <= y <= gy:
                count += 1
        if count > 0:
            slope1_lines.append((count, c))
    slope1_lines.sort(reverse=True)
    
    # Compute all possible slope-(-1) lines and their incidence counts
    slope_neg1_lines = []
    for c in range(2, gx + gy + 1):  # c = x + y, range is 2 to gx+gy
        # Count points where x + y = c
        count = 0
        for x in range(1, gx + 1):
            y = c - x
            if 1 <= y <= gy:
                count += 1
        if count > 0:
            slope_neg1_lines.append((count, c))
    slope_neg1_lines.sort(reverse=True)
    
    # Select best diagonal lines (alternating between slope 1 and slope -1 for diversity)
    s1_idx = 0
    sn1_idx = 0
    diag_lines = []
    while len(diag_lines) < n_diag:
        # Add from slope 1 if available
        if s1_idx < len(slope1_lines) and (sn1_idx >= len(slope_neg1_lines) or 
           (s1_idx < len(slope1_lines) and slope1_lines[s1_idx][0] >= slope_neg1_lines[sn1_idx][0])):
            c = slope1_lines[s1_idx][1]
            diag_lines.append((1, -1, c))
            s1_idx += 1
        elif sn1_idx < len(slope_neg1_lines):
            c = slope_neg1_lines[sn1_idx][1]
            diag_lines.append((1, 1, -c))
            sn1_idx += 1
        elif s1_idx < len(slope1_lines):
            c = slope1_lines[s1_idx][1]
            diag_lines.append((1, -1, c))
            s1_idx += 1
        else:
            break
    
    lines.extend(diag_lines)
    
    # Pad or trim to exactly 27 lines
    while len(lines) < 27:
        # Add low-incidence diagonal lines
        all_diag = slope1_lines + slope_neg1_lines
        all_diag.sort(key=lambda x: x[0], reverse=True)
        added = False
        for count, c in all_diag:
            if count <= 0:
                continue
            # Check if already added
            already = False
            for l in lines:
                if l[0] == 1 and l[1] == -1 and l[2] == c:
                    already = True
                    break
                if l[0] == 1 and l[1] == 1 and l[2] == -c:
                    already = True
                    break
            if not already:
                # Determine slope
                # This is tricky; let's just add slope 1 lines
                lines.append((1, -1, c))
                added = True
                break
        if not added:
            # Fallback: add a trivial line that won't match many points
            lines.append((1, 2, 1))
    
    lines = lines[:27]
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines

def _try_configuration(gx, gy, n_diag):
    """Try a specific configuration and return (incidences, points, lines) or None if invalid."""
    total_lines = gx + gy + n_diag
    if total_lines != 27:
        return None
    
    points, lines = _build_grid_and_lines(gx, gy, n_diag)
    
    if not _validate(points, lines):
        return None
    
    incidents = _count_incidents(points, lines)
    return incidents, points, lines

def szemeredi_trotter_incidences():
    """
    Construct a set of 54 points and 27 lines maximizing incidences.
    """
    start_time = time.time()
    deadline = start_time + 620  # Safety margin before 630s limit
    
    # Try various grid configurations
    best_incidents = 0
    best_points = None
    best_lines = None
    
    # Grid configurations: (gx, gy, n_diag) where gx*gy = 54 and gx+gy+n_diag = 27
    # 54 = 1*54, 2*27, 3*18, 6*9, 9*6, 18*3, 27*2, 54*1
    configs = [
        (6, 9, 12),   # 6+9+12=27
        (9, 6, 12),   # 9+6+12=27
        (3, 18, 6),   # 3+18+6=27
        (18, 3, 6),   # 18+3+6=27
        (2, 27, -2),  # invalid, skip
        (27, 2, -2),  # invalid, skip
    ]
    
    # Also try non-standard splits where we don't use all vertical/horizontal
    # But first try the straightforward ones
    
    for gx, gy, n_diag in configs:
        if time.time() > deadline:
            break
        if n_diag < 0:
            continue
        result = _try_configuration(gx, gy, n_diag)
        if result is not None:
            incidents, points, lines = result
            if incidents > best_incidents:
                best_incidents = incidents
                best_points = points.copy()
                best_lines = lines.copy()
    
    # If no valid configuration found, use the known good one
    if best_points is None:
        # Fallback: 6x9 grid with 12 diagonals
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        lines = []
        for i in range(1, 7):
            lines.append((1, 0, -i))
        for j in range(1, 10):
            lines.append((0, 1, -j))
        # Best 12 diagonals
        c_vals_slope_1 = [0, 1, 2, 3, -1, 4]
        for c in c_vals_slope_1:
            lines.append((1, -1, c))
        c_vals_slope_neg_1 = [7, 8, 9, 10, 6, 11]
        for c in c_vals_slope_neg_1:
            lines.append((1, 1, -c))
        best_lines = np.array(lines, dtype=np.int64)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
