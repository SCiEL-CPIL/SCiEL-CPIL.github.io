# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line (A, B, C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _count_incidence_for_line(A, B, C, points):
    """Count points on line A*x + B*y + C = 0."""
    vals = A * points[:, 0] + B * points[:, 1] + C
    return int(np.sum(vals == 0))

def _count_total_incidence(points, lines):
    """Total (point, line) incidences."""
    total = 0
    for A, B, C in lines:
        total += _count_incidence_for_line(A, B, C, points)
    return total

def _all_lines_from_grid(a, b):
    """
    Generate all distinct lines through at least 2 points of an a×b grid 
    with points at (1..a, 1..b). Returns dict line_key -> incidence count.
    """
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    n = points.shape[0]
    
    lines = {}
    
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = dy
            B = -dx
            C = dx * p1[1] - dy * p1[0]
            key = _normalize_line(A, B, C)
            if key is not None:
                lines.setdefault(key, 0)
    
    # Count incidences
    for key in lines:
        A, B, C = key
        lines[key] = _count_incidence_for_line(A, B, C, points)
    
    return points, lines

def _top27(lines_dict):
    sorted_lines = sorted(lines_dict.items(), key=lambda x: (-x[1], x[0]))
    return [l for l, c in sorted_lines[:27]]

def _try_grid(a, b):
    """Try a × b grid (a*b == 54) and return (points, lines, score) or None."""
    if a * b != 54:
        return None
    if a <= 0 or b <= 0:
        return None
    points, lines_dict = _all_lines_from_grid(a, b)
    selected = _top27(lines_dict)
    if len(selected) < 27:
        return None
    score = _count_total_incidence(points, selected)
    return points, np.array(selected, dtype=np.int64), score

def _try_rect_with_extras(a, b, extras):
    """
    Try a × b grid plus extra points.
    extras: list of (x, y) integer points to add.
    Returns (points, lines, score) or None.
    """
    base_x = np.arange(1, a + 1)
    base_y = np.arange(1, b + 1)
    px, py = np.meshgrid(base_x, base_y, indexing='ij')
    base_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    if extras:
        extra_arr = np.array(extras, dtype=np.int64)
        points = np.vstack([base_points, extra_arr])
    else:
        points = base_points
    
    if points.shape[0] != 54:
        return None
    
    # Check for duplicates
    pts_tuple = [tuple(p) for p in points]
    if len(set(pts_tuple)) != 54:
        return None
    
    n = 54
    lines = {}
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = dy
            B = -dx
            C = dx * p1[1] - dy * p1[0]
            key = _normalize_line(A, B, C)
            if key is not None:
                lines.setdefault(key, 0)
    
    for key in lines:
        A, B, C = key
        lines[key] = _count_incidence_for_line(A, B, C, points)
    
    selected = _top27(lines)
    if len(selected) < 27:
        return None
    score = _count_total_incidence(points, selected)
    return points, np.array(selected, dtype=np.int64), score

def _try_lattice_points(max_coord=10):
    """
    Try various 54-point subsets of small integer grids.
    """
    # For now, this is a placeholder for more exotic searches.
    # A simple approach: take 54 points from a slightly larger grid, 
    # e.g., 6x10 grid minus 6 points, or 7x9 minus 9, etc.
    # This is complex to do well without running, so we skip for now.
    pass

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines on the integer lattice.
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Candidate grids where a*b = 54
    # 54 = 2 * 27 = 3 * 18 = 6 * 9
    # Try both orientations
    grid_candidates = [
        (6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2), (1, 54), (54, 1)
    ]
    
    for a, b in grid_candidates:
        if time.time() > deadline - 60:
            break
        result = _try_grid(a, b)
        if result is not None:
            pts, lns, score = result
            if score > best_score:
                best_score = score
                best_points = pts
                best_lines = lns
    
    # Try grids with extras: e.g., 5x10 = 50, add 4 points
    # 5x10: x=1..5, y=1..10. Add (1,11),(2,11),(3,11),(4,11) to make a trapezoid?
    # Or add (6,1),(6,2),(6,3),(6,4) to make 6x10 minus 6?
    # Let's try 5x10 with 4 extras at (6,1)..(6,4) -> actually that's 6x10 minus 6.
    # Better: 5x10 = 50 points, add (1,11),(2,11),(3,11),(4,11).
    extras_5_10 = [(1, 11), (2, 11), (3, 11), (4, 11)]
    if time.time() < deadline - 60:
        result = _try_rect_with_extras(5, 10, extras_5_10)
        if result is not None:
            pts, lns, score = result
            if score > best_score:
                best_score = score
                best_points = pts
                best_lines = lns
    
    # 4x12 = 48, add 6 points at y=13: (1,13)..(6,13) -> 4x13 minus 8? 
    # Let's try 4x12 with 6 extras at (5,1)..(5,6)
    extras_4_12 = [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)]
    if time.time() < deadline - 60:
        result = _try_rect_with_extras(4, 12, extras_4_12)
        if result is not None:
            pts, lns, score = result
            if score > best_score:
                best_score = score
                best_points = pts
                best_lines = lns
    
    # 7x8 = 56, remove 2 points. Try removing (7,8) and (1,1)
    # Base 7x8 = 56 points. Remove (7,8) and (1,1).
    if time.time() < deadline - 60:
        base_x = np.arange(1, 8)
        base_y = np.arange(1, 9)
        px, py = np.meshgrid(base_x, base_y, indexing='ij')
        all_pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        # Remove (1,1) and (7,8)
        mask = ~((all_pts[:, 0] == 1) & (all_pts[:, 1] == 1)) & ~((all_pts[:, 0] == 7) & (all_pts[:, 1] == 8))
        pts78 = all_pts[mask]
        if pts78.shape[0] == 54:
            n = 54
            lines = {}
            for i in range(n):
                p1 = pts78[i]
                for j in range(i + 1, n):
                    p2 = pts78[j]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    A = dy
                    B = -dx
                    C = dx * p1[1] - dy * p1[0]
                    key = _normalize_line(A, B, C)
                    if key is not None:
                        lines.setdefault(key, 0)
            for key in lines:
                A, B, C = key
                lines[key] = _count_incidence_for_line(A, B, C, pts78)
            selected = _top27(lines)
            if len(selected) >= 27:
                score = _count_total_incidence(pts78, selected)
                if score > best_score:
                    best_score = score
                    best_points = pts78
                    best_lines = np.array(selected, dtype=np.int64)
    
    # 8x7 = 56, remove 2 points
    if time.time() < deadline - 60:
        base_x = np.arange(1, 9)
        base_y = np.arange(1, 8)
        px, py = np.meshgrid(base_x, base_y, indexing='ij')
        all_pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        mask = ~((all_pts[:, 0] == 1) & (all_pts[:, 1] == 1)) & ~((all_pts[:, 0] == 8) & (all_pts[:, 1] == 7))
        pts87 = all_pts[mask]
        if pts87.shape[0] == 54:
            n = 54
            lines = {}
            for i in range(n):
                p1 = pts87[i]
                for j in range(i + 1, n):
                    p2 = pts87[j]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    A = dy
                    B = -dx
                    C = dx * p1[1] - dy * p1[0]
                    key = _normalize_line(A, B, C)
                    if key is not None:
                        lines.setdefault(key, 0)
            for key in lines:
                A, B, C = key
                lines[key] = _count_incidence_for_line(A, B, C, pts87)
            selected = _top27(lines)
            if len(selected) >= 27:
                score = _count_total_incidence(pts87, selected)
                if score > best_score:
                    best_score = score
                    best_points = pts87
                    best_lines = np.array(selected, dtype=np.int64)
    
    # Fallback
    if best_points is None:
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing='ij')
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        best_lines = np.array([(1, 0, -x) for x in range(1, 7)] + [(0, 1, -y) for y in range(1, 10)] + [(1, -1, 0)]*12, dtype=np.int64)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
