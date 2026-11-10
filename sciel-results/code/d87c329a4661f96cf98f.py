# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
import time

def _normalize_line(A, B, C):
    """Normalize line coefficients so gcd=1 and first nonzero is positive."""
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        B, C = -B, -C
    return (A, B, C)

def _count_incidences(points, line):
    """Count how many points lie on the given line."""
    A, B, C = line
    return np.sum(A * points[:, 0] + B * points[:, 1] + C == 0)

def _build_finite_field_points(p, n_points=54):
    """
    Build points from a finite field F_p structure.
    Use a subset of the affine plane over F_p, mapped to integers.
    """
    # We'll use a structured grid inspired by finite field lines.
    # In F_p, lines have p points. We want subsets where many lines pass through our points.
    # Strategy: use a small grid and add structured perturbations.
    
    # Start with a base grid and try to optimize
    # Try several grid configurations
    configs = []
    
    # 6x9 grid
    for w in range(4, 10):
        h = n_points // w
        if w * h >= n_points:
            configs.append((w, h))
    
    best_points = None
    best_score = -1
    
    for w, h in configs:
        # Create w x h grid
        xs = np.arange(w)
        ys = np.arange(h)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1)
        
        # If we have more than n_points, remove some strategically
        if len(pts) > n_points:
            # Remove points that are on the fewest lines (corners tend to be on fewer)
            # Actually, keep points that create more line incidences
            # For now, just take the first n_points
            pts = pts[:n_points]
        elif len(pts) < n_points:
            # Need to add more - expand the grid
            # This shouldn't happen with our config selection
            continue
        
        # Count incidences for this configuration
        lines_dict = {}
        n_pts = len(pts)
        for i in range(n_pts):
            x1, y1 = pts[i]
            for j in range(i + 1, n_pts):
                x2, y2 = pts[j]
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                key = _normalize_line(A, B, C)
                if key not in lines_dict:
                    lines_dict[key] = 0
        
        for key in lines_dict:
            lines_dict[key] = _count_incidences(pts, key)
        
        sorted_lines = sorted(lines_dict.items(), key=lambda item: item[1], reverse=True)
        # Sum of top 27 line incidences
        score = sum(count for _, count in sorted_lines[:27])
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    return best_points, best_score

def _local_search_points(points, n_lines=27, max_iter=500, deadline=None):
    """
    Local search: try moving individual points to improve incidence count.
    """
    points = points.copy()
    n_pts = len(points)
    
    def _score(points):
        lines_dict = {}
        for i in range(n_pts):
            x1, y1 = points[i]
            for j in range(i + 1, n_pts):
                x2, y2 = points[j]
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                key = _normalize_line(A, B, C)
                if key not in lines_dict:
                    lines_dict[key] = 0
        for key in lines_dict:
            lines_dict[key] = _count_incidences(points, key)
        sorted_lines = sorted(lines_dict.items(), key=lambda item: item[1], reverse=True)
        return sum(count for _, count in sorted_lines[:n_lines])
    
    current_score = _score(points)
    
    for iteration in range(max_iter):
        if deadline and time.time() > deadline - 30:
            break
        
        improved = False
        for i in range(n_pts):
            if deadline and time.time() > deadline - 30:
                break
            
            x0, y0 = points[i]
            # Try small perturbations
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx == 0 and dy == 0:
                        continue
                    points[i] = [x0 + dx, y0 + dy]
                    
                    # Check for duplicate points
                    dup = False
                    for j in range(n_pts):
                        if j != i and points[j][0] == points[i][0] and points[j][1] == points[i][1]:
                            dup = True
                            break
                    if dup:
                        points[i] = [x0, y0]
                        continue
                    
                    new_score = _score(points)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        points[i] = [x0, y0]
        
        if not improved:
            break
    
    return points, current_score

def _select_best_lines(points, n_lines=27):
    """Select the n_lines lines with highest incidence counts."""
    n_pts = len(points)
    lines_dict = {}
    for i in range(n_pts):
        x1, y1 = points[i]
        for j in range(i + 1, n_pts):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            key = _normalize_line(A, B, C)
            if key not in lines_dict:
                lines_dict[key] = 0
    
    for key in lines_dict:
        lines_dict[key] = _count_incidences(points, key)
    
    sorted_lines = sorted(lines_dict.items(), key=lambda item: item[1], reverse=True)
    
    selected = []
    for key, count in sorted_lines:
        if len(selected) == n_lines:
            break
        selected.append(key)
    
    # Pad if needed
    while len(selected) < n_lines:
        selected.append((1, 0, -1000))
    
    return np.array(selected, dtype=np.int64)

def _try_finite_field_construction(p, n_points=54, n_lines=27):
    """
    Try to construct points using finite field structure.
    In F_p, each line has exactly p points. We want to embed a subset
    where many lines pass through our points.
    
    Strategy: Use a rectangular grid in Z^2 but choose dimensions
    that align with prime structure to maximize line incidences.
    """
    # For finite field F_p, lines have p points.
    # If we take a k x m grid, lines of various slopes pass through.
    # Key insight: use grids where many lines have high point counts.
    
    best_points = None
    best_score = -1
    
    # Try various grid configurations
    for w in range(3, 12):
        for h in range(3, 12):
            if w * h != n_points:
                continue
            
            # Create w x h grid
            xs = np.arange(w)
            ys = np.arange(h)
            px, py = np.meshgrid(xs, ys, indexing="ij")
            pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
            
            # Count incidences
            lines_dict = {}
            n_pts = len(pts)
            for i in range(n_pts):
                x1, y1 = pts[i]
                for j in range(i + 1, n_pts):
                    x2, y2 = pts[j]
                    A = y1 - y2
                    B = x2 - x1
                    C = x1 * y2 - x2 * y1
                    key = _normalize_line(A, B, C)
                    if key not in lines_dict:
                        lines_dict[key] = 0
            
            for key in lines_dict:
                lines_dict[key] = _count_incidences(pts, key)
            
            sorted_lines = sorted(lines_dict.items(), key=lambda item: item[1], reverse=True)
            score = sum(count for _, count in sorted_lines[:n_lines])
            
            if score > best_score:
                best_score = score
                best_points = pts.copy()
    
    # Also try non-rectangular configurations
    # e.g., 7x8 with 2 points removed
    for w, h, remove in [(7, 8, [(0,0), (6,7)]), (7, 8, [(0,7), (6,0)]), 
                          (7, 8, [(0,0), (6,0)]), (7, 8, [(0,7), (6,7)])]:
        xs = np.arange(w)
        ys = np.arange(h)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Remove specified points
        for rx, ry in remove:
            mask = (pts[:, 0] == rx) & (pts[:, 1] == ry)
            pts = pts[~mask]
        
        if len(pts) != n_points:
            continue
        
        lines_dict = {}
        n_pts = len(pts)
        for i in range(n_pts):
            x1, y1 = pts[i]
            for j in range(i + 1, n_pts):
                x2, y2 = pts[j]
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                key = _normalize_line(A, B, C)
                if key not in lines_dict:
                    lines_dict[key] = 0
        
        for key in lines_dict:
            lines_dict[key] = _count_incidences(pts, key)
        
        sorted_lines = sorted(lines_dict.items(), key=lambda item: item[1], reverse=True)
        score = sum(count for _, count in sorted_lines[:n_lines])
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    return best_points, best_score

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines on the integer lattice.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill
    
    # Phase 1: Try structured constructions
    points, score = _try_finite_field_construction(59, 54, 27)
    
    if points is None:
        # Fallback to 6x9 grid
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        score = 0
    
    # Phase 2: Local search to refine
    if time.time() < deadline - 60:
        points, score = _local_search_points(points, 27, max_iter=200, deadline=deadline)
    
    # Phase 3: Select best lines
    lines = _select_best_lines(points, 27)
    
    # Ensure valid output
    points = points.astype(np.int64)
    lines = lines.astype(np.int64)
    
    return points, lines

# EVOLVE-BLOCK-END
