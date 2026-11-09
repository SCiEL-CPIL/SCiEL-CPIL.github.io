import numpy as np
import time
import random
from math import gcd
from itertools import product

def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_key(A, B, C):
    return _normalize_line(A, B, C)

def _count_incidentences(points, lines):
    if len(points) == 0 or len(lines) == 0:
        return 0
    val = (lines[:, 0:1] * points[:, 0:1].T + 
           lines[:, 1:2] * points[:, 1:2].T + 
           lines[:, 2:3])
    return int(np.sum(val == 0))

def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    pt_set = set()
    for i in range(54):
        x, y = points[i]
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
            return False
        xi, yi = int(round(x)), int(round(y))
        if (xi, yi) in pt_set:
            return False
        pt_set.add((xi, yi))
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        if not np.isfinite(A) or not np.isfinite(B) or not np.isfinite(C):
            return False
        if abs(A - round(A)) > 1e-6 or abs(B - round(B)) > 1e-6 or abs(C - round(C)) > 1e-6:
            return False
        Ai, Bi, Ci = int(round(A)), int(round(B)), int(round(C))
        if Ai == 0 and Bi == 0:
            return False
        key = _line_key(Ai, Bi, Ci)
        if key in line_set:
            return False
        line_set.add(key)
    return True

def _best_lines_for_points(points, num_lines=27):
    """Find the top num_lines lines by incidence count for given points."""
    n = len(points)
    if n < 2:
        return np.zeros((num_lines, 3), dtype=np.int64)
    
    # Generate candidate lines from all pairs of points
    # We only need to consider lines that pass through at least 2 points.
    # Any line with 1 point has count 1, which is likely lower than grid lines.
    # But we should include them if we don't have enough high-count lines.
    # However, for n=54, there are 54*53/2 = 1431 pairs.
    # Many pairs define the same line.
    
    candidates = {}
    pts = points
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _line_key(A, B, C)
            if key[0] == 0 and key[1] == 0:
                continue
            if key not in candidates:
                candidates[key] = 0
                
    # Count incidences for each unique candidate line
    line_list = list(candidates.keys())
    if not line_list:
        return np.zeros((num_lines, 3), dtype=np.int64)
    
    lines_arr = np.array(line_list, dtype=np.int64)
    # Vectorized count: (L, 1, 1) * (1, N, 1) + ...
    val = (lines_arr[:, 0:1] * pts[:, 0:1].T + 
           lines_arr[:, 1:2] * pts[:, 1:2].T + 
           lines_arr[:, 2:3])
    counts = np.sum(val == 0, axis=1)
    
    # Sort by count descending
    order = np.argsort(-counts)
    top = order[:num_lines]
    result = lines_arr[top]
    return result

def _generate_grid_points(rows, cols, total_points=54):
    """Generate a grid of points, taking the first 'total_points' in row-major order."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            pts.append((r, c))
            if len(pts) == total_points:
                return np.array(pts, dtype=np.int64)
    # If grid is smaller than total_points, pad? No, shape must be 54.
    # So we must ensure rows*cols >= 54.
    return np.array(pts, dtype=np.int64)

def _generate_sheared_grid(rows, cols, shear, total_points=54):
    """Generate a sheared grid: y = r, x = c + shear*r."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = c + shear * r
            y = r
            pts.append((x, y))
            if len(pts) == total_points:
                return np.array(pts, dtype=np.int64)
    return np.array(pts, dtype=np.int64)

def _generate_perturbed_grid(rows, cols, total_points=54, seed=0):
    """Generate a grid with small random perturbations to some points."""
    rng = np.random.RandomState(seed)
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = c
            y = r
            # Perturb 10% of points
            if rng.random() < 0.1:
                x += rng.randint(-1, 2)
                y += rng.randint(-1, 2)
            pts.append((x, y))
            if len(pts) == total_points:
                return np.array(pts, dtype=np.int64)
    return np.array(pts, dtype=np.int64)

def szemeredi_trotter_incidences():
    deadline = time.time() + 600
    safety_margin = 30
    
    best_points = None
    best_lines = None
    best_count = 0
    
    # Strategy: Iterate through various grid shapes and shears
    # The optimal configuration for ST incidences is often a subgrid of a larger lattice.
    # We test many shapes.
    
    tested_configs = set()
    
    # 1. Standard Grids
    # rows * cols >= 54
    for rows in range(1, 60):
        if time.time() > deadline - safety_margin:
            break
        for cols in range(1, 60):
            if rows * cols < 54:
                continue
            if (rows, cols) in tested_configs:
                continue
            tested_configs.add((rows, cols))
            
            pts = _generate_grid_points(rows, cols)
            if pts.shape != (54, 2):
                continue
            # Check for duplicates in points (grid ensures unique unless shear makes them collide? No, standard grid is unique)
            # But we must ensure no duplicate points in the 54 selected.
            pt_tuples = set(map(tuple, pts))
            if len(pt_tuples) != 54:
                continue
                
            lines = _best_lines_for_points(pts, 27)
            count = _count_incidentences(pts, lines)
            
            if count > best_count:
                best_count = count
                best_points = pts.copy()
                best_lines = lines.copy()
                
    # 2. Sheared Grids
    for rows in range(3, 30):
        if time.time() > deadline - safety_margin:
            break
        for cols in range(3, 30):
            if rows * cols < 54:
                continue
            for shear in range(-2, 3):
                if (rows, cols, shear) in tested_configs:
                    continue
                tested_configs.add((rows, cols, shear))
                
                pts = _generate_sheared_grid(rows, cols, shear)
                if pts.shape != (54, 2):
                    continue
                pt_tuples = set(map(tuple, pts))
                if len(pt_tuples) != 54:
                    continue
                    
                lines = _best_lines_for_points(pts, 27)
                count = _count_incidentences(pts, lines)
                
                if count > best_count:
                    best_count = count
                    best_points = pts.copy()
                    best_lines = lines.copy()
                    
    # 3. Perturbed Grids (Random restarts)
    # Use a few seeds to explore the neighborhood of good grids
    # We pick the best grid shape found so far and perturb it
    if best_points is not None:
        for seed in range(100):
            if time.time() > deadline - safety_margin:
                break
            
            # Try perturbing the best found points
            rng = np.random.RandomState(seed)
            cand_pts = best_points.copy()
            # Perturb 5-10 points
            n_perturb = rng.randint(3, 8)
            indices = rng.choice(54, n_perturb, replace=False)
            for idx in indices:
                dx = rng.randint(-2, 2)
                dy = rng.randint(-2, 2)
                cand_pts[idx, 0] += dx
                cand_pts[idx, 1] += dy
                
            # Check duplicates
            pt_tuples = set(map(tuple, cand_pts))
            if len(pt_tuples) != 54:
                continue
                
            lines = _best_lines_for_points(cand_pts, 27)
            count = _count_incidentences(cand_pts, lines)
            
            if count > best_count:
                best_count = count
                best_points = cand_pts.copy()
                best_lines = lines.copy()
                
    # 4. Random Lattices
    # Generate random points in a box, then snap to integer grid?
    # Or just random integer points in a box.
    for trial in range(50):
        if time.time() > deadline - safety_margin:
            break
        rng = np.random.RandomState(trial + 1000)
        box_size = rng.randint(10, 20)
        pts = rng.randint(-box_size, box_size, size=(54, 2))
        
        # Ensure uniqueness
        pt_tuples = set(map(tuple, pts))
        if len(pt_tuples) != 54:
            continue
            
        lines = _best_lines_for_points(pts, 27)
        count = _count_incidentences(pts, lines)
        
        if count > best_count:
            best_count = count
            best_points = pts.copy()
            best_lines = lines.copy()

    if best_points is None:
        # Fallback to a simple grid
        best_points = _generate_grid_points(9, 6)
        best_lines = _best_lines_for_points(best_points, 27)
        
    # Final validation
    if not _is_valid(best_points, best_lines):
        # Fallback
        pts = _generate_grid_points(9, 6)
        lines = _best_lines_for_points(pts, 27)
        return pts, lines
        
    return best_points, best_lines
