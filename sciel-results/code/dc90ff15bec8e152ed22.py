# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product
from math import gcd
import sys

def _normalize_line(A, B, C):
    """Normalize line coefficients so the first nonzero is positive and gcd divides all."""
    if A == 0 and B == 0 and C == 0:
        return None
    # Find gcd of absolute values
    g = 0
    for v in [abs(A), abs(B), abs(C)]:
        g = gcd(g, v)
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    # Make first nonzero positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_from_points(p1, p2):
    """Compute line through two points, return normalized (A, B, C) or None if same point."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    if A == 0 and B == 0:
        return None
    return _normalize_line(A, B, C)

def _count_incidences(points, lines):
    """Count incidences: number of (point, line) pairs where A*x + B*y + C == 0."""
    n = len(points)
    m = len(lines)
    count = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count

def _validate(points, lines):
    """Validate the solution meets all constraints."""
    if len(points) != 54 or len(lines) != 27:
        return False
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    # Check for NaN, inf, non-integer values
    for arr in [points, lines]:
        for val in arr.flatten():
            if not np.isfinite(val):
                return False
            if abs(val - round(val)) > 1e-6:
                return False
    # Check for repeated points
    point_set = set()
    for p in points:
        key = (int(round(p[0])), int(round(p[1])))
        if key in point_set:
            return False
        point_set.add(key)
    # Check lines: A=B=0 invalid, duplicate lines
    line_set = set()
    for l in lines:
        A, B, C = int(round(l[0])), int(round(l[1])), int(round(l[2]))
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm is None:
            return False
        if norm in line_set:
            return False
        line_set.add(norm)
    return True

def _generate_grid_points(k):
    """Generate the Szemeredi-Trotter grid points."""
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _generate_grid_lines(k):
    """Generate the Szemeredi-Trotter grid lines."""
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    return np.array(rows, dtype=np.int64)

def _local_search(points, lines, max_iter=5000, time_budget=300):
    """Local search to improve incidence count by swapping points/lines."""
    start_time = time.time()
    best_score = _count_incidences(points, lines)
    current_points = points.copy()
    current_lines = lines.copy()
    
    # Precompute which lines each point lies on, and which points each line passes through
    def compute_incidence_matrix(points, lines):
        """Return boolean matrix: inc[i,j] = point i on line j"""
        n = len(points)
        m = len(lines)
        inc = np.zeros((n, m), dtype=bool)
        for i in range(n):
            x, y = points[i]
            for j in range(m):
                A, B, C = lines[j]
                if A * x + B * y + C == 0:
                    inc[i, j] = True
        return inc
    
    inc = compute_incidence_matrix(current_points, current_lines)
    
    # For local search, try replacing a point or line with a better one
    # Generate candidate points and lines from a larger pool
    
    # Candidate point pool: expand the grid
    k = 3
    candidate_xs = set(range(1, k + 1))
    candidate_ys = set(range(1, 2 * k * k + 1))
    # Add some extra candidates
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            for x in candidate_xs:
                for y in candidate_ys:
                    candidate_xs.add(x + dx)
                    candidate_ys.add(y + dy)
    
    # Build candidate point set
    candidate_points = set()
    for x in sorted(candidate_xs):
        for y in sorted(candidate_ys):
            if 1 <= x <= 10 and 1 <= y <= 30:
                candidate_points.add((x, y))
    
    # Candidate line pool: lines y = a*x + b for reasonable a, b
    candidate_lines = set()
    for a in range(-2, 5):
        for b in range(-5, 20):
            if a == 0:
                # Horizontal line y = -b
                if -b in range(1, 30):
                    candidate_lines.add(_normalize_line(0, 1, -(-b)))
            else:
                candidate_lines.add(_normalize_line(a, -1, b))
    # Also add vertical lines x = c
    for c in range(1, 10):
        candidate_lines.add(_normalize_line(1, 0, -c))
    
    # Remove current points/lines from candidates
    current_point_set = set((int(p[0]), int(p[1])) for p in current_points)
    current_line_set = set(_normalize_line(int(l[0]), int(l[1]), int(l[2])) for l in current_lines)
    
    candidate_points -= current_point_set
    candidate_lines -= current_line_set
    
    # Convert to lists for random access
    cand_pts = list(candidate_points)
    cand_lines = list(candidate_lines)
    
    if not cand_pts or not cand_lines:
        return best_score, current_points, current_lines
    
    rng = np.random.default_rng(42)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Randomly choose to swap a point or a line
        if rng.random() < 0.5:
            # Swap a point
            idx = rng.integers(0, len(current_points))
            old_pt = (int(current_points[idx, 0]), int(current_points[idx, 1]))
            
            # Try a few random candidate points
            for _ in range(10):
                new_pt = cand_pts[rng.integers(0, len(cand_pts))]
                if new_pt == old_pt:
                    continue
                
                # Compute change in incidences
                old_inc = inc[idx, :].sum()
                new_inc = 0
                for j in range(len(current_lines)):
                    A, B, C = current_lines[j]
                    if A * new_pt[0] + B * new_pt[1] + C == 0:
                        new_inc += 1
                
                delta = new_inc - old_inc
                if delta > 0:
                    current_points[idx] = new_pt
                    inc[idx, :] = False
                    for j in range(len(current_lines)):
                        A, B, C = current_lines[j]
                        if A * new_pt[0] + B * new_pt[1] + C == 0:
                            inc[idx, j] = True
                    best_score += delta
                    break
        
        else:
            # Swap a line
            idx = rng.integers(0, len(current_lines))
            old_line = _normalize_line(int(current_lines[idx, 0]), int(current_lines[idx, 1]), int(current_lines[idx, 2]))
            
            for _ in range(10):
                new_line = cand_lines[rng.integers(0, len(cand_lines))]
                if new_line == old_line:
                    continue
                
                # Compute change in incidences
                old_inc = inc[:, idx].sum()
                new_inc = 0
                for i in range(len(current_points)):
                    x, y = current_points[i]
                    A, B, C = new_line
                    if A * x + B * y + C == 0:
                        new_inc += 1
                
                delta = new_inc - old_inc
                if delta > 0:
                    current_lines[idx] = new_line
                    inc[:, idx] = False
                    for i in range(len(current_points)):
                        x, y = current_points[i]
                        A, B, C = new_line
                        if A * x + B * y + C == 0:
                            inc[i, idx] = True
                    best_score += delta
                    break
    
    return best_score, current_points, current_lines

def _hill_climb(points, lines, time_budget=200):
    """More systematic hill climbing: try all single replacements."""
    start_time = time.time()
    best_score = _count_incidences(points, lines)
    current_points = points.copy()
    current_lines = lines.copy()
    
    def compute_incidence_matrix(points, lines):
        n = len(points)
        m = len(lines)
        inc = np.zeros((n, m), dtype=bool)
        for i in range(n):
            x, y = points[i]
            for j in range(m):
                A, B, C = lines[j]
                if A * x + B * y + C == 0:
                    inc[i, j] = True
        return inc
    
    inc = compute_incidence_matrix(current_points, current_lines)
    
    # Build candidate pools
    k = 3
    candidate_points = set()
    for x in range(1, 11):
        for y in range(1, 31):
            candidate_points.add((x, y))
    
    candidate_lines = set()
    for a in range(-3, 6):
        for b in range(-10, 25):
            candidate_lines.add(_normalize_line(a, -1, b))
    for c in range(1, 11):
        candidate_lines.add(_normalize_line(1, 0, -c))
    
    current_point_set = set((int(p[0]), int(p[1])) for p in current_points)
    current_line_set = set(_normalize_line(int(l[0]), int(l[1]), int(l[2])) for l in current_lines)
    
    cand_pts = list(candidate_points - current_point_set)
    cand_lines = list(candidate_lines - current_line_set)
    
    improved = True
    while improved and time.time() - start_time < time_budget:
        improved = False
        
        # Try replacing each point
        for idx in range(len(current_points)):
            old_pt = (int(current_points[idx, 0]), int(current_points[idx, 1]))
            old_inc = inc[idx, :].sum()
            
            best_new_pt = None
            best_new_inc = old_inc
            
            for new_pt in cand_pts:
                if new_pt == old_pt:
                    continue
                new_inc = 0
                for j in range(len(current_lines)):
                    A, B, C = current_lines[j]
                    if A * new_pt[0] + B * new_pt[1] + C == 0:
                        new_inc += 1
                
                if new_inc > best_new_inc:
                    best_new_inc = new_inc
                    best_new_pt = new_pt
            
            if best_new_pt is not None:
                delta = best_new_inc - old_inc
                if delta > 0:
                    current_points[idx] = best_new_pt
                    inc[idx, :] = False
                    for j in range(len(current_lines)):
                        A, B, C = current_lines[j]
                        if A * best_new_pt[0] + B * best_new_pt[1] + C == 0:
                            inc[idx, j] = True
                    best_score += delta
                    improved = True
        
        # Try replacing each line
        for idx in range(len(current_lines)):
            old_line = _normalize_line(int(current_lines[idx, 0]), int(current_lines[idx, 1]), int(current_lines[idx, 2]))
            old_inc = inc[:, idx].sum()
            
            best_new_line = None
            best_new_inc = old_inc
            
            for new_line in cand_lines:
                if new_line == old_line:
                    continue
                new_inc = 0
                for i in range(len(current_points)):
                    x, y = current_points[i]
                    A, B, C = new_line
                    if A * x + B * y + C == 0:
                        new_inc += 1
                
                if new_inc > best_new_inc:
                    best_new_inc = new_inc
                    best_new_line = new_line
            
            if best_new_line is not None:
                delta = best_new_inc - old_inc
                if delta > 0:
                    current_lines[idx] = best_new_line
                    inc[:, idx] = False
                    for i in range(len(current_points)):
                        x, y = current_points[i]
                        A, B, C = best_new_line
                        if A * x + B * y + C == 0:
                            inc[i, idx] = True
                    best_score += delta
                    improved = True
    
    return best_score, current_points, current_lines

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization. Construct a set of n points and a set of
    m lines in the plane (on the integer lattice) so as to maximize the number of
    point-line incidences (pairs (p, l) with p lying on l). Here the instance is fixed
    at n = 54 points and m = 27 lines.
    """
    start_time = time.time()
    deadline = start_time + 620  # Safety margin before 630s kill
    
    # Start with the classical Szemeredi-Trotter grid construction
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    
    best_score = _count_incidences(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Try local search with time budget
    if time.time() < deadline - 100:
        score, pts, lns = _local_search(points, lines, max_iter=10000, time_budget=150)
        if score > best_score and _validate(pts, lns):
            best_score = score
            best_points = pts
            best_lines = lns
    
    # Try hill climbing with remaining time
    remaining = deadline - time.time() - 30
    if remaining > 50:
        score, pts, lns = _hill_climb(best_points, best_lines, time_budget=int(min(remaining, 200)))
        if score > best_score and _validate(pts, lns):
            best_score = score
            best_points = pts
            best_lines = lns
    
    # Final validation
    if not _validate(best_points, best_lines):
        # Fallback to the known valid grid construction
        points = _generate_grid_points(3)
        lines = _generate_grid_lines(3)
        return points, lines
    
    # Ensure integer types
    best_points = best_points.astype(np.int64)
    best_lines = best_lines.astype(np.int64)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
