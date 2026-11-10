import numpy as np
import time
import random
import math
from itertools import combinations
from collections import defaultdict

def _canon_line(A, B, C):
    """Canonicalize a line (A, B, C) -> (A, B, C) such that gcd(|A|,|B|,|C|)=1 and first non-zero is positive."""
    if A == 0 and B == 0:
        return None
    g = math.gcd(abs(A), math.gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count incidences exactly."""
    I = 0
    for x, y in points:
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                I += 1
    return I

def _build_candidate_lines(points):
    """Build all candidate lines determined by pairs of points."""
    candidates = {}
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            A = dy
            B = -dx
            C = dx*yi - dy*xi
            cl = _canon_line(A, B, C)
            if cl is not None:
                candidates[cl] = set()
    # Also add horizontal and vertical lines through each point
    for i in range(n):
        xi, yi = points[i]
        # Horizontal line y = yi: 0*x + 1*y + (-yi) = 0
        cl = _canon_line(0, 1, -yi)
        if cl is not None:
            if cl not in candidates:
                candidates[cl] = set()
        # Vertical line x = xi: 1*x + 0*y + (-xi) = 0
        cl = _canon_line(1, 0, -xi)
        if cl is not None:
            if cl not in candidates:
                candidates[cl] = set()
    # Compute incidence sets
    for cl in list(candidates.keys()):
        A, B, C = cl
        for i in range(n):
            xi, yi = points[i]
            if A*xi + B*yi + C == 0:
                candidates[cl].add(i)
    return candidates

def _greedy_select_lines(points, candidates, num_lines=27):
    """Greedy select top lines by incidence count."""
    # Sort by size descending
    sorted_lines = sorted(candidates.items(), key=lambda x: len(x[1]), reverse=True)
    selected = []
    for cl, pts_on_line in sorted_lines[:num_lines]:
        selected.append(cl)
    return selected

def _evaluate_config(points, lines):
    """Evaluate total incidences."""
    I = 0
    for x, y in points:
        for A, B, C in lines:
            if A*x + B*y + C == 0:
                I += 1
    return I

def _local_search(points, lines, candidates, max_iters=5000, deadline=None):
    """Local search that can swap points or lines."""
    points = list(points)
    lines = list(lines)
    n_pts = len(points)
    n_lines = len(lines)
    
    # Build incidence matrix
    M = np.zeros((n_pts, n_lines), dtype=np.int8)
    for i in range(n_pts):
        xi, yi = points[i]
        for j in range(n_lines):
            A, B, C = lines[j]
            if A*xi + B*yi + C == 0:
                M[i, j] = 1
    current_I = int(np.sum(M))
    best_I = current_I
    best_points = [tuple(p) for p in points]
    best_lines = [tuple(l) for l in lines]
    
    # Precompute point set for duplicate checking
    point_set = set(points)
    line_set = set(lines)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        if random.random() < 0.5:
            # Try swapping a point
            i = random.randint(0, n_pts-1)
            old_x, old_y = points[i]
            
            # Try nearby positions
            found = False
            for _ in range(10):
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = old_x + dx, old_y + dy
                new_pt = (new_x, new_y)
                if new_pt in point_set:
                    continue
                
                # Calculate delta
                delta = 0
                for j in range(n_lines):
                    A, B, C = lines[j]
                    old_inc = M[i, j]
                    new_inc = 1 if (A*new_x + B*new_y + C == 0) else 0
                    delta += new_inc - old_inc
                
                if delta > 0:
                    # Accept
                    point_set.remove((old_x, old_y))
                    points[i] = new_pt
                    point_set.add(new_pt)
                    for j in range(n_lines):
                        A, B, C = lines[j]
                        M[i, j] = 1 if (A*new_x + B*new_y + C == 0) else 0
                    current_I += delta
                    if current_I > best_I:
                        best_I = current_I
                        best_points = [tuple(p) for p in points]
                        best_lines = [tuple(l) for l in lines]
                    found = True
                    break
            if not found:
                continue
        else:
            # Try swapping a line
            j = random.randint(0, n_lines-1)
            old_line = lines[j]
            
            # Find alternative lines from candidates
            if not candidates:
                continue
            
            # Sample some candidate lines
            cand_keys = list(candidates.keys())
            found = False
            for _ in range(5):
                if not cand_keys:
                    break
                new_line = random.choice(cand_keys)
                if new_line == old_line or new_line in line_set:
                    continue
                
                # Calculate delta
                delta = 0
                A, B, C = new_line
                for i in range(n_pts):
                    xi, yi = points[i]
                    old_inc = M[i, j]
                    new_inc = 1 if (A*xi + B*yi + C == 0) else 0
                    delta += new_inc - old_inc
                
                if delta > 0:
                    # Accept
                    line_set.remove(old_line)
                    lines[j] = new_line
                    line_set.add(new_line)
                    for i in range(n_pts):
                        xi, yi = points[i]
                        M[i, j] = 1 if (A*xi + B*yi + C == 0) else 0
                    current_I += delta
                    if current_I > best_I:
                        best_I = current_I
                        best_points = [tuple(p) for p in points]
                        best_lines = [tuple(l) for l in lines]
                    found = True
                    break
            if not found:
                continue
    
    return best_points, best_lines, best_I

def _seed_grid():
    """The baseline Szemeredi-Trotter grid."""
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    points = [tuple(p) for p in points]
    
    lines = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            lines.append(_canon_line(a, -1, b))
    return points, lines

def _seed_dense_cluster():
    """Create a dense cluster with high line multiplicity."""
    points = []
    # Create a 6x9 grid centered at origin
    for x in range(-3, 3):
        for y in range(-4, 5):
            points.append((x, y))
    
    # Generate lines through many points
    lines = set()
    # Horizontal lines
    for y in range(-4, 5):
        lines.add(_canon_line(0, 1, -y))
    # Vertical lines  
    for x in range(-3, 3):
        lines.add(_canon_line(1, 0, -x))
    # Diagonal lines
    for d in range(-7, 8):
        lines.add(_canon_line(1, -1, -d))
    
    # Select top 27 by incidence
    candidates = {}
    for cl in lines:
        A, B, C = cl
        count = sum(1 for (x, y) in points if A*x + B*y + C == 0)
        candidates[cl] = count
    
    sorted_lines = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    selected_lines = [cl for cl, _ in sorted_lines[:27]]
    
    return points, selected_lines

def _seed_lattice():
    """Create a lattice structure."""
    points = []
    # 9x6 lattice
    for x in range(9):
        for y in range(6):
            points.append((x, y))
    
    # Generate candidate lines
    candidates = {}
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            A = dy
            B = -dx
            C = dx*yi - dy*xi
            cl = _canon_line(A, B, C)
            if cl is not None:
                if cl not in candidates:
                    candidates[cl] = 0
                candidates[cl] += 1
    
    # Also add axis-aligned
    for x in range(9):
        cl = _canon_line(1, 0, -x)
        if cl:
            if cl not in candidates:
                candidates[cl] = 0
            candidates[cl] = max(candidates.get(cl, 0), 6)
    for y in range(6):
        cl = _canon_line(0, 1, -y)
        if cl:
            if cl not in candidates:
                candidates[cl] = 0
            candidates[cl] = max(candidates.get(cl, 0), 9)
    
    sorted_lines = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    selected_lines = [cl for cl, _ in sorted_lines[:27]]
    
    return points, selected_lines

def _dual_space_optimize(points, deadline):
    """Use dual-space hypergraph approach to optimize."""
    # Build candidate lines from current points
    candidates = _build_candidate_lines(points)
    
    # Greedy select top 27 lines
    lines = _greedy_select_lines(points, candidates, 27)
    
    # Evaluate
    I = _evaluate_config(points, lines)
    
    # Local search
    points, lines, I = _local_search(points, lines, candidates, max_iters=3000, deadline=deadline)
    
    return points, lines, I

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    """
    start_time = time.time()
    deadline = start_time + 590  # 40s safety margin
    
    best_I = 0
    best_points = None
    best_lines = None
    
    # Seeds
    seeds = [
        ("grid", _seed_grid()),
        ("cluster", _seed_dense_cluster()),
        ("lattice", _seed_lattice()),
    ]
    
    # Run dual-space optimization on each seed
    for name, (pts, lns) in seeds:
        if time.time() > deadline - 60:
            break
        
        res_pts, res_lns, res_I = _dual_space_optimize(pts, deadline - 30)
        
        if res_I > best_I:
            best_I = res_I
            best_points = res_pts
            best_lines = res_lns
    
    # If time remains, do additional local search on best solution
    if best_points is not None and time.time() < deadline - 10:
        # Rebuild candidates for best solution
        candidates = _build_candidate_lines(best_points)
        res_pts, res_lns, res_I = _local_search(best_points, best_lines, candidates, 
                                                 max_iters=5000, deadline=deadline - 5)
        if res_I > best_I:
            best_I = res_I
            best_points = res_pts
            best_lines = res_lns

    # Convert to numpy arrays
    points_arr = np.array(best_points, dtype=np.int64)
    lines_arr = np.array(best_lines, dtype=np.int64)
    
    return points_arr, lines_arr

# EVOLVE-BLOCK-START
# The code above is the complete program.
# EVOLVE-BLOCK-END
