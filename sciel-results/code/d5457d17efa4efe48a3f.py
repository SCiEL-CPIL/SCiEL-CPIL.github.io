# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations

N_POINTS = 54
N_LINES = 27
TIME_LIMIT = 600  # seconds

def _normalize_line(A, B, C):
    """Normalize (A, B, C) so that first nonzero is positive, and gcd divides all."""
    if A == 0 and B == 0:
        return None
    # Make first nonzero positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    # Compute gcd
    g = math.gcd(abs(A), math.gcd(abs(B), abs(C)))
    if g > 1:
        A //= g
        B //= g
        C //= g
    return (A, B, C)

def _line_from_points(p1, p2):
    """Given two integer points, return normalized (A, B, C) for the line through them."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return _normalize_line(A, B, C)

def _count_incidence(points, lines):
    """Count number of (point, line) pairs where point lies on line. Exact int64."""
    count = 0
    for A, B, C in lines:
        for x, y in points:
            if A * x + B * y + C == 0:
                count += 1
    return count

def _pick_top_k_lines(points, k=27):
    """For a given set of points, find all lines determined by pairs, count incidences,
    pick top k lines. Returns (lines_array, total_incidence)."""
    n = len(points)
    line_counts = {}
    
    # Generate all lines from pairs of points
    seen_lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            norm = _line_from_points(points[i], points[j])
            if norm is None or norm in seen_lines:
                continue
            seen_lines.add(norm)
            # Count how many of the n points lie on this line
            A, B, C = norm
            cnt = 0
            for x, y in points:
                if A * x + B * y + C == 0:
                    cnt += 1
            line_counts[norm] = cnt
    
    # Also consider that a line might pass through only 1 point (if n==1), but n>=2 always
    # Sort by count descending
    sorted_lines = sorted(line_counts.items(), key=lambda kv: kv[1], reverse=True)
    
    top_k = sorted_lines[:k]
    total = sum(cnt for _, cnt in top_k)
    lines_arr = np.array([ln for ln, _ in top_k], dtype=np.int64)
    return lines_arr, total

def _grid_init():
    """Szemeredi-Trotter grid: k=3, points {1..3}x{1..18}, lines y=a*x+b."""
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
    return points, lines

def _diagonal_grid_init():
    """A different structured initialization: dense grid with diagonal emphasis.
    Use a 6x9 grid (54 points) centered, then pick top 27 lines."""
    # 6 columns, 9 rows = 54 points
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    lines, total = _pick_top_k_lines(points, 27)
    return points, lines

def _random_lattice_init():
    """Random points in a reasonable lattice range."""
    np.random.seed(42)
    points = np.random.randint(0, 12, size=(N_POINTS, 2)).astype(np.int64)
    lines, total = _pick_top_k_lines(points, 27)
    return points, lines

def _sa_optimize(points, lines, deadline, temp_start=10.0, temp_end=0.1):
    """Simulated annealing: perturb points, re-pick top lines, accept based on delta."""
    best_points = points.copy()
    best_lines = lines.copy()
    best_total = _count_incidence(best_points, best_lines)
    current_total = best_total
    
    t = time.time()
    elapsed = 0
    steps = 0
    rng = np.random.RandomState(123)
    
    while elapsed < deadline - 10:  # 10s safety margin
        # Choose a random point to move
        idx = rng.randint(N_POINTS)
        old_pt = best_points[idx].copy()
        
        # Propose a new position: small perturbation
        dx = rng.randint(-2, 3)
        dy = rng.randint(-2, 3)
        if dx == 0 and dy == 0:
            continue
        new_pt = old_pt + np.array([dx, dy], dtype=np.int64)
        
        # Check for duplicate points
        new_points = best_points.copy()
        new_points[idx] = new_pt
        # Check duplicates
        unique_count = len(np.unique(new_points, axis=0))
        if unique_count != N_POINTS:
            continue
        
        # Re-pick top lines for new point set
        new_lines, new_total = _pick_top_k_lines(new_points, 27)
        
        delta = new_total - current_total
        if delta >= 0:
            # Accept
            best_points = new_points
            best_lines = new_lines
            current_total = new_total
            if new_total > best_total:
                best_total = new_total
        else:
            # Accept with probability
            temp = temp_start * (temp_end / temp_start) ** (steps / max(1, 5000))
            prob = math.exp(delta / max(temp, 1e-10))
            if rng.random() < prob:
                best_points = new_points
                best_lines = new_lines
                current_total = new_total
        
        steps += 1
        if steps % 100 == 0:
            elapsed = time.time() - t
    
    return best_points, best_lines, best_total

def _local_search_points(points, deadline):
    """Greedy local search: try moving each point to nearby positions to improve incidence."""
    best_points = points.copy()
    best_total = _count_incidence(best_points, _pick_top_k_lines(best_points, 27)[0])
    
    t = time.time()
    improved = True
    while improved and (time.time() - t) < deadline - 15:
        improved = False
        for idx in range(N_POINTS):
            old_pt = best_points[idx].copy()
            best_move = None
            best_move_total = best_total
            
            # Try nearby positions
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = old_pt + np.array([dx, dy], dtype=np.int64)
                    new_points = best_points.copy()
                    new_points[idx] = new_pt
                    # Check duplicates
                    if len(np.unique(new_points, axis=0)) != N_POINTS:
                        continue
                    _, new_total = _pick_top_k_lines(new_points, 27)
                    if new_total > best_move_total:
                        best_move_total = new_total
                        best_move = new_pt.copy()
            
            if best_move is not None:
                best_points[idx] = best_move
                best_total = best_move_total
                improved = True
    
    lines, total = _pick_top_k_lines(best_points, 27)
    return best_points, lines, total

def _validate_result(points, lines):
    """Validate that points and lines meet all constraints. Returns True if valid."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.all(np.isfinite(points.astype(float))):
        return False
    if not np.all(np.isfinite(lines.astype(float))):
        return False
    # Check entries are within 1e-6 of integers
    if np.any(np.abs(points.astype(float) - np.round(points.astype(float))) > 1e-6):
        return False
    if np.any(np.abs(lines.astype(float) - np.round(lines.astype(float))) > 1e-6):
        return False
    # Check no repeated points
    if len(np.unique(points, axis=0)) != 54:
        return False
    # Check no line with A=B=0
    for A, B, C in lines:
        if A == 0 and B == 0:
            return False
    # Check no duplicate lines (after normalization)
    norm_set = set()
    for A, B, C in lines:
        norm = _normalize_line(int(A), int(B), int(C))
        if norm is None:
            return False
        if norm in norm_set:
            return False
        norm_set.add(norm)
    return True

def _final_validate_and_round(points, lines):
    """Round to int64 and ensure validity."""
    points = np.round(points).astype(np.int64)
    lines = np.round(lines).astype(np.int64)
    # Re-validate
    if not _validate_result(points, lines):
        return None
    return points, lines

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization for 54 points and 27 lines.
    
    Uses a structured grid initialization followed by simulated annealing
    with point perturbation and top-k line selection.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 30  # 30s safety margin
    
    # Phase 1: Generate multiple initial candidates
    candidates = []
    
    # Grid initialization (Szemeredi-Trotter classic)
    pts1, lines1 = _grid_init()
    total1 = _count_incidence(pts1, lines1)
    candidates.append((pts1, lines1, total1))
    
    # Diagonal/dense grid
    pts2, lines2 = _diagonal_grid_init()
    total2 = _count_incidence(pts2, lines2)
    candidates.append((pts2, lines2, total2))
    
    # Random lattice
    pts3, lines3 = _random_lattice_init()
    total3 = _count_incidence(pts3, lines3)
    candidates.append((pts3, lines3, total3))
    
    # Best initial candidate
    best_cand = max(candidates, key=lambda c: c[2])
    best_points, best_lines, best_total = best_cand
    
    # Phase 2: Local search on the best candidate
    if time.time() < deadline - 60:
        pts_ls, lines_ls, total_ls = _local_search_points(
            best_points, deadline - 60
        )
        if total_ls > best_total:
            best_points, best_lines, best_total = pts_ls, lines_ls, total_ls
    
    # Phase 3: Simulated annealing
    remaining = deadline - time.time()
    if remaining > 20:
        pts_sa, lines_sa, total_sa = _sa_optimize(
            best_points, best_lines, remaining
        )
        if total_sa > best_total:
            best_points, best_lines, best_total = pts_sa, lines_sa, total_sa
    
    # Final validation and rounding
    result = _final_validate_and_round(best_points, best_lines)
    if result is None:
        # Fallback: use the original grid which is always valid
        pts, lines = _grid_init()
        result = (pts, lines)
    
    return result

# EVOLVE-BLOCK-END
