import numpy as np
import time
import math
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) by dividing by gcd and making first nonzero positive."""
    A, B, C = int(A), int(B), int(C)
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 1:
        A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count exact integer incidences between points and lines."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    total = 0
    for i in range(len(lns)):
        A, B, C = lns[i]
        val = A * pts[:, 0] + B * pts[:, 1] + C
        total += int(np.sum(val == 0))
    return total

def _is_valid(points, lines):
    """Validate points and lines per the task contract."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    if pts.shape != (54, 2) or lns.shape != (27, 3):
        return False
    if not np.all(np.isfinite(pts.astype(np.float64))):
        return False
    if not np.all(np.isfinite(lns.astype(np.float64))):
        return False
    seen_pts = set()
    for p in pts:
        t = (int(p[0]), int(p[1]))
        if t in seen_pts:
            return False
        seen_pts.add(t)
    seen_lines = set()
    for l in lns:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm is None:
            return False
        if norm in seen_lines:
            return False
        seen_lines.add(norm)
    return True

def _find_optimal_lines(points, max_lines=27):
    """
    Greedy set-coverage: iteratively pick the line that adds the most
    *newly* covered points (points not already on any selected line).
    This accounts for overlap between lines.
    """
    pts = np.asarray(points, dtype=np.int64)
    n = len(pts)
    
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            norm = _normalize_line(A, B, C)
            if norm is None:
                continue
            if norm not in line_map:
                line_map[norm] = set()
            line_map[norm].add(i)
            line_map[norm].add(j)
    
    # Convert to frozensets for faster set operations
    line_sets = {norm: frozenset(indices) for norm, indices in line_map.items()}
    
    covered = set()
    selected = []
    
    for _ in range(max_lines):
        best_gain = 0
        best_norm = None
        for norm, s in line_sets.items():
            if norm in selected:
                continue
            gain = len(s - covered)
            if gain > best_gain:
                best_gain = gain
                best_norm = norm
        if best_norm is None:
            break
        selected.append(best_norm)
        covered |= line_sets[best_norm]
    
    lines_arr = np.array(selected, dtype=np.int64)
    return lines_arr

def _build_grid(nx, ny):
    """Build a nx * ny grid of points. If nx*ny > 54, trim to 54."""
    xs = np.arange(nx)
    ys = np.arange(ny)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    if len(points) > 54:
        points = points[:54]
    elif len(points) < 54:
        return None, None
    
    if len(points) != 54:
        return None, None
    
    lines = _find_optimal_lines(points, 27)
    return points, lines

def _local_search(points, lines, deadline):
    """
    Local search: iteratively try to improve by replacing the line whose
    removal loses the most *unique* coverage with a line that adds the most
    new coverage. Uses incremental score tracking.
    """
    pts = np.asarray(points, dtype=np.int64)
    n = len(pts)
    lines = np.array(lines, dtype=np.int64)
    current_score = _count_incidences(pts, lines)
    
    # Precompute all unique lines through point pairs as frozensets
    all_lines = {}  # norm -> frozenset of point indices
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            norm = _normalize_line(A, B, C)
            if norm is None:
                continue
            if norm not in all_lines:
                all_lines[norm] = set()
            all_lines[norm].add(i)
            all_lines[norm].add(j)
    
    # Convert to frozensets
    line_sets = {norm: frozenset(s) for norm, s in all_lines.items()}
    line_keys = list(line_sets.keys())
    
    # Track which lines are currently selected
    selected_norms = set()
    for l in lines:
        norm = _normalize_line(l[0], l[1], l[2])
        if norm is not None:
            selected_norms.add(norm)
    
    # Compute current covered set
    covered = set()
    for norm in selected_norms:
        covered |= line_sets.get(norm, frozenset())
    current_score = len(covered)
    
    improved = True
    max_iterations = 200
    
    while improved and time.time() < deadline and max_iterations > 0:
        improved = False
        max_iterations -= 1
        
        # For each selected line, compute how many points it uniquely covers
        # (points only on this line among selected lines)
        line_unique = {}
        for norm in selected_norms:
            s = line_sets.get(norm, frozenset())
            # Points on this line but not on any other selected line
            others = set()
            for other in selected_norms:
                if other != norm:
                    others |= line_sets.get(other, frozenset())
            unique = s - others
            line_unique[norm] = len(unique)
        
        # Find the line with minimum unique coverage
        min_norm = min(selected_norms, key=lambda x: line_unique[x])
        min_unique = line_unique[min_norm]
        
        # Try to find a better line to replace it
        best_gain = 0
        best_new_norm = None
        
        for norm in line_keys:
            if norm in selected_norms:
                continue
            s = line_sets[norm]
            # New points this line would add
            new_points = s - covered
            gain = len(new_points) - min_unique
            if gain > best_gain:
                best_gain = gain
                best_new_norm = norm
        
        if best_gain > 0 and best_new_norm is not None:
            # Replace the line
            selected_norms.discard(min_norm)
            selected_norms.add(best_new_norm)
            covered = set()
            for norm in selected_norms:
                covered |= line_sets.get(norm, frozenset())
            current_score = len(covered)
            improved = True
    
    # Convert back to line array
    lines_arr = np.array(list(selected_norms), dtype=np.int64)
    # Ensure exactly 27 lines (pad with duplicates if needed, though shouldn't happen)
    if len(lines_arr) < 27:
        # Add high-count lines not already selected
        for norm in line_keys:
            if len(lines_arr) >= 27:
                break
            if norm not in selected_norms:
                lines_arr = np.vstack([lines_arr, [norm]])
                selected_norms.add(norm)
    
    return lines_arr, current_score

def _try_point_swap(points, lines, deadline):
    """
    Try swapping individual points to improve the configuration.
    """
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    lines = np.array(lines, dtype=np.int64)
    current_score = _count_incidences(pts, lines)
    
    # For each point, try moving it to a nearby position
    for pi in range(n):
        if time.time() > deadline:
            break
        ox, oy = pts[pi]
        
        # Try nearby positions
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = ox + dx, oy + dy
                
                # Check if new position is already occupied
                if np.any((pts[:, 0] == nx_) & (pts[:, 1] == ny_)):
                    continue
                
                # Try the swap
                pts[pi] = [nx_, ny_]
                new_score = _count_incidences(pts, lines)
                
                if new_score > current_score:
                    current_score = new_score
                else:
                    pts[pi] = [ox, oy]
    
    return pts, current_score

def szemeredi_trotter_incidences():
    deadline = time.time() + 580.0
    best_pts = None
    best_lines = None
    best_score = -1
    
    # Try different grid aspect ratios
    configs = [
        (6, 9),
        (9, 6),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
        (1, 54),
        (54, 1),
        (4, 14),
        (5, 11),
        (11, 5),
        (14, 4),
    ]
    
    for nx, ny in configs:
        if time.time() > deadline:
            break
        pts, lns = _build_grid(nx, ny)
        if pts is None:
            continue
        if not _is_valid(pts, lns):
            continue
        score = _count_incidences(pts, lns)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            best_lines = lns.copy()
    
    # If no grid worked, fallback to 6x9
    if best_pts is None:
        pts, lns = _build_grid(6, 9)
        if pts is not None:
            best_pts = pts
            best_lines = lns
    
    # If still nothing, return a simple valid construction
    if best_pts is None:
        xs = np.arange(1, 4)
        ys = np.arange(1, 19)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for a in range(1, 4):
            for b in range(1, 10):
                lines.append((a, -1, b))
        best_lines = np.array(lines, dtype=np.int64)
    
    # Local search to improve the best configuration
    if time.time() < deadline - 60:
        improved_lines, improved_score = _local_search(
            best_pts, best_lines, deadline - 60
        )
        if improved_score > best_score:
            best_lines = improved_lines
            best_score = improved_score
    
    # Try point swaps if time remains
    if time.time() < deadline - 30:
        swapped_pts, swapped_score = _try_point_swap(
            best_pts, best_lines, deadline - 30
        )
        if swapped_score > best_score:
            best_pts = swapped_pts
            best_score = swapped_score
            # Re-run line optimization after point changes
            if time.time() < deadline - 10:
                new_lines = _find_optimal_lines(best_pts, 27)
                new_score = _count_incidences(best_pts, new_lines)
                if new_score > best_score:
                    best_lines = new_lines
                    best_score = new_score
                    # Run local search again after point swaps
                    if time.time() < deadline - 5:
                        ls_lines, ls_score = _local_search(
                            best_pts, best_lines, deadline - 5
                        )
                        if ls_score > best_score:
                            best_lines = ls_lines
                            best_score = ls_score
    
    # Final validation and return
    if not _is_valid(best_pts, best_lines):
        # Fallback to a simple valid construction
        xs = np.arange(1, 4)
        ys = np.arange(1, 19)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for a in range(1, 4):
            for b in range(1, 10):
                lines.append((a, -1, b))
        best_lines = np.array(lines, dtype=np.int64)
    
    return best_pts, best_lines
