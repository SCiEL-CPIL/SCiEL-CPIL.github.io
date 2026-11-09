# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from itertools import combinations


def _normalize_line(A, B, C):
    """Normalize a line so it's unique: divide by gcd, make first nonzero positive."""
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _count_incidences(points, lines):
    """Count point-line incidences exactly."""
    n, m = len(points), len(lines)
    count = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count


def _build_grid_points(a, b):
    """Build grid points (i+1, j+1) for i in [0,a), j in [0,b)."""
    points = []
    for i in range(a):
        for j in range(b):
            points.append((i + 1, j + 1))
    return np.array(points, dtype=np.int64)


def _generate_candidate_lines(points, max_slope_num=6, max_slope_den=6):
    """
    Generate candidate lines from a set of points.
    Includes:
    - All lines through pairs of points
    - Horizontal and vertical lines through each point
    """
    n = len(points)
    candidate_lines = []
    seen = set()
    
    # Lines through pairs of points
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            A = dy
            B = -dx
            C = -dy * x1 + dx * y1
            norm = _normalize_line(A, B, C)
            if norm not in seen:
                seen.add(norm)
                candidate_lines.append((A, B, C))
    
    return candidate_lines


def _select_best_lines(points, candidate_lines, m_lines=27):
    """Select the best m_lines lines to maximize total incidences."""
    n_points = len(points)
    n_cand = len(candidate_lines)
    
    # Count incidences for each candidate line
    line_inc_counts = np.zeros(n_cand, dtype=np.int64)
    for j in range(n_cand):
        A, B, C = candidate_lines[j]
        for i in range(n_points):
            x, y = points[i]
            if A * x + B * y + C == 0:
                line_inc_counts[j] += 1
    
    # Sort by incidence count descending
    order = np.argsort(-line_inc_counts)
    
    # Select top m_lines unique lines
    selected = []
    seen_norm = set()
    for j in order:
        norm = _normalize_line(*candidate_lines[j])
        if norm not in seen_norm:
            seen_norm.add(norm)
            selected.append(j)
        if len(selected) >= m_lines:
            break
    
    if len(selected) < m_lines:
        # Pad with remaining lines if needed
        for j in order:
            if j not in selected:
                selected.append(j)
            if len(selected) >= m_lines:
                break
    
    selected_lines = np.array([candidate_lines[j] for j in selected[:m_lines]], dtype=np.int64)
    total = sum(line_inc_counts[j] for j in selected[:m_lines])
    
    return selected_lines, total


def _local_search_points(points, lines, max_iterations=500, time_limit=120):
    """
    Local search: try moving each point to nearby positions to increase incidences.
    """
    start_time = time.time()
    current_points = points.copy()
    current_lines = lines.copy()
    current_score = _count_incidences(current_points, current_lines)
    
    for iteration in range(max_iterations):
        if time.time() > start_time + time_limit:
            break
        
        improved = False
        # Try moving each point
        for pi in range(len(current_points)):
            if time.time() > start_time + time_limit:
                break
            
            orig_x, orig_y = current_points[pi]
            # Try nearby positions in a 5x5 neighborhood
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    if new_x < 1 or new_y < 1:
                        continue
                    
                    # Check if position is already occupied
                    occupied = False
                    for i in range(len(current_points)):
                        if i != pi and current_points[i][0] == new_x and current_points[i][1] == new_y:
                            occupied = True
                            break
                    if occupied:
                        continue
                    
                    # Temporarily move the point
                    saved = current_points[pi].copy()
                    current_points[pi] = [new_x, new_y]
                    
                    # Recompute incidences
                    new_score = _count_incidences(current_points, current_lines)
                    
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        current_points[pi] = saved
            
            if not improved:
                break
        
        if not improved:
            break
    
    return current_points, current_lines, current_score


def _refine_lines_for_points(points, max_time=30):
    """
    Given fixed points, find the best 27 lines by exhaustive candidate generation
    and greedy selection.
    """
    start_time = time.time()
    n = len(points)
    
    # Generate all lines through pairs of points
    candidate_lines = _generate_candidate_lines(points)
    
    # Select best 27
    selected_lines, score = _select_best_lines(points, candidate_lines, 27)
    
    return selected_lines, score


def _try_grid_configuration(a, b, time_budget=60):
    """
    Try a specific grid configuration and refine.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    points = _build_grid_points(a, b)
    
    # Generate candidate lines
    candidate_lines = _generate_candidate_lines(points)
    
    # Select best 27 lines
    selected_lines, score = _select_best_lines(points, candidate_lines, 27)
    
    # Local search on points
    if time.time() < deadline - 10:
        refined_points, refined_lines, refined_score = _local_search_points(
            points, selected_lines, max_iterations=300, time_limit=30
        )
        if refined_score > score:
            points = refined_points
            score = refined_score
            selected_lines = refined_lines
    
    # Re-optimize lines for refined points
    if time.time() < deadline - 5:
        final_lines, final_score = _refine_lines_for_points(points, max_time=10)
        if final_score > score:
            selected_lines = final_lines
            score = final_score
    
    return points, selected_lines, score


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences with 54 points and 27 lines.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Try different grid configurations that give 54 points
    # a * b = 54
    grid_configs = []
    for a in range(1, 55):
        if 54 % a == 0:
            b = 54 // a
            grid_configs.append((a, b))
    
    # Sort by more balanced aspect ratio first (likely to have more collinearities)
    grid_configs.sort(key=lambda ab: abs(ab[0] - ab[1]))
    
    # Time budget per configuration
    total_configs = len(grid_configs)
    time_per_config = min(60, (deadline - start_time - 30) // max(1, total_configs))
    
    for a, b in grid_configs:
        if time.time() > deadline - 30:
            break
        
        points, lines, score = _try_grid_configuration(a, b, time_budget=time_per_config)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = lines.copy()
    
    # Final refinement: try to improve the best solution found
    if best_points is not None and time.time() < deadline - 60:
        # More aggressive local search
        refined_points, refined_lines, refined_score = _local_search_points(
            best_points, best_lines, max_iterations=1000, time_limit=60
        )
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
            best_lines = refined_lines
        
        # Re-optimize lines
        if time.time() < deadline - 30:
            final_lines, final_score = _refine_lines_for_points(best_points, max_time=20)
            if final_score > best_score:
                best_score = final_score
                best_lines = final_lines
    
    # Ensure we have valid output
    if best_points is None:
        # Fallback: simple 3x18 grid
        best_points = _build_grid_points(3, 18)
        candidate_lines = _generate_candidate_lines(best_points)
        best_lines, _ = _select_best_lines(best_points, candidate_lines, 27)
    
    # Ensure shapes are correct
    assert best_points.shape == (54, 2)
    assert best_lines.shape == (27, 3)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
