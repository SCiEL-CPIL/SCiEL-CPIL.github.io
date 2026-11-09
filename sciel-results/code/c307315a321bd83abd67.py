# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random
from itertools import combinations


def _canonical_line(a, b, c):
    """Normalize line (A,B,C) so first nonzero entry is positive and gcd divides all."""
    # Make sure at least one of a,b is nonzero (we should never have a=b=0)
    if a == 0 and b == 0:
        return (0, 0, 0)
    g = math.gcd(abs(a), math.gcd(abs(b), abs(c)))
    if g > 0:
        a, b, c = a // g, b // g, c // g
    # Make first nonzero entry positive
    if a < 0:
        a, b, c = -a, -b, -c
    elif a == 0 and b < 0:
        a, b, c = -a, -b, -c
    return (a, b, c)


def _line_from_points(p1, p2):
    """Given two points (x1,y1) and (x2,y2), return canonical line (A,B,C)."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return _canonical_line(a, b, c)


def _count_incidences(points, lines):
    """Count incidences: number of (point, line) pairs where A*x+B*y+C==0."""
    # points: (54,2) int array, lines: (27,3) int array
    # Fully vectorized: (27,54) matrix of A*x+B*y+C, count zeros
    if len(points) == 0 or len(lines) == 0:
        return 0
    A = lines[:, 0:1]  # (27,1)
    B = lines[:, 1:2]  # (27,1)
    C = lines[:, 2:3]  # (27,1)
    x = points[:, 0:1].T  # (1,54)
    y = points[:, 1:2].T  # (1,54)
    vals = A * x + B * y + C  # (27,54)
    return int(np.sum(vals == 0))


def _all_lines_through_pairs(points):
    """For a set of points, find all distinct lines through pairs, with point counts."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_points(points[i], points[j])
            if line not in line_counts:
                line_counts[line] = 0
            line_counts[line] += 1
    # Convert to list of (line, count)
    return line_counts


def _score_configuration(points, top_k_lines):
    """Score a configuration: points (54,2) and a list of (line, count) pairs (top_k_lines)."""
    # We need to select exactly 27 lines from top_k_lines to maximize total count
    # But the lines are already sorted by count descending
    if len(top_k_lines) <= 27:
        # Use all
        lines_arr = np.array([l[0] for l in top_k_lines], dtype=np.int64).reshape(-1, 3)
    else:
        lines_arr = np.array([l[0] for l in top_k_lines[:27]], dtype=np.int64).reshape(-1, 3)
    # Verify shape
    if lines_arr.shape != (27, 3):
        return 0
    return _count_incidences(points, lines_arr)


def _score_with_lines(points, lines_list):
    """Score with explicit list of 27 lines (each is a tuple (A,B,C))."""
    lines_arr = np.array(lines_list, dtype=np.int64).reshape(27, 3)
    return _count_incidences(points, lines_arr)


def _local_search(points, num_steps=50000, deadline=None, seed=42):
    """Local search: swap points and/or lines to improve incidence count."""
    best_points = points.copy()
    # Find initial top 27 lines
    line_counts = _all_lines_through_pairs(best_points)
    # Sort by count descending
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    best_lines = [l[0] for l in sorted_lines[:27]]
    best_score = _score_with_lines(best_points, best_lines)
    
    # Local search
    random.seed(seed)
    improved = True
    step = 0
    while improved and step < num_steps:
        improved = False
        for _ in range(num_steps):
            if deadline is not None and time.time() > deadline:
                break
            step += 1
            # Try moving a random point
            idx = random.randint(0, 53)
            old_p = best_points[idx].copy()
            # Generate a new random point
            new_p = np.array([random.randint(-20, 20), random.randint(-20, 20)], dtype=np.int64)
            if np.array_equal(new_p, old_p):
                continue
            # Check if new point is already in the set
            if any(np.array_equal(best_points[i], new_p) for i in range(54)):
                continue
            
            best_points[idx] = new_p
            # Recompute lines
            lc = _all_lines_through_pairs(best_points)
            sl = sorted(lc.items(), key=lambda x: -x[1])
            new_lines = [l[0] for l in sl[:27]]
            new_score = _score_with_lines(best_points, new_lines)
            if new_score > best_score:
                best_score = new_score
                best_lines = new_lines
                improved = True
            else:
                best_points[idx] = old_p
        
        if deadline is not None and time.time() > deadline:
            break
    
    # Line-based swap search: try replacing one of the 27 lines with a non-selected line
    if deadline is not None and time.time() < deadline:
        line_counts = _all_lines_through_pairs(best_points)
        all_sorted = sorted(line_counts.items(), key=lambda x: -x[1])
        selected_set = set(best_lines)
        # Try swapping each selected line with each non-selected line
        for i in range(27):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(27, min(len(all_sorted), 100)):
                if deadline is not None and time.time() > deadline:
                    break
                new_lines = list(best_lines)
                new_lines[i] = all_sorted[j][0]
                # Check for duplicates
                if len(set(new_lines)) != 27:
                    continue
                new_score = _score_with_lines(best_points, new_lines)
                if new_score > best_score:
                    best_score = new_score
                    best_lines = new_lines
    
    return best_points, best_lines, best_score


def _simulated_annealing(points, deadline, initial_temp=10.0, cooling=0.999, steps_per_iter=100, seed=123):
    """Simulated annealing to escape local optima."""
    best_points = points.copy()
    line_counts = _all_lines_through_pairs(best_points)
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    best_lines = [l[0] for l in sorted_lines[:27]]
    best_score = _score_with_lines(best_points, best_lines)
    cur_score = best_score
    
    temp = initial_temp
    random.seed(seed)
    
    while time.time() < deadline:
        if temp < 0.001:
            break
        for _ in range(steps_per_iter):
            if time.time() > deadline:
                break
            idx = random.randint(0, 53)
            old_p = best_points[idx].copy()
            new_p = np.array([random.randint(-25, 25), random.randint(-25, 25)], dtype=np.int64)
            if np.array_equal(new_p, old_p):
                continue
            if any(np.array_equal(best_points[i], new_p) for i in range(54)):
                continue
            
            best_points[idx] = new_p
            lc = _all_lines_through_pairs(best_points)
            sl = sorted(lc.items(), key=lambda x: -x[1])
            new_lines = [l[0] for l in sl[:27]]
            new_score = _score_with_lines(best_points, new_lines)
            
            delta = new_score - cur_score
            if delta >= 0 or random.random() < math.exp(delta / max(temp, 1e-9)):
                cur_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_lines = new_lines
            else:
                best_points[idx] = old_p
        
        temp *= cooling
    
    return best_points, best_lines, best_score


def _initial_grid_6x9():
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points


def _initial_grid_9x6():
    xs = np.arange(9)
    ys = np.arange(6)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points


def _initial_grid_3x18():
    xs = np.arange(3)
    ys = np.arange(18)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points


def _initial_grid_18x3():
    xs = np.arange(18)
    ys = np.arange(3)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points


def _initial_grid_2x27():
    xs = np.arange(2)
    ys = np.arange(27)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points


def _initial_grid_27x2():
    xs = np.arange(27)
    ys = np.arange(2)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points


def _initial_grid_7x8_minus2():
    points_list = []
    for x in range(7):
        for y in range(8):
            points_list.append((x, y))
    # Remove 2 points to get 54
    points_list.remove((0, 0))
    points_list.remove((6, 7))
    return np.array(points_list, dtype=np.int64)


def _initial_grid_8x7_minus2():
    points_list = []
    for x in range(8):
        for y in range(7):
            points_list.append((x, y))
    points_list.remove((0, 0))
    points_list.remove((7, 6))
    return np.array(points_list, dtype=np.int64)


def _initial_grid_5x11():
    xs = np.arange(5)
    ys = np.arange(11)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    # 5*11=55, remove one point
    points = np.delete(points, 0, axis=0)
    return points


def _initial_grid_11x5():
    xs = np.arange(11)
    ys = np.arange(5)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    points = np.delete(points, 0, axis=0)
    return points


def _initial_grid_6x9_v2():
    """6x9 grid with different corner removal pattern."""
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points


def _initial_grid_9x6_v2():
    """9x6 grid variant."""
    xs = np.arange(9)
    ys = np.arange(6)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points


def _initial_grid_3x18_v2():
    """3x18 grid variant with offset."""
    xs = np.arange(3)
    ys = np.arange(18)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    # Shift to center
    points[:, 0] += 5
    points[:, 1] += 3
    return points


def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences with 54 points and 27 lines.
    Uses multiple initializations, local search, and simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    # Multiple initial configurations
    initializers = [
        _initial_grid_6x9,
        _initial_grid_9x6,
        _initial_grid_3x18,
        _initial_grid_18x3,
        _initial_grid_2x27,
        _initial_grid_27x2,
        _initial_grid_7x8_minus2,
        _initial_grid_8x7_minus2,
        _initial_grid_5x11,
        _initial_grid_11x5,
    ]
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Phase 1: Local search on each initial configuration with increasing effort
    # Use more steps for the most promising configs
    for idx, init_func in enumerate(initializers):
        if time.time() > deadline:
            break
        points = init_func()
        
        # Allocate more steps to earlier (typically better) initializers
        if idx < 3:
            steps = 5000
        elif idx < 6:
            steps = 3000
        else:
            steps = 2000
        
        # Quick local search
        pts, lines, score = _local_search(points, num_steps=steps, deadline=deadline, seed=42 + idx)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines
    
    # Phase 2: Simulated annealing on best configuration
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 30:
            pts, lines, score = _simulated_annealing(
                best_points, deadline=deadline, initial_temp=20.0, cooling=0.9995, steps_per_iter=50, seed=123
            )
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                best_lines = lines
    
    # Phase 3: More local search on improved configuration
    if time.time() < deadline:
        pts, lines, score = _local_search(best_points, num_steps=50000, deadline=deadline, seed=999)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines
    
    # Phase 4: Final polish with line swaps and point moves
    if time.time() < deadline:
        # Try a few more SA restarts with different seeds
        for sa_seed in [200, 300, 400]:
            if time.time() > deadline:
                break
            pts, lines, score = _simulated_annealing(
                best_points, deadline=deadline, initial_temp=10.0, cooling=0.999, steps_per_iter=30, seed=sa_seed
            )
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                best_lines = lines
            # Follow up with local search
            if time.time() < deadline:
                pts, lines, score = _local_search(best_points, num_steps=10000, deadline=deadline, seed=sa_seed + 1000)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
                    best_lines = lines
    
    # Final: ensure we have exactly 27 lines
    lines_arr = np.array(best_lines, dtype=np.int64).reshape(27, 3)
    points_arr = best_points.astype(np.int64)
    
    # Validation
    if points_arr.shape != (54, 2):
        raise ValueError("Points shape mismatch")
    if lines_arr.shape != (27, 3):
        raise ValueError("Lines shape mismatch")
    
    # Check for duplicate lines
    canonical_lines = [_canonical_line(*l) for l in lines_arr]
    if len(set(canonical_lines)) != 27:
        # Try to fix by finding unique lines
        unique = []
        seen = set()
        for cl in canonical_lines:
            if cl not in seen:
                seen.add(cl)
                unique.append(cl)
        if len(unique) < 27:
            # This shouldn't happen, but fallback
            lines_arr = np.array(unique[:27], dtype=np.int64).reshape(27, 3)
    
    # Check for duplicate points
    point_tuples = [tuple(p) for p in points_arr]
    if len(set(point_tuples)) != 54:
        raise ValueError("Duplicate points found")
    
    return points_arr, lines_arr

# EVOLVE-BLOCK-END
