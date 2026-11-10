"""Minimize ordinary lines for 12 integer points via fast local search."""
import time
import random
import math
from collections import defaultdict

_DEADLINE = time.time() + 130  # safety margin before 150s kill

def _line_key(p1, p2):
    """Canonical key for line through p1=(x1,y1) and p2=(x2,y2)."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, 0, x1)  # vertical line: x = x1
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _build_line_map(points):
    """Build dict: line_key -> set of point indices on that line."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_map[key].add(i)
            line_map[key].add(j)
    return line_map

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    ordinary = sum(1 for v in line_counts.values() if v == 1)
    return ordinary

def _count_ordinary_from_line_map(line_map):
    """Count ordinary lines from a line_map (line_key -> set of indices)."""
    return sum(1 for s in line_map.values() if len(s) == 2)

def _is_valid(points):
    """Check: 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    p0, p1, p2 = points[0], points[1], points[2]
    (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
    if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
        return True
    for i in range(3, 12):
        xi, yi = points[i]
        if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
            return True
    return False

def _clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

def _fast_move(points, idx, new_pt, line_map, current_score):
    """
    Try moving point idx to new_pt. Returns (new_score, new_line_map) or (None, None).
    Uses incremental update: only lines through the moved point change.
    """
    n = len(points)
    old_pt = points[idx]
    if new_pt == old_pt:
        return None, None
    # Check distinctness
    for i in range(n):
        if i != idx and points[i] == new_pt:
            return None, None
    if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
        return None, None

    # Work on a copy of the line_map
    new_line_map = {k: set(s) for k, s in line_map.items()}

    # Collect all lines involving point idx (through old position)
    affected_old_lines = set()
    for j in range(n):
        if j != idx:
            key = _line_key(old_pt, points[j])
            affected_old_lines.add(key)

    # Remove idx from those lines; track score delta
    score_delta = 0
    for key in affected_old_lines:
        if key in new_line_map:
            old_size = len(new_line_map[key])
            new_line_map[key].discard(idx)
            new_size = len(new_line_map[key])
            # Before: was this an ordinary line?
            if old_size == 2:
                score_delta -= 1
            if new_size == 2:
                score_delta += 1
            if new_size == 0:
                del new_line_map[key]

    # Add new lines through new position
    for j in range(n):
        if j != idx:
            key = _line_key(new_pt, points[j])
            if key not in new_line_map:
                new_line_map[key] = set()
            old_size = len(new_line_map[key])
            new_line_map[key].add(idx)
            new_line_map[key].add(j)
            new_size = len(new_line_map[key])
            if old_size == 2:
                score_delta -= 1
            if new_size == 2:
                score_delta += 1

    new_score = current_score + score_delta
    return new_score, new_line_map

def construct():
    rng = random.Random(42)

    # Expanded structured seeds based on known extremal configurations
    structured_seeds = [
        # 3x4 grid
        [[x, y] for x in range(4) for y in range(3)],
        # 4x3 grid
        [[x, y] for x in range(3) for y in range(4)],
        # Points on two parallel lines
        [[i, 0] for i in range(6)] + [[i, 1] for i in range(6)],
        # Points on three parallel lines
        [[i, 0] for i in range(4)] + [[i, 1] for i in range(4)] + [[i, 2] for i in range(4)],
        # Hexagon-like (two concentric hexagons)
        [[3, 0], [1, 2], [-1, 2], [-3, 0], [-1, -2], [1, -2],
         [6, 0], [2, 4], [-2, 4], [-6, 0], [-2, -4], [2, -4]],
        # 12 points on a circle (approximate with integers)
        [[7, 24], [15, 8], [8, 15], [-7, 24], [-15, 8], [-8, 15],
         [7, -24], [15, -8], [8, -15], [-7, -24], [-15, -8], [-8, -15]],
        # 4x3 grid
        [[0, 0], [1, 0], [2, 0], [3, 0],
         [0, 1], [1, 1], [2, 1], [3, 1],
         [0, 3], [1, 3], [2, 3], [3, 3]],
        # 3x4 grid
        [[0, 0], [1, 0], [2, 0], [3, 0],
         [0, 1], [1, 1], [2, 1], [3, 1],
         [0, 2], [1, 2], [2, 2], [3, 2]],
        # 2x6 grid
        [[x, y] for x in range(6) for y in range(2)],
        # 6x2 grid
        [[x, y] for x in range(2) for y in range(6)],
        # 3x3 grid plus 3 extra
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2],
         [3, 0], [3, 1], [3, 2]],
        # Symmetric cross pattern
        [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1],
         [1, -1], [-1, 1], [2, 0], [0, 2], [0, -2]],
        # 3-3-3-3 on 4 lines forming a grid
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1],
         [0, 2], [1, 2], [2, 2], [0, 3], [1, 3], [2, 3]],
        # 5 on line y=0, 5 on line y=1, 2 connecting
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
         [0, 1], [1, 1], [2, 1], [3, 1], [4, 1],
         [5, 0], [5, 1]],
        # 4 on x-axis, 4 on y-axis, 4 off
        [[-3, 0], [-1, 0], [1, 0], [3, 0],
         [0, -3], [0, -1], [0, 1], [0, 3],
         [2, 2], [-2, 2], [2, -2], [-2, -2]],
        # 3x4 with shifted rows for more diagonal collinearities
        [[0, 0], [1, 0], [2, 0], [3, 0],
         [1, 1], [2, 1], [3, 1], [4, 1],
         [2, 2], [3, 2], [4, 2], [5, 2]],
        # Regular hexagon doubled (small + large)
        [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
         [3, 0], [0, 3], [-3, 3], [-3, 0], [0, -3], [3, -3]],
        # 12 points: 3 on each of 4 lines through origin
        [[1, 0], [2, 0], [4, 0],
         [0, 1], [0, 2], [0, 4],
         [1, 1], [2, 2], [4, 4],
         [1, -1], [2, -2], [4, -4]],
        # 12 points: 3 on each of 4 lines, different slopes
        [[1, 0], [2, 0], [3, 0],
         [0, 1], [0, 2], [0, 3],
         [1, 1], [2, 2], [3, 3],
         [1, -1], [2, -2], [3, -3]],
    ]

    best_points = None
    best_score = float('inf')

    for s in structured_seeds:
        if time.time() > _DEADLINE - 10:
            break
        if not _is_valid(s):
            continue
        sc = count_ordinary_lines(s)
        if sc < best_score:
            best_score = sc
            best_points = [p[:] for p in s]

    # Try random starting configurations
    for seed in range(100):
        if time.time() > _DEADLINE - 10:
            break
        rng2 = random.Random(seed + 42)
        points = []
        seen = set()
        while len(points) < 12:
            x = rng2.randint(-50, 50)
            y = rng2.randint(-50, 50)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        if not _is_valid(points):
            continue
        score = count_ordinary_lines(points)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in points]

    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = count_ordinary_lines(best_points)

    # Local search with fast incremental updates
    current_points = [list(p) for p in best_points]
    current_line_map = _build_line_map(current_points)
    current_score = _count_ordinary_from_line_map(current_line_map)
    if current_score < best_score:
        best_score = current_score
        best_points = [list(p) for p in current_points]

    no_improve = 0
    max_no_improve = 200000

    while time.time() < _DEADLINE and no_improve < max_no_improve:
        # Choose a random point and a random new position
        idx = rng.randint(0, 11)
        # Try multiple candidate moves for this point
        found_improvement = False
        for attempt in range(3):
            step_x = rng.randint(-8, 8)
            step_y = rng.randint(-8, 8)
            new_pt = [current_points[idx][0] + step_x, current_points[idx][1] + step_y]
            new_pt[0] = _clamp(new_pt[0], -1000, 1000)
            new_pt[1] = _clamp(new_pt[1], -1000, 1000)

            new_score, new_line_map = _fast_move(current_points, idx, new_pt, current_line_map, current_score)
            if new_score is None:
                continue

            if new_score < current_score:
                current_points[idx] = new_pt
                current_line_map = new_line_map
                current_score = new_score
                no_improve = 0
                found_improvement = True
                if current_score < best_score:
                    best_score = current_score
                    best_points = [list(p) for p in current_points]
                break
            elif new_score == current_score and rng.random() < 0.05:
                # Sideways move to escape local minima
                current_points[idx] = new_pt
                current_line_map = new_line_map
                no_improve += 1
                break

        if not found_improvement:
            no_improve += 1

        # Every 5000 no-improve iterations, try a bigger perturbation
        if no_improve > 0 and no_improve % 5000 == 0:
            new_points = [list(p) for p in current_points]
            for _ in range(4):
                idx2 = rng.randint(0, 11)
                step = rng.randint(-30, 30)
                axis = rng.randint(0, 1)
                new_points[idx2][axis] = _clamp(new_points[idx2][axis] + step, -1000, 1000)
            if _is_valid(new_points):
                # Check distinctness
                seen = set()
                valid = True
                for p in new_points:
                    if tuple(p) in seen:
                        valid = False
                        break
                    seen.add(tuple(p))
                if valid:
                    lm = _build_line_map(new_points)
                    s = _count_ordinary_from_line_map(lm)
                    if s <= current_score:
                        current_points = new_points
                        current_line_map = lm
                        current_score = s
                        if s < best_score:
                            best_score = s
                            best_points = [list(p) for p in current_points]
                            no_improve = 0

    # Final verification
    if not _is_valid(best_points):
        best_points = [[x, y] for x in range(4) for y in range(3)]

    return best_points
