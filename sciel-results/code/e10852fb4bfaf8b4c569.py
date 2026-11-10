"""Minimize ordinary lines for 12 integer points via local search."""
import time
import random
import math
from itertools import combinations

DEADLINE = time.time() + 140  # safety margin before 150s kill

def _line_key(p1, p2):
    """Canonical key for line through p1=(x1,y1) and p2=(x2,y2)."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, 0, x1)  # vertical line: x = x1
    # Normalize: make dx > 0, reduce by gcd
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    # Line: dy*x - dx*y = dy*x1 - dx*y1
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    # A line with k points is counted C(k,2) times in the pairs.
    # Ordinary lines are those with exactly 2 points, i.e., counted once.
    # But we need to be careful: line_counts[key] = number of pairs on that line.
    # If a line has m points, it contributes C(m,2) to the count.
    # We want lines where m == 2, i.e., line_counts[key] == 1.
    ordinary = sum(1 for v in line_counts.values() if v == 1)
    return ordinary

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
    if len(points) < 3:
        return True
    p0, p1, p2 = points[0], points[1], points[2]
    (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
    if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
        return True
    # All three collinear, check all
    for i in range(3, 12):
        xi, yi = points[i]
        if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
            return True
    return False

def _clamp(p, limit=10**9):
    x, y = p
    x = max(-limit, min(limit, x))
    y = max(-limit, min(limit, y))
    return [x, y]

def construct():
    random.seed(42)
    
    # Start with a 3x4 grid as initial seed
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = count_ordinary_lines(best_points)
    
    # Try several random starting configurations
    for seed in range(50):
        if time.time() > DEADLINE - 10:
            break
        random.seed(seed + 42)
        points = []
        for _ in range(12):
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            points.append([x, y])
        if not _is_valid(points):
            continue
        score = count_ordinary_lines(points)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in points]
    
    # Structured seeds including Böröczky-type and other geometric configurations
    structured_seeds = [
        # 3x4 grid
        [[x, y] for x in range(4) for y in range(3)],
        # 4x3 grid
        [[x, y] for x in range(3) for y in range(4)],
        # Points on two parallel lines (6 each)
        [[i, 0] for i in range(6)] + [[i, 2] for i in range(6)],
        # Points on three parallel lines (4 each)
        [[i, 0] for i in range(4)] + [[i, 3] for i in range(4)] + [[i, 6] for i in range(4)],
        # Hexagon-like (two concentric hexagons)
        [[3, 0], [1, 2], [-1, 2], [-3, 0], [-1, -2], [1, -2],
         [6, 0], [2, 4], [-2, 4], [-6, 0], [-2, -4], [2, -4]],
        # 12 points on a circle (approximate with integers)
        [[7, 24], [15, 8], [8, 15], [-7, 24], [-15, 8], [-8, 15],
         [7, -24], [15, -8], [8, -15], [-7, -24], [-15, -8], [-8, -15]],
        # Böröczky-type: 6 points on conic + 6 on line
        [[0, 0], [1, 1], [2, 4], [3, 9], [4, 16], [5, 25],
         [10, 0], [20, 0], [30, 0], [40, 0], [50, 0], [60, 0]],
        # Another Böröczky variant
        [[0, 0], [1, 2], [2, 5], [3, 10], [4, 17], [5, 26],
         [10, 1], [20, 1], [30, 1], [40, 1], [50, 1], [60, 1]],
        # 4 points on each of 3 concurrent lines
        [[0, 0], [2, 0], [4, 0], [6, 0],
         [0, 0], [0, 2], [0, 4], [0, 6],
         [0, 0], [1, 1], [2, 2], [3, 3]],
        # Rectangular grid 2x6
        [[x, y] for x in range(6) for y in range(2)],
        # 12 points forming a regular dodecagon (approximate)
        [[11, 0], [8, 8], [0, 11], [-8, 8], [-11, 0], [-8, -8],
         [0, -11], [8, -8], [5, 5], [-5, 5], [-5, -5], [5, -5]],
    ]
    
    for s in structured_seeds:
        if time.time() > DEADLINE - 10:
            break
        if not _is_valid(s):
            continue
        sc = count_ordinary_lines(s)
        if sc < best_score:
            best_score = sc
            best_points = [p[:] for p in s]
    
    # Local search from best
    points = [p[:] for p in best_points]
    score = count_ordinary_lines(points)
    
    # Hill climbing with multiple move types
    step_size = 20
    no_improve = 0
    max_no_improve = 5000
    
    for iteration in range(100000):
        if time.time() > DEADLINE - 5:
            break
        if score <= 6:  # theoretical minimum
            break
        if no_improve > max_no_improve:
            # Restart with a different configuration
            if best_score < score:
                points = [p[:] for p in best_points]
                score = best_score
            else:
                # Try a new random start
                random.seed(iteration % 1000)
                for _ in range(12):
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    points.append([x, y])
                if not _is_valid(points):
                    points = [p[:] for p in best_points]
                    score = best_score
                else:
                    score = count_ordinary_lines(points)
                    if score < best_score:
                        best_score = score
                        best_points = [p[:] for p in points]
            no_improve = 0
            continue
        
        # Choose move type: 0 = single point move, 1 = two point swap
        move_type = random.random()
        
        if move_type < 0.7:
            # Single point move
            idx = random.randint(0, 11)
            dx = random.randint(-step_size, step_size)
            dy = random.randint(-step_size, step_size)
            if dx == 0 and dy == 0:
                continue
            old_x, old_y = points[idx]
            new_x, new_y = old_x + dx, old_y + dy
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            new_pt = [new_x, new_y]
            valid = True
            for i in range(12):
                if i != idx and points[i] == new_pt:
                    valid = False
                    break
            if not valid:
                continue
            points[idx] = new_pt
            new_score = count_ordinary_lines(points)
            if new_score < score:
                score = new_score
                no_improve = 0
                if score < best_score:
                    best_score = score
                    best_points = [p[:] for p in points]
            elif new_score == score:
                no_improve += 1
            else:
                points[idx] = [old_x, old_y]
                no_improve += 1
        else:
            # Two point swap: move two points simultaneously
            i1, i2 = random.sample(range(12), 2)
            dx1 = random.randint(-step_size, step_size)
            dy1 = random.randint(-step_size, step_size)
            dx2 = random.randint(-step_size, step_size)
            dy2 = random.randint(-step_size, step_size)
            if (dx1 == 0 and dy1 == 0) or (dx2 == 0 and dy2 == 0):
                continue
            old1 = points[i1][:]
            old2 = points[i2][:]
            new1 = [old1[0] + dx1, old1[1] + dy1]
            new2 = [old2[0] + dx2, old2[1] + dy2]
            if abs(new1[0]) > 10**9 or abs(new1[1]) > 10**9:
                continue
            if abs(new2[0]) > 10**9 or abs(new2[1]) > 10**9:
                continue
            # Check distinctness
            valid = True
            for i in range(12):
                if i not in (i1, i2):
                    if points[i] == new1 or points[i] == new2:
                        valid = False
                        break
            if new1 == new2:
                valid = False
            if not valid:
                continue
            points[i1] = new1
            points[i2] = new2
            new_score = count_ordinary_lines(points)
            if new_score < score:
                score = new_score
                no_improve = 0
                if score < best_score:
                    best_score = score
                    best_points = [p[:] for p in points]
            elif new_score == score:
                no_improve += 1
            else:
                points[i1] = old1
                points[i2] = old2
                no_improve += 1
        
        # Occasionally decrease step size
        if iteration % 10000 == 0 and step_size > 1:
            step_size = max(1, step_size // 2)
    
    # Final polish pass with small steps
    points = [p[:] for p in best_points]
    score = count_ordinary_lines(points)
    step_size = 2
    for iteration in range(20000):
        if time.time() > DEADLINE - 2:
            break
        if score <= 6:
            break
        idx = random.randint(0, 11)
        dx = random.randint(-step_size, step_size)
        dy = random.randint(-step_size, step_size)
        if dx == 0 and dy == 0:
            continue
        old_x, old_y = points[idx]
        new_x, new_y = old_x + dx, old_y + dy
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
        new_pt = [new_x, new_y]
        valid = True
        for i in range(12):
            if i != idx and points[i] == new_pt:
                valid = False
                break
        if not valid:
            continue
        points[idx] = new_pt
        new_score = count_ordinary_lines(points)
        if new_score < score:
            score = new_score
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in points]
        else:
            points[idx] = [old_x, old_y]
    
    return best_points
