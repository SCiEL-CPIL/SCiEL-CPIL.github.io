import random
import time
import math
from collections import defaultdict
from itertools import combinations

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    lines = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0:
                dx = -dx
                dy = -dy
            elif dx == 0 and dy < 0:
                dy = -dy

            a = dy
            b = -dx
            c = a * p1[0] + b * p1[1]
            key = (a, b, c)
            lines[key].add(i)
            lines[key].add(j)

    ordinary = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary += 1
    return ordinary


def _make_initial_configs():
    configs = []

    # 3x4 grid (baseline)
    configs.append([[x, y] for x in range(4) for y in range(3)])

    # 4x3 grid (same set, different enumeration - same result)
    # configs.append([[x, y] for x in range(3) for y in range(4)])

    # 2x6 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])

    # 6x2 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])

    # Diagonal-shift grid: y in 0..3, x in 0..2, point = (x+y, y)
    # This creates 4-point diagonal lines with slope 1
    pts_diag = []
    for y in range(4):
        for x in range(3):
            pts_diag.append([x + y, y])
    configs.append(pts_diag)

    # 4x4 minus corners
    pts = [[x, y] for x in range(4) for y in range(4)]
    for corner in [[0, 0], [3, 0], [0, 3], [3, 3]]:
        pts.remove(corner)
    configs.append(pts)

    # 5x3 grid (first 12)
    configs.append([[x, y] for x in range(5) for y in range(3)][:12])

    # Hexagonal-like: two rows with offset
    pts_hex = []
    for x in range(6):
        pts_hex.append([2 * x, 0])
    for x in range(6):
        pts_hex.append([2 * x + 1, 2])
    configs.append(pts_hex)

    # Regular hexagon vertices + doubled vertices
    R = 10**6
    hex_pts = [
        [R, 0],
        [R // 2, int(R * 0.866025)],
        [-R // 2, int(R * 0.866025)],
        [-R, 0],
        [-R // 2, -int(R * 0.866025)],
        [R // 2, -int(R * 0.866025)],
    ]
    hex_pts2 = list(hex_pts)
    for p in hex_pts:
        hex_pts2.append([2 * p[0], 2 * p[1]])
    configs.append(hex_pts2)

    # 3 points on each of 4 lines through a common point (near-pencil variant)
    # Lines: y=0, y=x, y=-x, x=0, with 3 points each
    near_pencil = []
    for d in [1, 2, 3]:
        near_pencil.append([d, 0])
        near_pencil.append([d, d])
        near_pencil.append([-d, -d])
        near_pencil.append([0, d])
    configs.append(near_pencil)

    # Random structured configs
    rng = random.Random(12345)
    for _ in range(20):
        pts = []
        for _ in range(12):
            pts.append([rng.randint(-8, 8), rng.randint(-8, 8)])
        if len(set(map(tuple, pts))) == 12:
            configs.append(pts)

    # Filter valid configs
    valid_configs = []
    for cfg in configs:
        if len(cfg) != 12:
            continue
        if len(set(map(tuple, cfg))) < 12:
            continue
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in cfg):
            continue
        # Check not all collinear
        (x0, y0), (x1, y1), (x2, y2) = cfg[0], cfg[1], cfg[2]
        all_col = True
        for i in range(3, 12):
            xi, yi = cfg[i]
            if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
                all_col = False
                break
        if all_col:
            continue
        valid_configs.append(cfg)

    return valid_configs


def _find_collinear_targets(points):
    """Find positions where a new point would be collinear with 2 existing points."""
    n = len(points)
    targets = []
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            # Points on the line: p1 + k*(dx, dy) for integer k
            # Try k = -2, -1, 1, 2, 3, -3
            for k in [-3, -2, -1, 1, 2, 3]:
                nx = p1[0] + k * dx
                ny = p1[1] + k * dy
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    # Check it's not already a point
                    if [nx, ny] not in points:
                        targets.append([nx, ny])
    return targets


def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin (evaluator kills at 150s)

    initial_configs = _make_initial_configs()

    best_points = None
    best_score = float('inf')

    for cfg in initial_configs:
        score = count_ordinary_lines(cfg)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in cfg]
        if best_score == 6:
            break

    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = count_ordinary_lines(best_points)

    # Local search with multiple strategies
    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve = 5000

    while time.time() < deadline:
        strategy = rng.random()

        if strategy < 0.4:
            # Single point random move
            idx = rng.randint(0, 11)
            old_p = best_points[idx][:]

            step_size = rng.choice([1, 2, 3, 5, 8, 12, 16, 24])
            dx = rng.randint(-step_size, step_size)
            dy = rng.randint(-step_size, step_size)
            if dx == 0 and dy == 0:
                continue

            new_p = [old_p[0] + dx, old_p[1] + dy]

            if abs(new_p[0]) > 10**9 or abs(new_p[1]) > 10**9:
                continue

            new_points = [p[:] for p in best_points]
            new_points[idx] = new_p
            if len(set(map(tuple, new_points))) < 12:
                continue

            score = count_ordinary_lines(new_points)

            if score < best_score:
                best_score = score
                best_points = new_points
                no_improve_count = 0
                if best_score == 6:
                    break
            else:
                no_improve_count += 1

        elif strategy < 0.7:
            # Move a point to a collinear target position
            targets = _find_collinear_targets(best_points)
            if targets:
                # Pick a random target and a random point to move
                idx = rng.randint(0, 11)
                target = rng.choice(targets)
                if target == best_points[idx]:
                    continue

                new_points = [p[:] for p in best_points]
                new_points[idx] = target
                if len(set(map(tuple, new_points))) < 12:
                    continue

                score = count_ordinary_lines(new_points)

                if score < best_score:
                    best_score = score
                    best_points = new_points
                    no_improve_count = 0
                    if best_score == 6:
                        break
                else:
                    no_improve_count += 1

        elif strategy < 0.85:
            # Swap two points' positions (perturb both)
            i, j = rng.sample(range(12), 2)
            p1 = best_points[i][:]
            p2 = best_points[j][:]

            step_size = rng.choice([1, 2, 3, 5, 8])
            dx1 = rng.randint(-step_size, step_size)
            dy1 = rng.randint(-step_size, step_size)
            dx2 = rng.randint(-step_size, step_size)
            dy2 = rng.randint(-step_size, step_size)

            new_p1 = [p1[0] + dx1, p1[1] + dy1]
            new_p2 = [p2[0] + dx2, p2[1] + dy2]

            if abs(new_p1[0]) > 10**9 or abs(new_p1[1]) > 10**9:
                continue
            if abs(new_p2[0]) > 10**9 or abs(new_p2[1]) > 10**9:
                continue

            new_points = [p[:] for p in best_points]
            new_points[i] = new_p1
            new_points[j] = new_p2
            if len(set(map(tuple, new_points))) < 12:
                continue

            score = count_ordinary_lines(new_points)

            if score < best_score:
                best_score = score
                best_points = new_points
                no_improve_count = 0
                if best_score == 6:
                    break
            else:
                no_improve_count += 1

        else:
            # Occasional restart from a good initial config
            if time.time() < deadline - 10:
                cfg = rng.choice(initial_configs)
                s = count_ordinary_lines(cfg)
                if s < best_score:
                    best_score = s
                    best_points = [p[:] for p in cfg]
                    no_improve_count = 0
                else:
                    no_improve_count += 1

        # If stuck for too long, try a more aggressive restart
        if no_improve_count > max_no_improve:
            # Try a random restart from a random configuration
            cfg = rng.choice(initial_configs)
            s = count_ordinary_lines(cfg)
            if s <= best_score:
                best_score = s
                best_points = [p[:] for p in cfg]
            no_improve_count = 0

    return best_points
