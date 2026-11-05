"""Optimized program for maximizing equal-area triangle multiplicity."""
import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 75

    def max_mult(pts):
        cnt = Counter()
        n = len(pts)
        for i in range(n):
            xi, yi = pts[i]
            for j in range(i + 1, n):
                xj, yj = pts[j]
                for k in range(j + 1, n):
                    xk, yk = pts[k]
                    a = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                    if a > 0:
                        cnt[a] += 1
        return max(cnt.values()) if cnt else 0

    # Initial: 6x5 grid
    best_points = [(x, y) for x in range(6) for y in range(5)]
    best_score = max_mult(best_points)

    # Try alternative structured configs
    alts = [
        [(x, y) for x in range(5) for y in range(6)],
        [(x, y) for x in range(10) for y in range(3)][:30],
        [(x, y) for x in range(3) for y in range(10)][:30],
        [(x, y) for x in range(15) for y in range(2)][:30],
        [(x, y) for x in range(2) for y in range(15)][:30],
        [(x, y) for x in range(12) for y in range(3)][:30],
        [(x, y) for x in range(3) for y in range(12)][:30],
        [(x, y) for x in range(8) for y in range(4)][:30],
        [(x, y) for x in range(4) for y in range(8)][:30],
    ]

    for grid in alts:
        if time.time() > deadline:
            break
        if len(set(grid)) == 30 and all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in grid):
            s = max_mult(grid)
            if s > best_score:
                best_score = s
                best_points = grid

    # Local search
    rng = random.Random(42)
    points = [tuple(p) for p in best_points]
    current_score = best_score
    point_set = set(points)

    while time.time() < deadline:
        idx = rng.randint(0, 29)
        old_x, old_y = points[idx]

        found = False
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = old_x + dx, old_y + dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                if (nx, ny) in point_set:
                    continue

                points[idx] = (nx, ny)
                point_set.discard((old_x, old_y))
                point_set.add((nx, ny))
                new_score = max_mult(points)

                if new_score > current_score:
                    current_score = new_score
                    found = True
                    break
                else:
                    points[idx] = (old_x, old_y)
                    point_set.discard((nx, ny))
                    point_set.add((old_x, old_y))
            if found:
                break

        if not found and rng.random() < 0.05:
            nx = old_x + rng.randint(-5, 5)
            ny = old_y + rng.randint(-5, 5)
            if -30 <= nx <= 30 and -30 <= ny <= 30 and (nx, ny) not in point_set:
                points[idx] = (nx, ny)
                point_set.discard((old_x, old_y))
                point_set.add((nx, ny))
                new_score = max_mult(points)
                if new_score > current_score:
                    current_score = new_score
                    found = True
                else:
                    points[idx] = (old_x, old_y)
                    point_set.discard((nx, ny))
                    point_set.add((old_x, old_y))

        if current_score > best_score:
            best_score = current_score
            best_points = [tuple(p) for p in points]

    return best_points
