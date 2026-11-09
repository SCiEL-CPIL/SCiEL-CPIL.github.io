"""Combined program: structured candidates + incremental scoring + target-area local search."""
import time
import itertools
from collections import Counter
import numpy as np


def _doubled_areas(points):
    """Compute all positive doubled areas for all triples."""
    n = len(points)
    areas = []
    for i in range(n):
        ax, ay = points[i]
        for j in range(i + 1, n):
            bx, by = points[j]
            dx1 = bx - ax
            dy1 = by - ay
            for k in range(j + 1, n):
                cx, cy = points[k]
                dx2 = cx - ax
                dy2 = cy - ay
                det = abs(dx1 * dy2 - dy1 * dx2)
                if det > 0:
                    areas.append(det)
    return areas


def _area_counter(points):
    """Return Counter of positive doubled areas."""
    return Counter(_doubled_areas(points))


def _max_multiplicity(points):
    """Return the maximum count of any single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    return Counter(areas).most_common(1)[0][1]


def _valid(points):
    """Check 30 distinct integer points in [-30,30]^2."""
    if len(points) != 30:
        return False
    if len(set(points)) != 30:
        return False
    for x, y in points:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
    return True


def _score(points):
    if not _valid(points):
        return 0
    return _max_multiplicity(points)


def _fast_score(points, prev_counter, moved_idx, old_point, new_point):
    """
    Fast scoring: if we only moved one point, update the area counter
    by removing areas involving the old point and adding areas with the new point.
    Returns (max_multiplicity, area_counter).
    """
    counter = Counter(prev_counter)
    n = len(points)
    idx = moved_idx

    # Remove old contributions (triples containing idx with old_point)
    ox, oy = old_point
    for i in range(n):
        if i == idx:
            continue
        px, py = points[i]
        for j in range(i + 1, n):
            if j == idx:
                continue
            qx, qy = points[j]
            det = abs((px - ox) * (qy - oy) - (py - oy) * (qx - ox))
            if det > 0:
                counter[det] -= 1
                if counter[det] <= 0:
                    del counter[det]

    # Add new contributions (triples containing idx with new_point)
    nx, ny = new_point
    for i in range(n):
        if i == idx:
            continue
        px, py = points[i]
        for j in range(i + 1, n):
            if j == idx:
                continue
            qx, qy = points[j]
            det = abs((px - nx) * (qy - ny) - (py - ny) * (qx - nx))
            if det > 0:
                counter[det] += 1

    if not counter:
        return 0, counter
    return max(counter.values()), counter


def _structured_candidates():
    """Generate various structured grid-like candidates."""
    candidates = []

    # Standard grids: m x n with m*n = 30
    for m, n in [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15)]:
        pts = []
        for i in range(m):
            for j in range(n):
                x = i - m // 2
                y = j - n // 2
                pts.append((x, y))
        if _valid(pts):
            candidates.append(pts)

    # Grids with different spacings
    for m, n in [(6, 5), (5, 6), (10, 3)]:
        for sx in [1, 2, 3]:
            for sy in [1, 2, 3]:
                pts = []
                for i in range(m):
                    for j in range(n):
                        x = i * sx - (m * sx) // 2
                        y = j * sy - (n * sy) // 2
                        pts.append((x, y))
                if _valid(pts):
                    candidates.append(pts)

    # Interleaved: points on lines
    for offset in range(0, 5):
        pts = []
        for line in range(3):
            for k in range(10):
                x = k * 6 - 30 + offset
                y = line * 10 - 10
                pts.append((x, y))
        if _valid(pts):
            candidates.append(pts)

    # Staggered rows (hexagonal-ish)
    for dx in [1, 2, 3]:
        for dy in [1, 2, 3]:
            pts = []
            for row in range(5):
                for col in range(6):
                    x = col * dx + (row % 2) * (dx // 2 + 1) - 15
                    y = row * dy - 10
                    pts.append((x, y))
            if _valid(pts):
                candidates.append(pts)

    # Arithmetic progression in both directions
    for step_x in [1, 2, 3]:
        for step_y in [1, 2, 3]:
            for start_x in range(-20, 20, 2):
                for start_y in range(-20, 20, 2):
                    pts = []
                    for i in range(30):
                        row = i // 6
                        col = i % 6
                        x = start_x + col * step_x
                        y = start_y + row * step_y
                        if -30 <= x <= 30 and -30 <= y <= 30:
                            pts.append((x, y))
                    if len(pts) == 30 and _valid(pts):
                        candidates.append(pts)

    return candidates


def _local_search(points, max_iterations=5000, time_limit=60.0):
    """Hill-climbing local search with incremental scoring and target-area focus."""
    start_time = time.time()
    best_points = list(points)
    best_score = _score(best_points)
    best_counter = _area_counter(best_points)

    if best_score == 0:
        return best_points

    target_area = best_counter.most_common(1)[0][0]

    occupied = set(best_points)

    # Neighborhood sizes to try in sequence
    neighborhoods = [
        [(dx, dy) for dx in range(-4, 5) for dy in range(-4, 5) if (dx, dy) != (0, 0)],
        [(dx, dy) for dx in range(-8, 9) for dy in range(-8, 9) if (dx, dy) != (0, 0)],
        [(dx, dy) for dx in range(-15, 16) for dy in range(-15, 16) if (dx, dy) != (0, 0)],
    ]

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit - 3:
            break

        improved = False
        nh_size = iteration % 3
        if time.time() - start_time > time_limit - 10:
            nh_size = min(nh_size, 1)

        for idx in range(30):
            if time.time() - start_time > time_limit - 3:
                break

            px, py = best_points[idx]
            for dx, dy in neighborhoods[nh_size]:
                nx, ny = px + dx, py + dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue

                new_point = (nx, ny)
                if new_point in occupied:
                    continue

                new_points = list(best_points)
                new_points[idx] = new_point
                s, new_counter = _fast_score(new_points, best_counter, idx, (px, py), new_point)

                if s > best_score:
                    best_score = s
                    best_points = new_points
                    best_counter = new_counter
                    occupied.discard((px, py))
                    occupied.add(new_point)
                    target_area = best_counter.most_common(1)[0][0]
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # Try a larger random restart within the current neighborhood
            for _ in range(10):
                if time.time() - start_time > time_limit - 3:
                    break
                idx = np.random.randint(0, 30)
                px, py = best_points[idx]
                dx = np.random.randint(-15, 16)
                dy = np.random.randint(-15, 16)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = px + dx, py + dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                new_point = (nx, ny)
                if new_point in occupied:
                    continue
                new_points = list(best_points)
                new_points[idx] = new_point
                s, new_counter = _fast_score(new_points, best_counter, idx, (px, py), new_point)
                if s > best_score:
                    best_score = s
                    best_points = new_points
                    best_counter = new_counter
                    occupied.discard((px, py))
                    occupied.add(new_point)
                    target_area = best_counter.most_common(1)[0][0]
                    improved = True
                    break
            if not improved:
                break

    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s

    # Generate structured candidates
    candidates = _structured_candidates()

    if not candidates:
        # Fallback: 6x5 grid
        return [(x, y) for x in range(6) for y in range(5)]

    # Score all candidates
    best_points = None
    best_score = 0
    for cand in candidates:
        if time.time() > deadline:
            break
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)

    if best_points is None:
        best_points = list(candidates[0])
        best_score = _score(best_points)

    # Local search from best candidate
    if time.time() < deadline - 10:
        result = _local_search(best_points, max_iterations=3000, time_limit=deadline - start_time)
        rs = _score(result)
        if rs > best_score:
            best_points = result
            best_score = rs

    return best_points
