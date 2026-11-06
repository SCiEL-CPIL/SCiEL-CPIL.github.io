import time
import random
from collections import Counter

def _build_counter(points):
    """Build area counter for all non-zero doubled areas."""
    n = len(points)
    area_counts = Counter()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx1 = x2 - x1
            dy1 = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = points[k]
                dx2 = x3 - x1
                dy2 = y3 - y1
                area = abs(dx1 * dy2 - dy1 * dx2)
                if area > 0:
                    area_counts[area] += 1
    return area_counts

def _max_of_counter(area_counts):
    """Return max multiplicity in counter."""
    if not area_counts:
        return 0
    return max(area_counts.values())

def _calculate_score(points):
    """
    Calculate the maximum multiplicity of any non-zero doubled area.
    points: list of (x, y) tuples
    """
    area_counts = _build_counter(points)
    return _max_of_counter(area_counts)

def _get_nearby_positions(point, occupied_set, radius=4):
    """Get valid positions near a given point."""
    x, y = point
    candidates = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if -30 <= nx <= 30 and -30 <= ny <= 30:
                p = (nx, ny)
                if p not in occupied_set:
                    candidates.append(p)
    return candidates

def _evaluate_move(points, idx, new_p, area_counts):
    """
    Evaluate moving point idx to new_p.
    Returns (delta_score, new_area_counts) or (None, None) if invalid.
    """
    old_p = points[idx]
    if new_p == old_p:
        return None, None

    n = len(points)
    others = []
    for i in range(n):
        if i != idx:
            others.append(i)

    m = len(others)  # 29
    old_x, old_y = old_p
    new_x, new_y = new_p

    deltas = {}

    for i in range(m):
        a = others[i]
        xa, ya = points[a]
        for j in range(i + 1, m):
            b = others[j]
            xb, yb = points[b]
            dxa = xa - old_x
            dya = ya - old_y
            dxb = xb - old_x
            dyb = yb - old_y
            old_area = abs(dxa * dyb - dya * dxb)
            dxa_n = xa - new_x
            dya_n = ya - new_y
            dxb_n = xb - new_x
            dyb_n = yb - new_y
            new_area = abs(dxa_n * dyb_n - dya_n * dxb_n)

            if old_area > 0:
                deltas[old_area] = deltas.get(old_area, 0) - 1
            if new_area > 0:
                deltas[new_area] = deltas.get(new_area, 0) + 1

    new_counts = dict(area_counts)
    for area, delta in deltas.items():
        if delta == 0:
            continue
        cur = new_counts.get(area, 0)
        if cur + delta < 0:
            return None, None
        new_val = cur + delta
        if new_val == 0:
            del new_counts[area]
        else:
            new_counts[area] = new_val

    new_score = _max_of_counter(new_counts)
    old_score = _max_of_counter(area_counts)
    delta_score = new_score - old_score

    return delta_score, new_counts

def _local_search(initial_points, time_limit=55.0, seed=None):
    """
    Perform hill climbing to maximize the score using fast incremental updates.
    """
    if seed is not None:
        random.seed(seed)

    start_time = time.time()
    best_points = [p for p in initial_points]
    best_score = _calculate_score(best_points)
    best_counts = _build_counter(best_points)

    current_points = [p for p in initial_points]
    current_score = best_score
    current_counts = dict(best_counts)

    occupied_set = set(current_points)

    stuck_count = 0
    max_stuck = 300
    all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]

    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = current_points[idx]

        # Get nearby positions (prefer closer ones)
        nearby = _get_nearby_positions(old_p, occupied_set, radius=4)

        if not nearby:
            # Fallback to random position
            empty = [c for c in all_coords if c not in occupied_set]
            if not empty:
                break
            new_p = random.choice(empty)
        else:
            # Weight toward closer positions
            weights = [1.0 / (1.0 + sum((a - b) ** 2 for a, b in zip(old_p, p))) for p in nearby]
            new_p = random.choices(nearby, weights=weights, k=1)[0]

        # Evaluate move
        delta_score, new_counts = _evaluate_move(current_points, idx, new_p, current_counts)

        if delta_score is None:
            stuck_count += 1
            if stuck_count > max_stuck:
                current_points = [p for p in best_points]
                occupied_set = set(current_points)
                current_score = best_score
                current_counts = dict(best_counts)
                stuck_count = 0
                # Perturbation to escape local optimum
                if random.random() < 0.5:
                    for _ in range(3):
                        ridx = random.randint(0, 29)
                        old_rp = current_points[ridx]
                        nearby_r = _get_nearby_positions(old_rp, occupied_set, radius=5)
                        if nearby_r:
                            new_rp = random.choice(nearby_r)
                            current_points[ridx] = new_rp
                            occupied_set.remove(old_rp)
                            occupied_set.add(new_rp)
                    current_counts = _build_counter(current_points)
                    current_score = _max_of_counter(current_counts)
            continue

        # Apply move
        current_points[idx] = new_p
        occupied_set.remove(old_p)
        occupied_set.add(new_p)
        current_counts = new_counts
        current_score += delta_score

        if delta_score > 0:
            stuck_count = 0
            if current_score > best_score:
                best_score = current_score
                best_points = [p for p in current_points]
                best_counts = dict(current_counts)
        else:
            # Revert move
            current_points[idx] = old_p
            occupied_set.remove(new_p)
            occupied_set.add(old_p)
            # Revert counts by recomputing (expensive but correct)
            # Actually, let's just recompute current_counts
            current_counts = _build_counter(current_points)
            current_score = _max_of_counter(current_counts)
            stuck_count += 1
            if stuck_count > max_stuck:
                current_points = [p for p in best_points]
                occupied_set = set(current_points)
                current_score = best_score
                current_counts = dict(best_counts)
                stuck_count = 0
                if random.random() < 0.5:
                    for _ in range(3):
                        ridx = random.randint(0, 29)
                        old_rp = current_points[ridx]
                        nearby_r = _get_nearby_positions(old_rp, occupied_set, radius=5)
                        if nearby_r:
                            new_rp = random.choice(nearby_r)
                            current_points[ridx] = new_rp
                            occupied_set.remove(old_rp)
                            occupied_set.add(new_rp)
                    current_counts = _build_counter(current_points)
                    current_score = _max_of_counter(current_counts)

    return best_points, best_score

def _hexagonal_lattice(center_x, center_y, rows, cols):
    """Generate a hexagonal lattice of points."""
    points = []
    for i in range(rows):
        for j in range(cols):
            x = center_x + j
            y = center_y + i
            if i % 2 == 1:
                x += 1
            points.append((x, y))
    return points

def construct():
    # Initial guesses
    grid_5x6 = [(x, y) for x in range(5) for y in range(6)]
    grid_6x5 = [(x, y) for x in range(6) for y in range(5)]
    grid_3x10 = [(x, y) for x in range(3) for y in range(10)]
    grid_2x15 = [(x, y) for x in range(2) for y in range(15)]
    hex_lattice = _hexagonal_lattice(0, 0, 5, 6)

    random.seed(42)
    random_points = random.sample([(x, y) for x in range(-30, 31) for y in range(-30, 31)], 30)

    candidates = [grid_5x6, grid_6x5, grid_3x10, grid_2x15, hex_lattice, random_points]

    best_overall_points = None
    best_overall_score = -1

    for c in candidates:
        s = _calculate_score(c)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = c

    # Run local search from the best candidate
    final_points, final_score = _local_search(best_overall_points, time_limit=50.0, seed=42)

    if final_score > best_overall_score:
        return final_points
    else:
        return best_overall_points
