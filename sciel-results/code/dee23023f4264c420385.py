"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START

def _are_collinear(p1, p2, p3):
    return (p1[0] - p2[0]) * (p2[1] - p3[1]) == (p2[0] - p3[0]) * (p1[1] - p2[1])

def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                if _are_collinear(points[i], points[j], points[k]):
                    count += 1
                    if count >= 4:
                        return True
    return False

def _calculate_score(points):
    """Exact branch and bound for largest subset with no 3 collinear."""
    n = len(points)
    pair_conflicts = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mask = 0
            for k in range(n):
                if k != i and k != j:
                    if _are_collinear(points[i], points[j], points[k]):
                        mask |= (1 << k)
            pair_conflicts[i][j] = mask
            pair_conflicts[j][i] = mask

    best_score = 0
    full_mask = (1 << n) - 1

    def backtrack(available, current_size, selected_mask):
        nonlocal best_score
        if current_size + bin(available).count('1') <= best_score:
            return
        if available == 0:
            if current_size > best_score:
                best_score = current_size
            return
        i_bit = available & -available
        i = i_bit.bit_length() - 1
        # Skip i
        backtrack(available ^ i_bit, current_size, selected_mask)
        # Include i
        new_selected = selected_mask | i_bit
        new_available = available ^ i_bit
        forbidden = 0
        temp_sel = selected_mask
        while temp_sel:
            k_bit = temp_sel & -temp_sel
            k = k_bit.bit_length() - 1
            forbidden |= pair_conflicts[i][k]
            temp_sel ^= k_bit
        new_available &= ~forbidden
        backtrack(new_available, current_size + 1, new_selected)

    backtrack(full_mask, 0, 0)
    return best_score

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def generate_random_points():
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        return points

    def generate_parabola_points():
        return [[x, x * x] for x in range(-10, 10)]

    best_points = None
    best_score = 21

    # Phase 1: Random restarts to find a valid starting point
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        points = generate_random_points()
        if not _has_four_collinear(points):
            score = _calculate_score(points)
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in points]

    if best_points is None:
        best_points = generate_parabola_points()
        best_score = 20

    # Phase 2: Local search with line-snapping moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(20))
        random.shuffle(indices)

        for i in indices:
            if time.time() > deadline:
                break
            old_point = best_points[i][:]

            # Generate candidate new positions
            candidates = set()

            # Random positions
            for _ in range(15):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                candidates.add((x, y))

            # Line-snapping: try to place on lines defined by pairs of other points
            for j in range(20):
                if j == i:
                    continue
                for k in range(j + 1, 20):
                    if k == i:
                        continue
                    # Line through points[j] and points[k]
                    x1, y1 = best_points[j]
                    x2, y2 = best_points[k]
                    dx = x2 - x1
                    dy = y2 - y1
                    # Find integer points on this line within [-100, 100]
                    # Parametric: (x1 + t*dx, y1 + t*dy) for integer t
                    # We need |x1 + t*dx| <= 100 and |y1 + t*dy| <= 100
                    if dx == 0 and dy == 0:
                        continue
                    if dx == 0:
                        # Vertical line at x = x1
                        if abs(x1) > 100:
                            continue
                        for t in range(-100, 101):
                            y_val = y1 + t * dy
                            if abs(y_val) <= 100 and (x1, y_val) not in [tuple(p) for p in best_points if p is not best_points[i]]:
                                candidates.add((x1, y_val))
                    elif dy == 0:
                        # Horizontal line at y = y1
                        if abs(y1) > 100:
                            continue
                        for t in range(-100, 101):
                            x_val = x1 + t * dx
                            if abs(x_val) <= 100 and (x_val, y1) not in [tuple(p) for p in best_points if p is not best_points[i]]:
                                candidates.add((x_val, y1))
                    else:
                        # General line: step through t values
                        # gcd to get primitive direction
                        g = math.gcd(abs(dx), abs(dy))
                        pdx = dx // g
                        pdy = dy // g
                        # t range: |x1 + t*pdx| <= 100 and |y1 + t*pdy| <= 100
                        t_min = max(math.ceil((-100 - x1) / pdx) if pdx > 0 else math.ceil((100 - x1) / pdx) if pdx < 0 else -10**9,
                                    math.ceil((-100 - y1) / pdy) if pdy > 0 else math.ceil((100 - y1) / pdy) if pdy < 0 else -10**9)
                        t_max = min(math.floor((100 - x1) / pdx) if pdx > 0 else math.floor((-100 - x1) / pdx) if pdx < 0 else 10**9,
                                    math.floor((100 - y1) / pdy) if pdy > 0 else math.floor((-100 - y1) / pdy) if pdy < 0 else 10**9)
                        # Limit t range to keep candidates manageable
                        t_min = max(t_min, -50)
                        t_max = min(t_max, 50)
                        for t in range(int(t_min), int(t_max) + 1):
                            x_val = x1 + t * pdx
                            y_val = y1 + t * pdy
                            if abs(x_val) <= 100 and abs(y_val) <= 100:
                                if (x_val, y_val) not in [tuple(p) for p in best_points if p is not best_points[i]]:
                                    candidates.add((x_val, y_val))

            # Try each candidate
            for cx, cy in candidates:
                if time.time() > deadline:
                    break
                if (cx, cy) == (old_point[0], old_point[1]):
                    continue
                if (cx, cy) in [tuple(p) for p in best_points]:
                    continue

                test_points = best_points[:]
                test_points[i] = [cx, cy]

                if _has_four_collinear(test_points):
                    continue

                score = _calculate_score(test_points)
                if score < best_score:
                    best_score = score
                    best_points = test_points
                    improved = True
                    break  # Restart the loop with the new best

    return [p[:] for p in best_points]

# EVOLVE-BLOCK-END
