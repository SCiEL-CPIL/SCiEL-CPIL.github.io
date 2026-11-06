"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Distance squared on triangular lattice
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def count_edges(points):
        point_set = set(points)
        count = 0
        for a, b in points:
            if (a+1, b) in point_set: count += 1
            if (a, b+1) in point_set: count += 1
            if (a-1, b+1) in point_set: count += 1
            if (a-1, b) in point_set: count += 1
            if (a, b-1) in point_set: count += 1
            if (a+1, b-1) in point_set: count += 1
        return count // 2

    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a+b)) // 2

    def angle_of(a, b):
        x = a + b * 0.5
        y = b * (math.sqrt(3) / 2.0)
        return math.atan2(y, x)

    # Generate the hexagonal spiral / compact cluster
    # Radius 2 hexagon has 19 points, radius 3 adds 18 more (total 37)
    # We need 32 points.
    inner = []
    outer = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            d = hex_dist(a, b)
            if d <= 2:
                inner.append((a, b))
            elif d == 3:
                outer.append((a, b))

    # Sort outer ring by angle to get perimeter ordering
    outer.sort(key=lambda p: angle_of(p[0], p[1]))

    n_inner = len(inner)
    n_outer_needed = 32 - n_inner  # 13

    # Evaluate all contiguous blocks of 13 from the outer ring
    best_score = -1
    best_points = None
    n_out = len(outer)

    for start in range(n_out):
        block = [outer[(start + i) % n_out] for i in range(n_outer_needed)]
        candidate = inner + block
        score = count_edges(candidate)
        if score > best_score:
            best_score = score
            best_points = candidate

    if best_score == 76:
        return [[p[0], p[1]] for p in best_points]

    # Fallback: Local search if the heuristic didn't hit 76
    # (Unlikely, but safe)
    if best_points is None:
        best_points = inner + outer[:n_outer_needed]
        best_score = count_edges(best_points)

    current_points = best_points[:]
    current_set = set(current_points)
    current_score = best_score

    offsets = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    stall_counter = 0

    while time.time() < deadline - 5:
        i = random.randint(0, 31)
        a_i, b_i = current_points[i]

        found_move = False
        # Try random moves
        for _ in range(20):
            a_new = a_i + random.randint(-2, 2)
            b_new = b_i + random.randint(-2, 2)

            if (a_new, b_new) == (a_i, b_i):
                continue
            if (a_new, b_new) in current_set:
                continue
            if abs(a_new) > 64 or abs(b_new) > 64:
                continue

            # Perform move
            current_set.remove((a_i, b_i))
            current_set.add((a_new, b_new))
            current_points[i] = (a_new, b_new)

            new_score = count_edges(current_points)

            if new_score >= current_score:
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points[:]
                        if best_score == 76:
                            return [[p[0], p[1]] for p in best_points]
                    stall_counter = 0
                else:
                    if random.random() < 0.1:
                        stall_counter = 0
                    else:
                        current_set.remove((a_new, b_new))
                        current_set.add((a_i, b_i))
                        current_points[i] = (a_i, b_i)
                        stall_counter += 1
                found_move = True
                break
            else:
                current_set.remove((a_new, b_new))
                current_set.add((a_i, b_i))
                current_points[i] = (a_i, b_i)

        if not found_move:
            stall_counter += 1

        if stall_counter > 100:
            current_points = best_points[:]
            current_set = set(current_points)
            current_score = best_score
            stall_counter = 0

    return [[p[0], p[1]] for p in best_points]
# EVOLVE-BLOCK-END
