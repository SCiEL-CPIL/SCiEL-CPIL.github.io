"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # 150s wall clock, safety margin

    def elapsed():
        return time.time() - start_time

    def is_perfect_square(n):
        """Exact check if n is a perfect square (n >= 0)."""
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def check_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def _exact_det_4x4(p1, p2, p3, p4):
        rows = []
        for p in [p1, p2, p3, p4]:
            x, y = p
            rows.append([x, y, x * x + y * y, 1])

        def det3(m):
            return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

        def det4(m):
            result = 0
            signs = [1, -1, 1, -1]
            for j in range(4):
                minor = [[m[i][k] for k in range(4) if k != j] for i in range(1, 4)]
                result += signs[j] * m[0][j] * det3(minor)
            return result

        return det4(rows)

    def check_concyclic(p1, p2, p3, p4):
        return _exact_det_4x4(p1, p2, p3, p4) == 0

    def all_constraints_satisfied(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                d2 = dx * dx + dy * dy
                if not is_perfect_square(d2):
                    return False
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        for combo in combinations(range(n), 4):
            if check_concyclic(points[combo[0]], points[combo[1]], points[combo[2]], points[combo[3]]):
                return False
        return True

    def try_add_point(current_points, candidate):
        new_points = current_points + [candidate]
        if all_constraints_satisfied(new_points):
            return new_points
        return None

    # Known 5-point integral point set
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    # Generate candidates using Pythagorean triples for guaranteed integer distances from origin
    # A point (x,y) has integer distance from origin iff x^2+y^2 is a perfect square
    max_coord = 300
    candidates = []
    seen = set(best_points)

    # Add all integer points with integer distance to origin (Pythagorean pairs)
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            if (x, y) in seen:
                continue
            d2 = x * x + y * y
            if is_perfect_square(d2):
                candidates.append((x, y))
                seen.add((x, y))

    # Also add some random integer points (may have integer distances to other points)
    import random
    random.seed(42)
    for _ in range(5000):
        x = random.randint(-max_coord, max_coord)
        y = random.randint(-max_coord, max_coord)
        if (x, y) not in seen:
            candidates.append((x, y))
            seen.add((x, y))

    # Sort by distance from origin
    candidates.sort(key=lambda p: p[0] ** 2 + p[1] ** 2)

    # Greedy: try to add points one by one
    improved = True
    while improved:
        improved = False
        for idx in range(len(candidates)):
            if elapsed() > DEADLINE:
                break
            cand = candidates[idx]
            result = try_add_point(best_points, cand)
            if result is not None:
                best_points = result
                candidates.pop(idx)
                improved = True
                if len(best_points) >= 12:
                    break
        if len(best_points) >= 12:
            break

    # Try multiple starting configurations
    initial_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (3, 4), (4, 0), (0, 4)],
        [(0, 0), (5, 0), (0, 12), (5, 12)],
        [(0, 0), (9, 12), (12, 9), (15, 0)],
        [(0, 0), (8, 15), (15, 8), (17, 0)],
    ]

    for init in initial_sets:
        if elapsed() > DEADLINE:
            break
        if not all_constraints_satisfied(init):
            continue
        pts = list(init)
        # Build fresh candidate list for this start
        seen2 = set(pts)
        cands2 = []
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                if (x, y) in seen2:
                    continue
                d2 = x * x + y * y
                if is_perfect_square(d2):
                    cands2.append((x, y))
                    seen2.add((x, y))
        cands2.sort(key=lambda p: p[0] ** 2 + p[1] ** 2)

        improved2 = True
        while improved2:
            improved2 = False
            for idx in range(len(cands2)):
                if elapsed() > DEADLINE:
                    break
                cand = cands2[idx]
                result = try_add_point(pts, cand)
                if result is not None:
                    pts = result
                    cands2.pop(idx)
                    improved2 = True
                    if len(pts) >= 12:
                        break
            if len(pts) >= 12:
                break

        if len(pts) > len(best_points):
            best_points = pts

    # Final: try random perturbation hill-climbing on best_points
    if elapsed() < DEADLINE - 30:
        import random
        random.seed(123)
        for trial in range(200):
            if elapsed() > DEADLINE:
                break
            # Start from best and try adding random candidates
            pts = list(best_points)
            for _ in range(50):
                if elapsed() > DEADLINE:
                    break
                x = random.randint(-max_coord, max_coord)
                y = random.randint(-max_coord, max_coord)
                cand = (x, y)
                if cand in pts:
                    continue
                result = try_add_point(pts, cand)
                if result is not None:
                    pts = result
                    if len(pts) > len(best_points):
                        best_points = pts

    # Convert to required format
    D = 1
    points_out = [[[x, 1], [y, 1]] for x, y in best_points]

    return {'D': D, 'points': points_out}
# EVOLVE-BLOCK-END
