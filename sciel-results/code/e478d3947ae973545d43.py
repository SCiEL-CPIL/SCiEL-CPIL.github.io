"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from math import sqrt

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280  # keep safety margin before 300s limit

    # Precompute all grid points
    all_points = [(x, y) for y in range(N) for x in range(N)]

    def dist2(p, q):
        return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

    def is_isosceles(a, b, c):
        d1 = dist2(a, b)
        d2 = dist2(b, c)
        d3 = dist2(a, c)
        # An isosceles triangle has two equal sides.
        # Degenerate cases (three collinear points) with two equal distances
        # are also excluded since they form a degenerate isosceles triangle.
        return d1 == d2 or d1 == d3 or d2 == d3

    best_points = []
    best_size = 0

    # Strategy 1: Greedy construction with random ordering
    for attempt in range(200):
        if time.time() > deadline:
            break
        random.seed(attempt * 1000 + 42)
        candidates = all_points[:]
        random.shuffle(candidates)

        selected = []
        selected_set = set()

        for p in candidates:
            # Check against all existing points
            valid = True
            for q in selected:
                if is_isosceles(p, q, selected[0]):
                    # This is O(n^2) per check which is too slow.
                    # We need a smarter approach.
                    pass
            # This naive approach is too slow. Let's use a different method.
            break

    # Strategy 2: Use a structured approach - points on a parabola y = x^2 mod N
    # This ensures no three points form an isosceles triangle in many cases
    # because the geometry of the parabola avoids symmetric configurations.

    # Actually, let's use a known construction:
    # Points (i, i^2) for i in a Sidon-like sequence
    # But we need to ensure they're in [0, N)^2

    # Let's try: for each x in [0, N), place point at (x, f(x)) where f is chosen
    # to avoid isosceles triangles.

    # A better approach: use the fact that on a parabola, three points form
    # an isosceles triangle iff they are symmetric about the axis.
    # So if we pick points with distinct x-coordinates on a parabola,
    # we need to avoid any three that are symmetric.

    # Actually, let's use a simpler structural approach:
    # Pick points greedily, but with efficient conflict checking.

    # Precompute distance squared between all pairs
    n_points = N * N
    dist2_matrix = [[0] * n_points for _ in range(n_points)]
    for i in range(n_points):
        for j in range(i + 1, n_points):
            d = dist2(all_points[i], all_points[j])
            dist2_matrix[i][j] = d
            dist2_matrix[j][i] = d

    # For each point, track which other points would form an isosceles triangle
    # with it and any third point. This is complex, so let's use a greedy approach
    # with efficient checking.

    # Greedy with bitsets for efficiency
    # For each candidate point p, it conflicts with set S if there exist a, b in S
    # such that {p, a, b} is isosceles.

    # Let's use a different structural insight:
    # Points on the line y = 0 with x-coordinates forming a Sidon set have all
    # pairwise distances distinct, so no isosceles triangles. But we got score 1.0.
    # We need MORE points.

    # New approach: 2D construction using modular arithmetic.
    # Place points at (i, i^2 mod N) for i in [0, N).
    # Check if this avoids isosceles triangles.

    points_parabola = [(i, (i * i) % N) for i in range(N)]

    # Verify
    valid = True
    for i in range(len(points_parabola)):
        for j in range(i + 1, len(points_parabola)):
            for k in range(j + 1, len(points_parabola)):
                a, b, c = points_parabola[i], points_parabola[j], points_parabola[k]
                if is_isosceles(a, b, c):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        return points_parabola

    # If parabola doesn't work, try a greedy approach with time management
    best_points = []
    best_size = 0

    for attempt in range(50):
        if time.time() > deadline:
            break
        random.seed(attempt * 7 + 13)
        candidates = all_points[:]
        random.shuffle(candidates)

        selected = []
        # For efficient checking, maintain for each existing point the set of
        # distance-squareds to other selected points

        for p in candidates:
            pi = all_points.index(p)
            can_add = True
            for q in selected:
                qi = all_points.index(q)
                # Check if adding p creates an isosceles triangle with any
                # two existing points r, s
                # We need to check: for all pairs (r, s) in selected, is {p, r, s} isosceles?
                # This is O(n^2) per candidate, too slow.

            # Use a smarter check: p conflicts with selected if there exist
            # a, b in selected such that dist2(p,a) == dist2(p,b) or
            # dist2(p,a) == dist2(a,b) or dist2(p,b) == dist2(a,b)

            # For fixed p and existing set, precompute dist2(p, x) for all x in selected
            dists_from_p = {}
            for q in selected:
                qi = all_points.index(q)
                d = dist2_matrix[pi][qi]
                dists_from_p.setdefault(d, []).append(q)

            # Check condition 1: two points at same distance from p
            for d, pts_at_d in dists_from_p.items():
                if len(pts_at_d) >= 2:
                    can_add = False
                    break
            if not can_add:
                continue

            # Check condition 2: dist2(p, a) == dist2(a, b) for some a, b in selected
            for a in selected:
                ai = all_points.index(a)
                da = dist2_matrix[pi][ai]
                for b in selected:
                    bi = all_points.index(b)
                    ab = dist2_matrix[ai][bi]
                    if da == ab:
                        can_add = False
                        break
                if not can_add:
                    break
            if not can_add:
                continue

            selected.append(p)

        if len(selected) > best_size:
            best_size = len(selected)
            best_points = selected[:]

    if len(best_points) > len(points_parabola):
        return best_points
    else:
        return points_parabola if len(points_parabola) > 0 else best_points

# EVOLVE-BLOCK-END
