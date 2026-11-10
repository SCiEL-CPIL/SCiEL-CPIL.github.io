# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 using greedy exclusion.
    Start with N = F_49^2, then greedily exclude points while preserving
    the Nikodym property. Track coverage counts for efficiency.
    """
    start = time.time()
    deadline = start + 600  # safety margin

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    total_points = q * q  # 2401

    # Field operations via lookup tables
    # Element index: idx = a + b*p, representing a + b*alpha
    add_table = np.zeros((q, q), dtype=np.int32)
    mul_table = np.zeros((q, q), dtype=np.int32)

    for a1 in range(p):
        for b1 in range(p):
            idx1 = a1 + b1 * p
            for a2 in range(p):
                for b2 in range(p):
                    idx2 = a2 + b2 * p
                    add_table[idx1, idx2] = ((a1 + a2) % p) + ((b1 + b2) % p) * p
                    ra = (a1 * a2 + b1 * b2 * k) % p
                    rb = (a1 * b2 + a2 * b1) % p
                    mul_table[idx1, idx2] = ra + rb * p

    # Precompute: for each point index (0..2400), store (x_idx, y_idx)
    # Point index: pt = x_idx * q + y_idx
    pt_x = np.zeros(total_points, dtype=np.int32)
    pt_y = np.zeros(total_points, dtype=np.int32)
    for x_idx in range(q):
        for y_idx in range(q):
            pt = x_idx * q + y_idx
            pt_x[pt] = x_idx
            pt_y[pt] = y_idx

    # Precompute for each direction (50 directions: m=0..48 for v=(1,m), and m=49 for v=(0,1)):
    # For each point, which other points are on the same line in that direction.
    # We'll precompute line assignments: for each direction d, each point pt gets a line_id.
    # Then for each direction, for each line, we have the list of points on it.

    num_dirs = q + 1  # 50 directions

    # For each direction d, for each point pt, compute line_id
    # For v=(1,m): line through (x0,y0) has equation y - m*x = c
    #   c = y0 - m*x0, line_id = c_idx
    # For v=(0,1): line is x = x0, line_id = x0

    # Precompute line_id[direction, point]
    line_id = np.zeros((num_dirs, total_points), dtype=np.int32)

    for m in range(q):
        # v = (1, m)
        for x_idx in range(q):
            for y_idx in range(q):
                pt = x_idx * q + y_idx
                # c = y - m*x
                mx = mul_table[m, x_idx]
                mx_a = mx // p
                mx_b = mx % p
                neg_mx_a = (-mx_a) % p
                neg_mx_b = (-mx_b) % p
                y_a = y_idx // p
                y_b = y_idx % p
                c_a = (y_a + neg_mx_a) % p
                c_b = (y_b + neg_mx_b) % p
                c_idx = c_a + c_b * p
                line_id[m, pt] = c_idx

    # v = (0, 1): line_id = x_idx
    for x_idx in range(q):
        for y_idx in range(q):
            pt = x_idx * q + y_idx
            line_id[q, pt] = x_idx

    # For each direction d and each line l, precompute the list of points on that line
    # line_points[d, l] = array of point indices
    # We'll use a different approach: for each direction, group points by line_id

    # For efficiency in the greedy algorithm, we need:
    # For each point pt, for each direction d, the other points on the same line.
    # Let's precompute: line_members[d, l] as a list/array of points.

    # Actually, for the greedy approach, let's think differently.
    # 
    # A point pt is "coverable" by direction d if all other points on the line
    # through pt in direction d are in N (not excluded).
    # 
    # Let covered[pt] = number of directions d for which pt is coverable.
    # pt can be excluded if covered[pt] >= 1.
    #
    # When we exclude a point e, for each direction d and each other point pt on the
    # same line as e in direction d, pt loses coverability in direction d (if e was
    # the only excluded point... no wait).
    #
    # Actually: pt is coverable by direction d iff NO excluded point is on the line
    # through pt in direction d (other than pt itself).
    #
    # So if we exclude point e, then for each direction d, for each point pt on the
    # same line as e in direction d (pt != e), pt becomes NOT coverable by direction d
    # (because e is now excluded and on that line).
    #
    # So: excluding e decreases covered[pt] by 1 for each (d, pt) where pt is on the
    # same line as e in direction d and pt != e.
    #
    # Let's precompute for each point e, the list of (d, pt) pairs that would be affected.

    # For each point e, for each direction d, find all other points on the same line.
    # affected[e] = list of (d, pt) where pt != e and pt is on same line as e in dir d

    # This is: for each e, for each d, the line through e in dir d has q points.
    # The other q-1 = 48 points are affected.
    # Total: 2401 * 50 * 48 ≈ 5.7 million entries. That's a lot but manageable.

    # Instead, let's compute it on the fly or use a smarter representation.

    # Alternative: for each direction d, for each line l, store the set of points.
    # Then for point e, for each direction d, the affected points are line_points[d, line_id[d,e]] \ {e}.

    # Let's precompute line_points as a list of arrays.
    # line_points[d, l] = np.array of point indices on line l in direction d.

    line_points = [None] * num_dirs
    for d in range(num_dirs):
        # Group points by line_id[d, pt]
        # line_id[d, pt] ranges over 0..q-1 (q lines per direction)
        groups = [[] for _ in range(q)]
        for pt in range(total_points):
            lid = line_id[d, pt]
            groups[lid].append(pt)
        line_points[d] = [np.array(g, dtype=np.int32) for g in groups]

    # Now, greedy exclusion:
    # covered[pt] = number of directions where pt is fully covered (no excluded on line except self)
    # Initially, covered[pt] = num_dirs = 50 for all pt (no points excluded yet).

    excluded = np.zeros(total_points, dtype=bool)
    covered = np.full(total_points, num_dirs, dtype=np.int32)

    # Greedy: repeatedly exclude a point with covered[pt] >= 1 that minimizes
    # the damage (i.e., causes the fewest points to drop below coverage 1).
    # Simple heuristic: exclude the point whose exclusion causes the least decrease
    # in the minimum coverage, or just exclude any point with covered >= 1.

    # For speed, let's just do a simple greedy: in each round, find a point with
    # covered >= 1 and exclude it. Update covered for all affected points.

    # To make it more effective, we can use a priority: prefer points whose exclusion
    # doesn't cause many points to become uncovered.

    # Let's try: in each iteration, for each candidate point e with covered[e] >= 1,
    # compute the "cost" = number of points that would drop from covered >= 1 to 0.
    # Pick the one with minimum cost.

    # This is O(2401 * 50 * 48) per iteration in the worst case, which is ~5.7M ops.
    # With 2401 iterations, that's ~13.7B ops. Too slow.

    # Simpler approach: just exclude points in a fixed order, or randomly.
    # Let's try: exclude points in order of their index, skipping those with covered < 1.

    # Even simpler: random greedy with restarts.

    best_N_size = total_points  # We want to minimize |N| = maximize |S|
    best_excluded = np.zeros(total_points, dtype=bool)

    # Strategy: multiple random greedy passes
    np.random.seed(42)

    for trial in range(20):
        if time.time() > deadline - 30:
            break

        excluded = np.zeros(total_points, dtype=bool)
        covered = np.full(total_points, num_dirs, dtype=np.int32)

        # Process points in random order
        order = np.random.permutation(total_points)

        for e in order:
            if time.time() > deadline - 60:
                break
            if covered[e] < 1:
                continue

            # Check if excluding e is valid: after exclusion, no point should have covered < 1
            # We need to simulate: for each (d, pt) affected by excluding e,
            # covered[pt] would decrease by 1. Check if any would go below 1.

            # Affected points: for each direction d, all other points on the line through e in dir d
            # Let's check if any affected point would drop to 0.

            can_exclude = True
            for d in range(num_dirs):
                lid = line_id[d, e]
                members = line_points[d][lid]
                for pt in members:
                    if pt == e:
                        continue
                    if covered[pt] <= 1:
                        # This point would drop to 0
                        can_exclude = False
                        break
                if not can_exclude:
                    break

            if not can_exclude:
                continue

            # Exclude e
            excluded[e] = True
            # Update covered for all affected points
            for d in range(num_dirs):
                lid = line_id[d, e]
                members = line_points[d][lid]
                for pt in members:
                    if pt != e:
                        covered[pt] -= 1

        s_size = excluded.sum()
        if s_size > total_points - best_N_size:
            best_N_size = total_points - s_size
            best_excluded = excluded.copy()
            print(f"Trial {trial}: |S| = {s_size}, |N| = {best_N_size}")

    # Build the point list
    points = []
    for pt in range(total_points):
        if not best_excluded[pt]:
            x_idx = pt_x[pt]
            y_idx = pt_y[pt]
            x_a = x_idx // p
            x_b = x_idx % p
            y_a = y_idx // p
            y_b = y_idx % p
            points.append([[x_a, x_b], [y_a, y_b]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
