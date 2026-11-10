# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (F_7[alpha], alpha^2 = 3).
    Returns an integer array of shape (k, 2, 2) listing points of N.
    Every point x in F_49^2 must lie on a line {x + t*v : t in F_49}
    with v = (1, m) for some m in F_49 or v = (0, 1), such that all
    other points on that line belong to N.
    We maximize |complement| = 2401 - |N|.
    """
    p = 7
    q = p * p  # 49
    k_const = 3  # alpha^2 = 3

    # Precompute field addition and multiplication tables for F_49
    # Element (a, b) represents a + b*alpha
    # (a1,b1) + (a2,b2) = (a1+a2, b1+b2) mod p
    # (a1,b1) * (a2,b2) = (a1*a2 + b1*b2*k_const, a1*b2 + a2*b1) mod p

    # Index: idx = a + b * p, where a = element_a, b = element_b
    # Point index: pidx = (x_a + x_b * p) * q + (y_a + y_b * p)
    # Total points: q * q = 2401

    # Field multiplication table: mul_table[i][j] = i * j as index
    mul_table = np.zeros((q, q), dtype=np.int32)
    for a1 in range(p):
        for b1 in range(p):
            i = a1 + b1 * p
            for a2 in range(p):
                for b2 in range(p):
                    j = a2 + b2 * p
                    ca = (a1 * a2 + b1 * b2 * k_const) % p
                    cb = (a1 * b2 + a2 * b1) % p
                    mul_table[i, j] = ca + cb * p

    # Field inverse table
    inv_table = np.zeros(q, dtype=np.int32)
    for i in range(1, q):
        for j in range(1, q):
            if mul_table[i, j] == 0:
                continue
            if mul_table[i, j] == 1:
                inv_table[i] = j
                break
    inv_table[0] = 0  # undefined, won't be used

    # Precompute all lines
    # A line is defined by a point and a direction.
    # Directions: v = (1, m) for m in F_49 (49 directions) and v = (0, 1) (1 direction)
    # Total 50 directions.
    # For each direction, we can enumerate all lines.
    # For v = (0, 1): lines are vertical lines x = const. There are q such lines.
    # For v = (1, m): lines are y = x*m + b for b in F_49. There are q*q = 2401 such lines.
    # Total lines: q + q*q = 49 + 2401 = 2450.

    # For each line, store the list of point indices on it.
    # Each line has q = 49 points.

    # Line 0: vertical line x = 0, line 1: x = 1, ..., line 48: x = 48
    # Lines 49..2449: y = x*m + b, indexed by m (0..48) and b (0..48)
    # line_idx = 49 + m * 49 + b

    # Precompute line membership: for each point, which lines does it belong to?
    # point_lines[pidx] = list of line indices
    point_lines = [[] for _ in range(q * q)]

    # Build lines
    # Vertical lines: x = const
    for x_val in range(q):
        line_id = x_val
        for y_val in range(q):
            pidx = x_val * q + y_val
            point_lines[pidx].append(line_id)

    # Slanted lines: y = x * m + b
    for m in range(q):
        for b in range(q):
            line_id = q + m * q + b
            for t in range(q):
                x_val = t
                y_val = (mul_table[m, t] + b) % q
                pidx = x_val * q + y_val
                point_lines[pidx].append(line_id)

    # Now, the Nikodym condition:
    # For every point x, there exists a line L containing x such that
    # all other points on L are in N (i.e., NOT in the excluded set S).
    # Equivalently: for every point x, there exists a line L through x such that
    # L \ {x} is disjoint from S.
    # This means: x is "covered" if there exists a line L through x where
    # S ∩ (L \ {x}) = empty.

    # We want to find a large set S such that every point x has at least one
    # line L through x with S ∩ (L \ {x}) = empty.

    # This is equivalent to: S is an independent set in a certain hypergraph,
    # or more precisely, for each point x, not all lines through x have
    # their other points intersecting S.

    # Greedy approach:
    # Start with S = empty. For each point, check if it can be added to S.
    # A point x can be added to S if, after adding x, every other point y
    # still has at least one "safe" line (a line through y where all other
    # points are not in S).
    #
    # Actually, the condition is on every point x in F_49^2 (whether in S or N).
    # If x is in S, x still needs a safe line. If x is in N, x needs a safe line.
    #
    # Let's re-examine: "every point x of F_49^2 (in N or not) must lie on a line
    # {x + t*v : t in F_49}, with v = (1, m) for some m in F_49 or v = (0, 1),
    # whose points other than x all belong to N."
    #
    # So for EVERY point x (whether x ∈ N or x ∈ S), there must exist a direction v
    # such that all points {x + t*v : t ≠ 0} are in N (i.e., not in S).
    #
    # This means: for every x, there exists a line L through x such that
    # (L \ {x}) ∩ S = ∅.
    #
    # Note: x itself may or may not be in S. The condition is about the OTHER points.
    #
    # So the constraint is: for every x, at least one line L through x has
    # no other points in S.
    #
    # Equivalently: for every x, it is NOT the case that for every line L through x,
    # (L \ {x}) ∩ S ≠ ∅.
    #
    # So x is "violated" if for every line L through x, there exists some y ∈ S
    # with y ≠ x and y on L.
    #
    # We want S as large as possible such that no point is violated.

    # Greedy: try to add points to S one by one. For each candidate point x,
    # check if adding x to S would violate any point.
    #
    # Adding x to S affects: for every point y that shares a line L with x (y ≠ x),
    # the line L now has x in S. If L was the "last safe line" for y (i.e., the only
    # line through y with no other points in S), then y becomes violated.
    #
    # So we need to track, for each point y, how many "safe lines" it has.
    # A line L is "safe" for y if (L \ {y}) ∩ S = ∅.
    #
    # Initially (S = empty), every line is safe for every point on it.
    #
    # When we add x to S:
    # - For every line L containing x: L is no longer safe for any other point y on L
    #   (because x ∈ S and x ≠ y, so (L \ {y}) ∩ S contains x).
    #   Wait, that's not right. L is safe for y if (L \ {y}) ∩ S = ∅.
    #   If we add x to S, and x is on L and x ≠ y, then (L \ {y}) ∩ S now contains x.
    #   So L is no longer safe for y.
    #   But if x = y, then (L \ {y}) doesn't include x, so L could still be safe for y.
    #   But x is being added to S, and x = y means y is being added to S.
    #   For y itself, the lines through y: L is safe for y if (L \ {y}) ∩ S = ∅.
    #   Adding y to S doesn't affect (L \ {y}) ∩ S because y is removed from L in the
    #   punctured line. So y's safe lines are unaffected by adding y to S.
    #
    # So when we add x to S:
    # - For every other point y (y ≠ x) that shares a line L with x:
    #   L is no longer safe for y.
    # - For x itself: x's safe lines are unchanged.
    #
    # We need to ensure that after adding x, no point y has 0 safe lines.
    #
    # Let's track safe_line_count[y] = number of lines L through y with (L \ {y}) ∩ S = ∅.
    #
    # Initially: safe_line_count[y] = number of lines through y = 50 (for each y).
    # Wait, each point is on 1 vertical line and 49 slanted lines = 50 lines total.
    #
    # When we add x to S:
    # - For each line L containing x (50 lines):
    #   - For each other point y on L (48 points):
    #     - safe_line_count[y] decreases by 1 (L is no longer safe for y).
    #
    # We can add x to S iff after this update, all safe_line_count[y] > 0 for all y.
    #
    # This is O(50 * 48) = 2400 per candidate, times 2401 candidates = ~5.7M operations.
    # That's very fast.

    # But we need to be smarter about the order. A greedy that picks the "safest"
    # point first (the one whose addition causes the least damage) might do better.
    #
    # Let's try: at each step, for each point x not yet in S, compute the
    # "damage" = number of (y, L) pairs where y ≠ x, L contains both x and y,
    # and L is currently the ONLY safe line for y.
    # If damage = 0, adding x is safe. Pick the one with minimum damage (0 is best).
    #
    # Actually, simpler: just iterate through points in some order and add if possible.
    # Let's try multiple orders and keep the best.

    start_time = time.time()
    deadline = start_time + 600  # 600 second budget

    # Precompute: for each point x, the list of (y, L) pairs where y ≠ x and
    # L is a line containing both x and y.
    # This is: for each of the 50 lines L through x, for each of the 48 other points y on L.
    # Total: 50 * 48 = 2400 pairs per x.

    # Let's precompute the "neighbors" structure.
    # neighbors[x] = list of (y, line_id) for all y ≠ x sharing a line with x.
    # But we also need to know, for a given line L and point y on L, whether L is
    # currently safe for y.

    # Alternative: maintain a 2D array safe[y][L] = True if L is safe for y.
    # And safe_count[y] = sum of safe[y][L] over all L through y.

    # Lines through each point:
    # point_line_ids[x] = list of 50 line IDs.

    # Let's precompute point_line_ids.
    point_line_ids = [None] * (q * q)
    for x in range(q * q):
        point_line_ids[x] = point_lines[x]

    # For each line, the list of points on it.
    line_points = [[] for _ in range(2450)]
    for x in range(q * q):
        for L in point_lines[x]:
            line_points[L].append(x)

    # Now the greedy algorithm.
    # We'll try different orderings and keep the best result.

    best_S = set()
    best_score = 0

    # Orderings to try:
    # 1. Natural order (0 to 2400)
    # 2. Reverse order
    # 3. Random order (multiple seeds)
    # 4. Order by some heuristic

    orderings = []
    orderings.append(list(range(q * q)))
    orderings.append(list(range(q * q - 1, -1, -1)))

    # Add some random orderings
    rng = np.random.RandomState(42)
    for _ in range(10):
        order = rng.permutation(q * q)
        orderings.append(order.tolist())

    for order in orderings:
        if time.time() > deadline - 30:
            break

        S = set()
        safe_count = [50] * (q * q)  # each point is on 50 lines
        # safe[y][L] is implicitly tracked: we need to know if a specific line L
        # is safe for y. Let's use a 2D boolean array.
        # safe[y_idx, L_idx] but L_idx is not contiguous per y.
        # Better: for each y, a set of safe line IDs.
        safe_lines = [set(range(2450)) for _ in range(q * q)]
        # Actually, only the 50 lines through y matter.
        safe_lines = [set(point_line_ids[y]) for y in range(q * q)]

        for x in order:
            if time.time() > deadline - 30:
                break

            # Check if adding x to S is valid.
            # For each line L through x:
            #   For each other point y on L:
            #     If L is in safe_lines[y], then removing L from safe_lines[y]
            #     would make safe_count[y] decrease by 1.
            #     If safe_count[y] would become 0, it's a violation.
            #
            # We need to check: does there exist a y ≠ x such that
            # safe_count[y] == 1 and L is the only safe line for y, and x is on L?
            #
            # More precisely: for each L through x, for each y on L with y ≠ x:
            #   if L in safe_lines[y] and safe_count[y] == 1: violation.

            valid = True
            # For efficiency, precompute the "dangerous" y's for each line.
            # A y is dangerous for line L if safe_count[y] == 1 and L in safe_lines[y].
            # When we add x to S, for each L through x, all dangerous y's on L
            # (except possibly x itself) would become violated.

            # Let's just check directly.
            for L in point_line_ids[x]:
                for y in line_points[L]:
                    if y == x:
                        continue
                    if L in safe_lines[y] and safe_count[y] == 1:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                # Add x to S
                S.add(x)
                # Update safe_lines and safe_count for all affected y's.
                for L in point_line_ids[x]:
                    for y in line_points[L]:
                        if y == x:
                            continue
                        if L in safe_lines[y]:
                            safe_lines[y].discard(L)
                            safe_count[y] -= 1

        score = q * q - len(S)
        if score > best_score:
            best_score = score
            best_S = S.copy()

    # Build the output: N = all points not in S.
    points = []
    for x in range(q * q):
        if x not in best_S:
            xa = x // q
            ya = x % q
            # Convert field element index to (a, b)
            xa_a = xa % p
            xa_b = xa // p
            ya_a = ya % p
            ya_b = ya // p
            points.append([[xa_a, xa_b], [ya_a, ya_b]])

    result = np.array(points, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
