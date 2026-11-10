import numpy as np
import time


def nikodym_set() -> np.ndarray:
    p = 7
    n = p * p  # 49
    n2 = n * n  # 2401

    # Precompute coordinate extraction
    xa_arr = np.arange(n2) // (n * n) % p
    xb_arr = np.arange(n2) // n % p
    ya_arr = np.arange(n2) // p % p
    yb_arr = np.arange(n2) % p

    # Precompute all lines as lists of point indices
    # Lines with direction (1, m) for m in F_49, plus vertical (0, 1)
    # For each point x, the lines through x are:
    #   - vertical: {(x_a, x_b + t) : t in F_49}
    #   - for each m: {(x_a + t, x_b + m*t) : t in F_49}
    # Total lines through each point: 50

    # We'll precompute for each point, the set of lines (as frozensets of indices) passing through it
    # But that's 2401 * 50 = 120050 lines, each with 49 points. Too much memory if stored as sets.
    # Instead, for each line, store the list of points. Then for each point, store which lines it's on.

    # Let's precompute all lines
    # Line identified by (type, param) where type=0 is vertical, type=1..49 is slope m
    # For vertical lines: x_a = c, points are (c, b) for b in F_49
    # For slope m lines: y_a = m * x_a + c, points are (x_a, m*x_a + c) for x_a in F_49

    # Actually, let's just precompute for each point, the list of line IDs that pass through it
    # And for each line ID, the list of points on it.

    # Total lines: 49 vertical + 49*49 slope lines = 49 + 2401 = 2450
    # Wait, for each slope m (49 choices), there are 49 parallel lines. So 49*49 = 2401 slope lines.
    # Plus 49 vertical lines. Total 2450 lines.

    # Let's index lines:
    # Lines 0..48: vertical lines, line c has x_a = c
    # Lines 49..2449: slope lines. Line 49 + m*49 + c has slope m, intercept c
    #   where y_a = m * x_a + c (in F_49)

    # Precompute points_on_line[line_id] = list of point indices
    # And lines_through_point[point_idx] = list of line IDs

    # We need field arithmetic in F_49 = F_7[alpha]/(alpha^2 - 3)
    # Elements are (a, b) representing a + b*alpha
    # Addition: (a1+b1*alpha) + (a2+b2*alpha) = (a1+a2) + (b1+b2)*alpha
    # Multiplication: (a1+b1*alpha)*(a2+b2*alpha) = a1*a2 + b1*b2*alpha^2 + (a1*b2+a2*b1)*alpha
    #   = (a1*a2 + 3*b1*b2) + (a1*b2+a2*b1)*alpha

    def f49_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def f49_mul(a, b):
        return ((a[0]*b[0] + 3*b[1]*b[1]) % p, (a[0]*b[1] + a[1]*b[0]) % p)

    def f49_coord(idx):
        return (idx // p, idx % p)

    def f49_idx(a, b):
        return a * p + b

    # Precompute multiplication table for F_49
    mul_table = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            ci = f49_coord(i)
            cj = f49_coord(j)
            mul_table[i, j] = f49_idx(*f49_mul(ci, cj))

    # Precompute all lines
    # points_on_line: list of arrays, each of length 49
    points_on_line = []
    lines_through = [[] for _ in range(n2)]

    # Vertical lines: x_a = c for c in F_7
    for c in range(p):
        pts = []
        for b_a in range(p):
            for b_b in range(p):
                idx = f49_idx(c, b_a) * n + f49_idx(0, b_b)  # Wait, this is wrong
                # Point is (x_a, x_b, y_a, y_b) = (c, b_x, y_a, y_b)
                # idx = ((x_a * p + x_b) * p + y_a) * p + y_b
                # x_a = c, x_b = b_x, y_a, y_b vary
                # Actually for vertical line x_a = c, x_b varies too? No.
                # Vertical line has direction (0, 1), so x is fixed, y varies.
                # x = (x_a, x_b) fixed, y = (y_a, y_b) varies.
                # So for fixed x = (c_xa, c_xb), points are (c_xa, c_xb, y_a, y_b) for all y.
                pass

    # Let me redo this. A point is (x, y) where x, y in F_49.
    # x = (xa, xb), y = (ya, yb)
    # idx = ((xa * p + xb) * p + ya) * p + yb

    # Vertical line through x = (xa, xb): all points (xa, xb, ya, yb) for ya, yb in F_7
    # That's 49 points.

    # Slope line with slope m through x: y = m * (x - x0) + y0 where (x0, y0) is a reference point.
    # Actually, line through (x0, y0) with direction (1, m): points are (x0 + t, y0 + m*t) for t in F_49.
    # Here x0 + t means component-wise addition in F_49.

    # Let's precompute properly.
    points_on_line = []
    lines_through = [[] for _ in range(n2)]

    # Vertical lines: 49 of them (one for each x in F_49)
    for x_idx in range(n):
        xa = x_idx // p
        xb = x_idx % p
        pts = []
        for y_idx in range(n):
            pt_idx = x_idx * n + y_idx
            pts.append(pt_idx)
        line_id = len(points_on_line)
        points_on_line.append(np.array(pts, dtype=np.int32))
        for pt in pts:
            lines_through[pt].append(line_id)

    # Slope lines: for each slope m (49 choices) and each intercept, 49 lines per slope
    # Line through (x0, y0) with direction (1, m): (x0 + t, y0 + m*t) for t in F_49
    # To enumerate all distinct lines with slope m, we can fix x0 = 0 and vary y0.
    # For x0 = 0, the line is (t, y0 + m*t) for t in F_49.
    # Different y0 give different parallel lines.

    for m_idx in range(n):
        for y0_idx in range(n):
            pts = []
            for t_idx in range(n):
                # x = 0 + t = t
                # y = y0 + m * t
                x_coord = t_idx
                y_coord = f49_idx(*f49_add(f49_coord(y0_idx), f49_coord(mul_table[m_idx, t_idx])))
                pt_idx = x_coord * n + y_coord
                pts.append(pt_idx)
            line_id = len(points_on_line)
            points_on_line.append(np.array(pts, dtype=np.int32))
            for pt in pts:
                lines_through[pt].append(line_id)

    # Now we have all lines. Total lines = 49 + 49*49 = 2450
    # Each point is on 50 lines (1 vertical + 49 slope lines)

    # Greedy algorithm:
    # We want to find a set S (complement of N) that is as large as possible,
    # such that for every point x, there exists a line L through x with L \ {x} subset N = complement of S.
    # Equivalently, for every point x, there exists a line L through x such that L ∩ S ⊆ {x}.
    # i.e., for every x, there is a line L through x where no other point of L is in S.

    # Greedy: start with S = empty. For each point in some order, tentatively add to S.
    # When we add point p to S, for each line L through p, the condition for points on L changes.
    # Specifically, for a point x on L, x can still be "covered" by L only if L ∩ S ⊆ {x}.
    # After adding p, if p != x, then L ∩ S now contains p, so L no longer covers x (unless p = x).
    # So adding p to S removes p as a valid covering line for all other points on lines through p.

    # Let's track for each point x, how many lines through x are still "valid" (i.e., L ∩ S ⊆ {x}).
    # Initially, all 50 lines through x are valid.
    # When we add p to S:
    #   - For p itself: lines through p that don't contain any other S point remain valid. But we just added p,
    #     so for lines L through p, L ∩ S now includes p. For L to still cover p, we need L ∩ S ⊆ {p},
    #     which means no other point of L is in S. This is the same condition as before for other points.
    #   - For other points x on lines through p: the lines L through x that also pass through p are no longer valid
    #     for x (because L ∩ S now contains p ≠ x).

    # So: when adding p to S, for each line L through p, for each x on L with x ≠ p,
    #   the count of valid lines for x decreases by 1.

    # We need to ensure that after adding p, no point has 0 valid lines.

    # Precompute: for each point p, the list of other points on lines through p.
    # Actually, for each p, and for each line L through p, the other points on L.

    # Let's precompute for each p, a list of (x, line_id) pairs where x is on a line through p, x ≠ p.
    # But we need to avoid double-counting if two lines through p share another point? No, two distinct lines
    # through p intersect only at p (in a projective plane sense, but in affine plane, two lines with different
    # slopes intersect at exactly one point, which is p). Vertical and slope lines also intersect at one point.
    # So each other point x on a line through p is on exactly one line through p.

    # So for each p, we can list all other points that are on some line through p.
    # There are 50 lines through p, each with 48 other points, but... wait, do lines through p share other points?
    # No! Two distinct lines through p intersect only at p. So the 50 lines through p partition the other 2351 points.
    # 50 * 48 = 2400, but there are only 2400 other points. Yes, that's correct.

    # So for each p, we can precompute the list of other points on lines through p, grouped by line.

    # For the greedy update: when adding p, we need to decrement the valid count for all 2400 other points.
    # That's expensive if done naively. But we can precompute for each p, the list of all other points on lines through p.

    # Let's precompute: for each p, other_points_on_lines[p] = array of 2400 point indices.
    # And for the reverse: for each point x, which p's have x in their other_points? That's all p ≠ x, so 2400 of them.
    # That's too much to precompute in a useful way.

    # Alternative: just do the decrement. For each p we try to add, we decrement valid_count for 2400 points.
    # With 2401 points to try, that's 2401 * 2400 ≈ 5.76 million operations per greedy pass.
    # With numpy, this should be fast.

    # Precompute for each p, the array of other points on lines through p.
    other_points = np.zeros((n2, 2400), dtype=np.int32)
    for p in range(n2):
        pts_list = []
        for line_id in lines_through[p]:
            for x in points_on_line[line_id]:
                if x != p:
                    pts_list.append(x)
        # Should be exactly 2400
        other_points[p] = np.array(pts_list, dtype=np.int32)

    def greedy_once(seed):
        valid_count = np.full(n2, 50, dtype=np.int32)
        in_S = np.zeros(n2, dtype=bool)
        rng = np.random.RandomState(seed)
        candidates = np.arange(n2)
        rng.shuffle(candidates)

        for p in candidates:
            # Tentatively add p to S
            # Decrement valid_count for all other points on lines through p
            others = other_points[p]
            valid_count[others] -= 1
            # p itself: its valid count doesn't change in a simple way, but we need to check
            # Actually, for p, the lines through p that had no other S points are still valid for p.
            # But we don't track that separately. The valid_count for p represents the number of lines
            # through p that are valid for p. When we add p to S, for a line L through p,
            # L is valid for p iff L ∩ S ⊆ {p}. Before adding p, L was valid for p iff L ∩ S ⊆ {} (empty),
            # i.e., L ∩ S = empty. After adding p, L is valid for p iff L ∩ S ⊆ {p}, i.e., L ∩ (S \ {p}) = empty.
            # So the condition for p is the same as the condition for other points on L.
            # We don't need to update valid_count[p] when adding p, because the lines that were valid
            # for p before (no other S points on L) are still valid for p after (p is the only S point on L).
            # And lines that were not valid for p before (some other S point on L) are still not valid.
            # So valid_count[p] stays the same.

            # Check if any point has valid_count == 0
            if np.min(valid_count) > 0:
                in_S[p] = True
            else:
                # Revert
                valid_count[others] += 1

        return np.sum(in_S), in_S

    start_time = time.time()
    deadline = start_time + 570  # Leave 60s for safety

    best_size = 0
    best_in_S = None
    seed_id = 0

    # Try a few structured orders first
    # Order 1: natural order
    size, in_S = greedy_once(0)
    if size > best_size:
        best_size = size
        best_in_S = in_S.copy()
    seed_id = 1

    # Random restarts
    while time.time() < deadline:
        size, in_S = greedy_once(seed_id)
        if size > best_size:
            best_size = size
            best_in_S = in_S.copy()
        seed_id += 1

    if best_in_S is None:
        best_in_S = np.zeros(n2, dtype=bool)
        best_in_S[0] = True
        best_size = 1

    # Construct N = complement of S
    result_list = []
    for i in range(n2):
        if not best_in_S[i]:
            xa = i // (n * n) % p
            xb = i // n % p
            ya = i // p % p
            yb = i % p
            result_list.append([[xa, xb], [ya, yb]])

    result = np.array(result_list, dtype=np.int64)
    return result
