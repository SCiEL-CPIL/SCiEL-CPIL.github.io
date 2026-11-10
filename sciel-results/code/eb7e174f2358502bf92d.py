# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 (q = 49 = 7^2).

    Strategy:
      1. Start from a valid baseline Nikodym set (Hermitian unital construction).
      2. Greedily try to remove points from N that are still "coverable" by some
         line whose other points remain in N.
      3. Use multiple random restarts with shuffled removal orders to find better
         local optima.
      4. Return the smallest valid set found within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0  # seconds

    p = 7
    q = p * p  # 49

    # Find smallest quadratic non-residue mod p
    k_const = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k_const = c
            break

    # Precompute field operations
    # Element is (a, b) representing a + b*alpha, alpha^2 = k_const
    # We'll use integer indices 0..48 for points in F_49^2
    # Point index = (xa * p + xb) * q + (ya * p + yb)
    # Or more naturally: point = (x, y) where x, y in F_49
    # x = xa + xb*alpha, y = ya + yb*alpha
    # Linear index: idx = ((xa * p + xb) * p + (ya * p + yb))

    # Directions: v = (1, m) for m in F_49, and v = (0, 1)
    # Total directions: q + 1 = 50

    # Precompute all lines for each direction
    # For direction (1, m): line through x0 with slope m is {x0 + t*(1, m) : t in F_49}
    # For direction (0, 1): vertical lines {x0 + t*(0, 1) : t in F_49}

    # We need to efficiently check: for a point x, is there a direction v such that
    # all other points on the line through x with direction v are in N?

    # Let's precompute for each direction and each "base" point, the set of points on that line
    # For direction (1, m), the lines are parameterized by their y-intercept (or equivalently,
    # the point where x=0). Line: y = x*m + c for some c.
    # For direction (0, 1), lines are x = constant.

    # Actually, let's think differently. For a given direction v and point x, the line is
    # L(x, v) = {x + t*v : t in F_49}. The punctured line is L(x,v) \ {x}.
    # We need: exists v such that L(x,v) \ {x} subset N.

    # Precompute for each direction:
    #   For each line in that direction, store the set of points on it.
    # Then for point x and direction v, check if all other points on the line are in N.

    # Number of lines per direction: q = 49
    # Total lines: 50 * 49 = 2450
    # Each line has q = 49 points

    # Let's index points by their linear index 0..2400
    # Point (xa, xb, ya, yb) -> idx = xa*p^3 + xb*p^2 + ya*p + yb
    # Or: idx = ((xa*p + xb)*p + ya)*p + yb

    def point_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        return xa, xb, ya, yb

    # Field arithmetic helpers
    def add_f(a, b):
        # a, b are (a_a, a_b), (b_a, b_b)
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul_f(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = (a0*b0 + a1*b1*k) + (a0*b1 + a1*b0)*alpha
        return ((a[0]*b[0] + a[1]*b[1]*k_const) % p,
                (a[0]*b[1] + a[1]*b[0]) % p)

    # Precompute all points as tuples
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append((xa, xb, ya, yb))

    # Precompute lines
    # directions: list of (dx, dy) where each is a field element (a, b)
    # Direction (1, m): dx = (1, 0), dy = m
    # Direction (0, 1): dx = (0, 0), dy = (1, 0)

    directions = []
    # (1, m) for all m in F_49
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    # (0, 1)
    directions.append(((0, 0), (1, 0)))

    # For each direction, compute all lines
    # A line is determined by a "base point" and the direction
    # For direction (1, m): lines are y = x*m + c, parameterized by c (the y-value at x=0)
    # For direction (0, 1): lines are x = constant, parameterized by x

    # Let's precompute: for each direction index d, a list of lines
    # Each line is a list of point indices (all 49 points on that line)

    lines_by_direction = []
    for d, (dx, dy) in enumerate(directions):
        lines = []
        if dx[0] == 0 and dx[1] == 0:
            # Vertical direction (0, 1): lines are x = constant
            # There are q = 49 such lines, one for each x in F_49
            for xa in range(p):
                for xb in range(p):
                    line_pts = []
                    for ya in range(p):
                        for yb in range(p):
                            idx = point_to_idx(xa, xb, ya, yb)
                            line_pts.append(idx)
                    lines.append(line_pts)
        else:
            # Direction (1, m): lines are parameterized by c = y - x*m
            # For each c in F_49, the line is {(x, x*m + c) : x in F_49}
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_pts = []
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            # y = x * m + c
                            xm = mul_f(x, dy)
                            y = add_f(xm, c)
                            idx = point_to_idx(xa, xb, y[0], y[1])
                            line_pts.append(idx)
                    lines.append(line_pts)
        lines_by_direction.append(lines)

    # Now, for efficiency, let's precompute for each point and each direction,
    # the line it belongs to and the set of other points on that line
    # point_line_other[point_idx][dir_idx] = set of point indices on the line through point_idx with direction dir_idx, excluding point_idx

    # This is 2401 * 50 sets of 48 points each. That's a lot of memory but manageable.
    # Actually, let's compute it on the fly or use a different approach.

    # Better approach: for each direction and each line, we know all 49 points.
    # For a point x on that line, the other points are line_pts \ {x}.
    # So we can precompute: for each (dir_idx, line_idx, point_in_line_idx), the set of other points.

    # But that's 50 * 49 * 49 = 120,050 entries, each a set of 48 ints. Should be fine.

    # Actually, let's just precompute a 3D structure:
    # other_points[dir_idx][line_idx][pos] = list of other point indices
    # where pos is the position of the point in the line (0..48)

    # But we need to quickly find which line a point belongs to for a given direction.
    # Let's precompute: point_to_line[dir_idx][point_idx] = (line_idx, pos_in_line)

    point_to_line = []
    for d in range(len(directions)):
        pt2line = [None] * (q * q)
        for line_idx, line_pts in enumerate(lines_by_direction[d]):
            for pos, pt_idx in enumerate(line_pts):
                pt2line[pt_idx] = (line_idx, pos)
        point_to_line.append(pt2line)

    # Precompute other_points for quick access
    # other_pts[d][line_idx][pos] = list of other 48 point indices
    other_pts = []
    for d in range(len(directions)):
        op = []
        for line_idx, line_pts in enumerate(lines_by_direction[d]):
            line_set = set(line_pts)
            for pos in range(q):
                pt = line_pts[pos]
                others = [p2 for p2 in line_pts if p2 != pt]
                op.append(others)
        other_pts.append(op)

    # Baseline Nikodym set construction
    def baseline_nikodym():
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(((u * u) % p, v))

        points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (2 * xa + ya * ya - yb * yb * k_const) % p == 0
                        in_y_special = (ya, yb) in y_special
                        if in_unital and not in_y_special:
                            continue
                        idx = point_to_idx(xa, xb, ya, yb)
                        points.append(idx)
        return set(points)

    def is_valid(nikodym_set_indices):
        """Check if the set is a valid Nikodym set."""
        n_set = set(nikodym_set_indices)
        for x_idx in range(q * q):
            found = False
            for d in range(len(directions)):
                line_idx, pos = point_to_line[d][x_idx]
                others = other_pts[d][line_idx][pos]
                if all(o in n_set for o in others):
                    found = True
                    break
            if not found:
                return False
        return True

    def greedy_remove(nikodym_indices, rng, time_limit):
        """
        Try to remove points from the Nikodym set greedily.
        Returns the resulting set.
        """
        n_set = set(nikodym_indices)
        # Create a list of points to consider for removal, shuffled
        points_list = list(n_set)
        rng.shuffle(points_list)

        # For efficiency, track which points are "coverable"
        # A point is coverable if there exists a direction such that all other points on its line are in N

        # We'll iterate through the shuffled list and try to remove each point
        # After each removal, we need to update the coverability of other points

        # For simplicity, let's just try to remove in order and recheck validity
        # This is O(n^2 * 50 * 48) which might be slow, but let's try

        removed_count = 0
        for pt_idx in points_list:
            if time.time() > time_limit:
                break
            if pt_idx not in n_set:
                continue

            # Check if we can remove pt_idx
            # i.e., is there a direction d such that all other points on the line through pt_idx with direction d are in n_set?
            can_remove = False
            for d in range(len(directions)):
                line_idx, pos = point_to_line[d][pt_idx]
                others = other_pts[d][line_idx][pos]
                if all(o in n_set for o in others):
                    can_remove = True
                    break

            if can_remove:
                n_set.remove(pt_idx)
                removed_count += 1

        return n_set

    # Generate baseline
    baseline = baseline_nikodym()
    best_set = set(baseline)
    best_size = len(baseline)

    # Try multiple restarts with different random orders
    rng = np.random.default_rng(42)
    restart = 0
    while time.time() < deadline - safety_margin:
        # Create a shuffled copy of the baseline
        baseline_list = list(baseline)
        rng.shuffle(baseline_list)

        # Try greedy removal with this order
        result = greedy_remove(baseline_list, rng, deadline - safety_margin)
        if len(result) < best_size:
            best_set = result
            best_size = len(result)
            # Verify validity
            if not is_valid(best_set):
                # Should not happen, but just in case
                best_set = set(baseline)
                best_size = len(baseline)
        restart += 1

        # If we've tried enough restarts or are running out of time, break
        if restart >= 100:
            break

    # Convert best_set to the required format
    points = []
    for idx in best_set:
        xa, xb, ya, yb = idx_to_point(idx)
        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
