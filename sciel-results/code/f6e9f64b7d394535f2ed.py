# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 with q = 49, p = 7.
    Uses a greedy set-cover approach to maximize the complement size.
    """
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill

    p = 7
    k = 3  # alpha^2 = 3, non-residue mod 7

    # Field operations on F_49 = F_7[alpha] where alpha^2 = 3
    def f_add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)

    def f_sub(e1, e2):
        return ((e1[0] - e2[0]) % p, (e1[1] - e2[1]) % p)

    def f_mul(e1, e2):
        a, b = e1
        c, d = e2
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)

    def f_neg(e):
        return ((-e[0]) % p, (-e[1]) % p)

    def f_inv(e):
        a, b = e
        norm = (a * a - b * b * k) % p
        if norm == 0:
            raise ValueError("zero")
        norm_inv = pow(norm, p - 2, p)
        return ((a * norm_inv) % p, (-b * norm_inv) % p)

    def f_div(e1, e2):
        return f_mul(e1, f_inv(e2))

    # All field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]

    # All points in F_49^2
    all_points = []
    for x in field_elements:
        for y in field_elements:
            all_points.append((x[0], x[1], y[0], y[1]))

    # Total number of points
    total_points = len(all_points)

    # Create a set for fast lookup
    all_points_set = set(all_points)

    # All possible directions: v = (1, m) for m in F_49, or v = (0, 1)
    # For each direction, we can compute the line through a point
    # A line is defined by a point x and a direction v
    # The line is {x + t*v : t in F_49}

    # Precompute all lines for each direction
    # For a fixed direction v, the lines partition the space into q lines of q points each
    # We need to find, for each point x, which lines pass through x

    # Strategy:
    # 1. For each direction v, compute all lines (there are q of them for each v)
    # 2. For each point x, find all lines that pass through x
    # 3. Use greedy set cover to select lines that cover all points with minimal total size
    #    but we want to MINIMIZE the Nikodym set, so we want to MAXIMIZE the complement
    #    Actually, we want to find a set N such that for every point x, there exists
    #    a line through x (in some direction) whose other q-1 points are all in N.

    # Let me think about this differently.
    # For each point x, we need to choose a direction v(x) such that
    # all points on the line through x in direction v(x), except x itself, are in N.
    # This means: if we exclude a point y from N, then for every point x on the same line
    # (in the direction v(x)), we cannot use that line to satisfy x's requirement.

    # Equivalently: the complement S = F_49^2 \ N must have the property that
    # for every point x, there exists a direction v such that the line through x in direction v
    # contains no points of S other than possibly x itself.

    # So: for every x, there exists a direction v such that the punctured line
    # {x + t*v : t != 0} is disjoint from S.

    # This is equivalent to: for every x, there exists a direction v such that
    # the line through x in direction v has at most one point in S (namely x itself).

    # So we want to find the largest set S such that for every x, there exists a direction v
    # where the line through x in direction v has |S ∩ line| <= 1 (and if 1, it's x).

    # Actually, more precisely: for every x, there exists a direction v such that
    # S ∩ {x + t*v : t != 0} = empty.
    # This means: for every x, there exists a direction v such that no point y != x
    # on the line through x in direction v is in S.

    # Let's define for each direction v, the lines in that direction.
    # For direction v = (1, m) or v = (0, 1), the lines are parallel.

    # For a fixed direction v, the lines partition F_49^2 into q lines of q points each.
    # Two points are on the same line in direction v iff their difference is a scalar multiple of v.

    # Let me precompute the lines for each direction.

    # Directions:
    # Type 1: v = (0, 1) - vertical lines (constant x)
    # Type 2: v = (1, m) for m in F_49 - lines with slope m

    # For v = (0, 1): lines are x = constant, so for each x0, the line is {(x0, y) : y in F_49}
    # For v = (1, m): lines are y - m*x = constant, so for each c, the line is {(x, m*x + c) : x in F_49}

    # Let's precompute all lines for all directions.
    # There are q + q = 2q = 14 directions? No, q = 49, so 49 + 49 = 98 directions? 
    # Wait, v = (1, m) for m in F_49 gives 49 directions, and v = (0, 1) is 1 more, so 50 directions total.

    # Actually, the problem says: v = (1, m) for some m in F_49 or v = (0, 1).
    # So there are 49 + 1 = 50 directions.

    # For each direction, there are q = 49 lines, each with q = 49 points.
    # Total lines: 50 * 49 = 2450 lines.

    # For each point x, it lies on exactly one line in each direction, so 50 lines total pass through x.

    # We need to select a set S (the complement) such that for every x, there exists a direction v
    # where the line through x in direction v has no other points of S.

    # This is a combinatorial optimization problem. Let's use a greedy approach.

    # Greedy strategy:
    # Start with S = empty.
    # Repeatedly add the point y that can be added to S while maintaining the Nikodym property,
    # prioritizing points that are "least constrained" (i.e., have the most directions where
    # the line through them has few other points in S).

    # But this might be slow. Let's think of a better approach.

    # Alternative: think of it as a constraint satisfaction problem.
    # For each point x, we need at least one direction v where the line through x in direction v
    # has no other points in S.

    # Let's define for each point x and direction v, a "constraint" that is satisfied if
    # the line through x in direction v has no other points in S.

    # We want to maximize |S| subject to: for every x, at least one constraint (x, v) is satisfied.

    # This is a maximum independent set-like problem, which is NP-hard in general.
    # But with q = 49, we might be able to use a greedy or heuristic approach.

    # Let's try a different angle: start with S = all points, and remove points until the property holds.
    # Or start with S = empty and add points.

    # Actually, let's think about the structure more carefully.
    # For a fixed direction v, the lines partition the space. If we put a point y in S,
    # it "blocks" all points x on the same line as y in direction v (except y itself) from using direction v.

    # So if we put y in S, then for every x on the line through y in direction v (x != y),
    # x cannot use direction v to satisfy its requirement.

    # We want to choose S to maximize |S| while ensuring that for every x, at least one direction v
    # is not blocked for x.

    # Let's use a greedy algorithm:
    # 1. Start with S = empty.
    # 2. For each point y (in some order), check if adding y to S maintains the property.
    # 3. If yes, add y to S.
    # 4. Repeat until no more points can be added.

    # The order matters. We should add points that block the fewest other points' options first.

    # But checking if adding y maintains the property is expensive. Let's think of a faster way.

    # Actually, let's just implement a simple greedy and hope it works within the time limit.

    # Precompute for each point x and direction v, the line through x in direction v.
    # And for each line, the set of points on it.

    # Directions:
    # v_0 = (0, 1)
    # v_i = (1, m_i) for i = 0, ..., 48 where m_i is the i-th field element

    # For v = (0, 1): line through (xa, xb, ya, yb) is {(xa, xb, ya', yb') : ya', yb' in F_49}
    #   i.e., all points with the same x-coordinate.

    # For v = (1, m): line through (xa, xb, ya, yb) is {(xa', xb', ya', yb') : 
    #   (xa', xb') = (xa, xb) + t*(1, 0) for some t, and (ya', yb') = (ya, yb) + t*m for the same t}
    #   Wait, v = (1, m) means the direction vector has x-component 1 and y-component m.
    #   So x + t*v = (x_x + t*1, x_y + t*m).
    #   The line is {(x_x + t, x_y + t*m) : t in F_49}.

    # Let's precompute the lines.

    # For each direction, store the lines as lists of points.
    # Also, for each point, store which line it's on for each direction.

    # This is 50 directions * 49 lines * 49 points = 122,500 point-direction-line assignments.
    # That's manageable.

    # Let's represent points as tuples (xa, xb, ya, yb).

    # For direction v = (0, 1):
    #   The line through (xa, xb, ya, yb) is determined by (xa, xb).
    #   Line ID: (xa, xb)
    #   Points on line: all (xa, xb, ya', yb') for ya', yb' in F_49

    # For direction v = (1, m) where m = (ma, mb):
    #   The line through (xa, xb, ya, yb) is determined by the "intercept" c = y - m*x.
    #   c = (ya, yb) - m*(xa, xb) = (ya - ma*xa - mb*xb*k, yb - ma*xb + mb*xa) ... wait, let me compute this properly.
    #   m*x = (ma, mb)*(xa, xb) = (ma*xa + mb*xb*k, ma*xb + mb*xa)
    #   c = (ya, yb) - (ma*xa + mb*xb*k, ma*xb + mb*xa)
    #     = ((ya - ma*xa - mb*xb*k) % p, (yb - ma*xb - mb*xa) % p)
    #   Line ID: c = (ca, cb)
    #   Points on line: {(x, m*x + c) : x in F_49}
    #   For x = (xa', xb'), y = m*x + c = (ma*xa' + mb*xb'*k + ca, ma*xb' + mb*xa' + cb)

    # Let's precompute all this.

    # Field elements indexed
    field_idx = {}
    for i, e in enumerate(field_elements):
        field_idx[e] = i

    # Precompute lines for each direction
    # directions: list of (vx, vy) where vx, vy are field elements
    directions = []
    # v = (0, 1)
    directions.append(((0, 0), (1, 0)))  # (vx, vy) = ((0,0), (1,0))
    # v = (1, m) for m in F_49
    for m in field_elements:
        directions.append(((1, 0), m))

    num_directions = len(directions)

    # For each direction, compute the lines
    # lines[dir_idx] = list of lines, each line is a list of point indices
    # point_line[dir_idx][point_idx] = line_idx

    # Let's use numpy arrays for efficiency
    point_line = np.zeros((num_directions, total_points), dtype=np.int32)
    lines_by_dir = [[] for _ in range(num_directions)]

    for dir_idx, (vx, vy) in enumerate(directions):
        # Compute the line ID for each point
        line_ids = {}
        for pt_idx, (xa, xb, ya, yb) in enumerate(all_points):
            x = (xa, xb)
            y = (ya, yb)
            if vx == (0, 0):
                # v = (0, 1), line ID is x
                lid = x
            else:
                # v = (1, m), line ID is c = y - m*x
                mx = f_mul(vy, x)
                c = f_sub(y, mx)
                lid = c
            if lid not in line_ids:
                line_ids[lid] = []
            line_ids[lid].append(pt_idx)
        
        for lid, pts in line_ids.items():
            lines_by_dir[dir_idx].append(pts)
            for pt_idx in pts:
                point_line[dir_idx, pt_idx] = len(lines_by_dir[dir_idx]) - 1

    # Now, for the greedy algorithm:
    # We want to build S (the complement) such that for every point x, there exists a direction v
    # where the line through x in direction v has no other points in S.

    # Let's define for each point x, the set of directions that are "available" for x,
    # i.e., directions v where the line through x in direction v has no other points in S.

    # Initially, all directions are available for all points.
    # When we add a point y to S, for every direction v, we "block" all other points x on the line
    # through y in direction v from using direction v.

    # We maintain:
    # available[dir_idx][pt_idx] = 1 if direction dir_idx is available for point pt_idx, else 0
    # Initially all 1s.

    # When we add y to S:
    # For each direction v:
    #   Let L be the line through y in direction v.
    #   For each x in L, x != y:
    #     available[v][x] = 0  (block direction v for x)

    # A point x can be in S only if... wait, x itself doesn't need to have an available direction
    # when we're deciding whether to add x to S. The constraint is on the OTHER points.

    # Actually, the constraint is: for every point x (whether in S or not), there must be an available direction.
    # So when we add y to S, we must ensure that all other points still have at least one available direction.

    # Greedy:
    # 1. Start with S = empty, available all 1s.
    # 2. For each point y (in some order), check if adding y to S would cause any point x != y
    #    to have zero available directions.
    # 3. If not, add y to S and update available.
    # 4. If yes, skip y.
    # 5. Repeat.

    # The order of processing y matters. We should process y in an order that minimizes blocking.
    # A good heuristic: process y in order of how many points it blocks (i.e., how many points
    # would lose their last available direction if we add y).

    # But this is complex. Let's just process in a fixed order and see if it works.

    # Actually, let's try a different heuristic: process points in random order, with multiple restarts.

    # But first, let's just try a simple sequential approach.

    available = np.ones((num_directions, total_points), dtype=np.int8)
    S = set()
    in_S = np.zeros(total_points, dtype=np.int8)

    # For each point, count available directions
    avail_count = np.ones(total_points, dtype=np.int32)

    # Process points in order of index
    # To improve, we could sort by some heuristic, but let's start simple

    # For efficiency, when we add y to S, we need to update available for all points on the same lines as y.
    # For each direction v, the line through y has q-1 other points.
    # So we block q-1 points in each of the 50 directions, total 50*48 = 2400 point-direction updates.

    # Let's precompute for each point y and direction v, the list of other points on the same line.
    # other_points[dir_idx][pt_idx] = list of point indices on the same line as pt_idx in direction dir_idx, excluding pt_idx

    other_points = [None] * num_directions
    for dir_idx in range(num_directions):
        other_points[dir_idx] = [None] * total_points
        for line_idx, pts in enumerate(lines_by_dir[dir_idx]):
            for pt_idx in pts:
                other_points[dir_idx][pt_idx] = [q for q in pts if q != pt_idx]

    # Now the greedy:
    # We'll process points in a specific order. Let's try processing in order of "least blocking" first.
    # But computing that is expensive. Let's just process in index order.

    # Actually, let's try a randomized approach with multiple restarts to find a good S.

    best_S_size = 0
    best_S = set()

    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() - start_time > deadline - 60:
            break

        available = np.ones((num_directions, total_points), dtype=np.int8)
        in_S = np.zeros(total_points, dtype=np.int8)
        avail_count = np.ones(total_points, dtype=np.int32)
        S = set()

        # Random order
        order = list(range(total_points))
        np.random.shuffle(order)

        for y in order:
            if time.time() - start_time > deadline - 30:
                break

            # Check if adding y to S would cause any point to have zero available directions
            # For each direction v, the points that would be blocked are other_points[v][y]
            # For each such point x, if avail_count[x] == 1 and the only available direction is v,
            # then x would have zero available directions after blocking.

            # So we need to check: for each direction v, for each x in other_points[v][y],
            # if avail_count[x] == 1 and available[v][x] == 1, then x would be in trouble.

            can_add = True
            for v in range(num_directions):
                for x in other_points[v][y]:
                    if avail_count[x] == 1 and available[v][x] == 1:
                        can_add = False
                        break
                if not can_add:
                    break

            if can_add:
                # Add y to S
                S.add(y)
                in_S[y] = 1
                # Update available and avail_count
                for v in range(num_directions):
                    for x in other_points[v][y]:
                        if available[v][x] == 1:
                            available[v][x] = 0
                            avail_count[x] -= 1

        if len(S) > best_S_size:
            best_S_size = len(S)
            best_S = set(S)

    # Now, N = all_points \ best_S
    # Convert to the required format

    # Build the set of points in N
    S_set = set(best_S)
    N_points = []
    for pt_idx, (xa, xb, ya, yb) in enumerate(all_points):
        if pt_idx not in S_set:
            N_points.append([[xa, xb], [ya, yb]])

    result = np.array(N_points, dtype=np.int64)

    return result

# EVOLVE-BLOCK-END
