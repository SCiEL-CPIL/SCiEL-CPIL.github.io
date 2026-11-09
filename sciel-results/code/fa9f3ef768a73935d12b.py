# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (q = 49, p = 7).
    Uses a greedy priority-queue based removal algorithm to minimize |N|,
    starting from the known valid construction (score ~1.0).
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0  # seconds to reserve for finalization

    p = 7
    q = 49
    total_points = q * q  # 2401

    # Find smallest quadratic non-residue mod p
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    if k is None:
        k = 3  # fallback

    # Build the initial valid Nikodym set (the "skip step 2.5" construction)
    # N = F_q^2 \ S, where S = U \ B
    # U = {(x,y) : Trace(x) + Norm(y) = 0}
    # B = {points in U with y in Y}, Y = {(u^2 mod p, v) : u,v in F_p}

    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    # Represent points as flat indices 0..2400
    # point index = ((xa * p + xb) * p + ya) * p + yb
    # Or more simply: idx = xa * p^3 + xb * p^2 + ya * p + yb

    def point_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def idx_to_point(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        idx //= p
        xa = idx % p
        return xa, xb, ya, yb

    # Initialize: all points are in N
    in_N = np.ones(total_points, dtype=np.int8)

    # Determine which points are in S (excluded from N)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        idx = point_to_idx(xa, xb, ya, yb)
                        in_N[idx] = 0  # excluded

    # Current size of N
    n_size = np.sum(in_N)

    # Build line structure for validation and greedy removal
    # For each point x and each direction v, we need to know the other q-1 points on the line
    # Directions: v = (1, m) for m in F_49, and v = (0, 1)
    # Total directions: q + 1 = 50

    # Precompute line membership for each point-direction pair
    # This is expensive: 2401 * 50 * 49 = ~5.9M entries
    # We'll compute on-the-fly during validation

    # For greedy removal, we need to track for each point:
    # - How many (point, direction) constraints it helps satisfy
    # - When we remove a point, we need to check if any constraint becomes unsatisfied

    # A constraint is: for point x and direction v, all other points on the line must be in N
    # Let's define: constraint (x, v) is satisfied iff all points x + t*v (t != 0) are in N

    # Number of constraints: 2401 * 50 = 120050

    # For each constraint, track which points (other than x) are on its line
    # And track how many of those points are still in N

    # This is a lot of memory. Let's be smarter.

    # Alternative approach: 
    # For each point x, it belongs to exactly q-1 lines in each direction (no, it belongs to exactly 1 line in each direction passing through it)
    # Actually, for a fixed direction v, each point x lies on exactly one line with direction v.
    # The line through x with direction v is {x + t*v : t in F_49}.
    # The constraint for (x, v) is that all other q-1 points on this line are in N.

    # Let's precompute for each point and each direction, the list of other points on the line.
    # This is 2401 * 50 * 48 ≈ 5.8M integers. That's ~46 MB. Manageable.

    # Directions: indexed 0..49
    # Direction 0: v = (0, 1)
    # Direction 1..49: v = (1, m) where m is the (i-1)th element of F_49

    # Let's enumerate F_49 elements as 0..48
    # Element i corresponds to (i // p, i % p)
    # Wait, let's use: element = a + b*alpha, represented as a*p + b? No, let's use a + b*p for simplicity in indexing
    # Actually, let's just use a flat index 0..48 where idx = a * p + b

    def elem_to_idx(a, b):
        return a * p + b

    def idx_to_elem(idx):
        a = idx // p
        b = idx % p
        return a, b

    # Precompute addition and multiplication in F_49
    # (a1 + b1*alpha) + (a2 + b2*alpha) = (a1+a2) + (b1+b2)*alpha
    def elem_add(i1, i2):
        a1, b1 = idx_to_elem(i1)
        a2, b2 = idx_to_elem(i2)
        return elem_to_idx((a1 + a2) % p, (b1 + b2) % p)

    def elem_mul(i1, i2):
        a1, b1 = idx_to_elem(i1)
        a2, b2 = idx_to_elem(i2)
        # (a1 + b1*alpha)(a2 + b2*alpha) = a1*a2 + b1*b2*k + (a1*b2 + a2*b1)*alpha
        a = (a1 * a2 + b1 * b2 * k) % p
        b = (a1 * b2 + a2 * b1) % p
        return elem_to_idx(a, b)

    # Precompute all sums for speed
    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        for j in range(q):
            add_table[i, j] = elem_add(i, j)

    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        for j in range(q):
            mul_table[i, j] = elem_mul(i, j)

    # For each point (x, y) and direction v, compute the line points
    # Point (x, y) is represented as (x_idx, y_idx) where x_idx, y_idx in 0..48
    # Direction v = (vx, vy) where vx, vy in 0..48, not both zero
    # Line: (x + t*vx, y + t*vy) for t in F_49

    # Let's precompute for each of the 2401 points and 50 directions, the 49 line points
    # line_points[point_idx, dir_idx, t] = point_idx of the t-th point on the line
    # This is 2401 * 50 * 49 = 5,882,450 entries. ~23 MB as int32.

    # Directions:
    # dir 0: v = (0, 1) -> vx=0, vy=1
    # dir 1..49: v = (1, m) -> vx=1, vy=m for m in 0..48

    num_dirs = q + 1  # 50

    # We'll build this incrementally to save memory and time
    # For each direction, for each point, compute the line

    # Actually, let's just compute on the fly during the greedy step, but cache the line membership
    # For greedy, we need: for each point x, for each direction v, the set of other points on the line
    # And we need to track how many of those are in N

    # Let's precompute line_members[point_idx, dir_idx] = list of other point indices
    # This is 2401 * 50 lists of 48 elements each

    # Memory: 2401 * 50 * 48 * 4 bytes ≈ 23 MB. OK.

    # But building it takes time. Let's do it efficiently.

    # For a fixed direction v = (vx, vy):
    # For point (x, y), the line is {(x + t*vx, y + t*vy) : t in F_49}
    # The other points are for t = 1, 2, ..., 48 (all t != 0)

    # Let's precompute for each direction and each point, the 48 other points
    # We'll store as a flat array: line_members_flat[point_idx * num_dirs * (q-1) + dir_idx * (q-1) + t]

    # This is 2401 * 50 * 48 = 5,764,800 entries. ~23 MB as int32.

    # Let's build it
    line_members = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)

    # For each direction
    for d in range(num_dirs):
        if d == 0:
            vx, vy = 0, 1
        else:
            vx, vy = 1, d - 1  # vy is the element index

        # For each point
        for x_idx in range(q):
            for y_idx in range(q):
                point_idx = x_idx * q + y_idx
                # Compute line points for t = 1..48
                for t_idx in range(1, q):
                    # t is the t_idx-th nonzero element
                    # We need to iterate t over all nonzero elements
                    # Let's just iterate t over 0..48 and skip t=0
                    pass

    # This is getting complicated. Let me simplify by just iterating t over all elements.

    # For each direction d, vx, vy:
    # For each point (x_idx, y_idx):
    #   For each t in 0..48:
    #     if t == 0: continue
    #     new_x = add_table[x_idx, mul_table[t_idx, vx]]  # wait, t * vx
    #     new_y = add_table[y_idx, mul_table[t_idx, vy]]
    #     line_point_idx = new_x * q + new_y
    #     line_members[point_idx, d, t_idx] = line_point_idx

    # But t_idx is the index of t in 0..48. We want t to range over all elements.
    # So for each t_idx in 0..48:
    #   if t_idx == 0: continue
    #   ...

    # Let's redo this properly

    for d in range(num_dirs):
        if d == 0:
            vx, vy = 0, 1
        else:
            vx, vy = 1, d - 1

        for x_idx in range(q):
            for y_idx in range(q):
                point_idx = x_idx * q + y_idx
                count = 0
                for t_idx in range(q):
                    if t_idx == 0:
                        continue
                    tx = mul_table[t_idx, vx]
                    ty = mul_table[t_idx, vy]
                    new_x = add_table[x_idx, tx]
                    new_y = add_table[y_idx, ty]
                    line_point_idx = new_x * q + new_y
                    line_members[point_idx, d, count] = line_point_idx
                    count += 1

    # Now line_members[point_idx, d, j] for j=0..47 gives the 48 other points on the line
    # through point_idx in direction d.

    # For greedy removal:
    # For each constraint (point_idx, d), we need to know how many of the 48 other points are in N.
    # If this count is 48, the constraint is satisfied.
    # When we remove a point x, for each direction d, the constraint (x, d) is no longer relevant
    # (x is not in N anymore, but the constraint is about points other than x).
    # Wait, the constraint is: for point x (in F_q^2, not necessarily in N), all other points on the line
    # through x in direction d must be in N.
    # So even if x is not in N, the constraint still exists.
    # The constraint (x, d) is satisfied iff all 48 other points are in N.

    # When we remove a point p from N, it affects all constraints (x, d) where p is on the line
    # through x in direction d, i.e., x is on the line through p in direction d.
    # For a fixed p and direction d, the line through p in direction d has 49 points.
    # p is one of them. The other 48 points are x1, ..., x48.
    # For each xi, the constraint (xi, d) has p as one of its "other points".
    # So removing p decreases the "satisfied count" for constraints (xi, d) for i=1..48.

    # Let's maintain:
    # constraint_sat[point_idx, d] = number of other points on the line that are in N
    # Initially, for each (point_idx, d), constraint_sat[point_idx, d] = number of other points in N

    # This is 2401 * 50 = 120050 entries. ~480 KB as int32. Fine.

    constraint_sat = np.zeros((total_points, num_dirs), dtype=np.int32)

    for point_idx in range(total_points):
        for d in range(num_dirs):
            count = 0
            for j in range(q - 1):
                other_idx = line_members[point_idx, d, j]
                if in_N[other_idx] == 1:
                    count += 1
            constraint_sat[point_idx, d] = count

    # A constraint is satisfied iff constraint_sat[point_idx, d] == q - 1 = 48

    # For greedy removal:
    # We want to remove points from N to minimize |N| while keeping all constraints satisfied.
    # A point p can be removed iff for all constraints (x, d) where p is one of the "other points",
    # constraint_sat[x, d] > q - 2 (i.e., after removing p, it will still be >= q - 1... wait)
    # Actually, after removing p, constraint_sat[x, d] decreases by 1 for each (x, d) where p is on the line.
    # The constraint remains satisfied iff the new constraint_sat[x, d] >= q - 1.
    # Since constraint_sat[x, d] is an integer, this means constraint_sat[x, d] >= q - 1 before removal
    # and constraint_sat[x, d] - 1 >= q - 1 after removal, i.e., constraint_sat[x, d] >= q.
    # But constraint_sat[x, d] <= q - 1 always. So constraint_sat[x, d] >= q is impossible.
    # Wait, I think I'm confusing myself.

    # The constraint (x, d) requires all 48 other points to be in N.
    # constraint_sat[x, d] = number of other points currently in N.
    # The constraint is satisfied iff constraint_sat[x, d] == 48.
    # If we remove a point p that is one of the other points for constraint (x, d),
    # then constraint_sat[x, d] decreases by 1.
    # The constraint remains satisfied iff constraint_sat[x, d] was 48 and becomes 47... no, it needs to be 48.
    # So if constraint_sat[x, d] == 48 and we remove one of its other points, it becomes 47 and is no longer satisfied.
    # Therefore, we can only remove p if for all (x, d) where p is an other point, constraint_sat[x, d] > 48.
    # But constraint_sat[x, d] <= 48. So this is never possible.

    # I think I have a fundamental misunderstanding. Let me re-read the task.

    # "every point x of F_49^2 (in N or not) must lie on a line {x + t*v : t in F_49},
    # with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all belong to N."

    # So for each point x, there exists a direction v such that all other points on the line are in N.
    # It's not that ALL directions work for x, just that at least one direction works.

    # This changes everything! The constraint is per-point, not per-point-direction.
    # For each point x, we need at least one direction d such that all 48 other points on the line are in N.

    # So the constraint is: for each x, min over d of (number of other points NOT in N on the line through x in direction d) == 0
    # Or equivalently, for each x, there exists d such that constraint_sat[x, d] == 48.

    # This is a much looser constraint. Let's redo the greedy.

    # For each point x, let best_dir[x] = the direction d that maximizes constraint_sat[x, d].
    # The constraint for x is satisfied iff best_dir[x] has constraint_sat[x, best_dir[x]] == 48.

    # When we remove a point p, it affects constraint_sat[x, d] for all x, d where p is on the line.
    # For each such (x, d), constraint_sat[x, d] decreases by 1.
    # This may change best_dir[x] and may violate the constraint for x.

    # Greedy: remove the point p that, when removed, causes the fewest new violations
    # or the point that is "least critical".

    # A point p is critical for x if best_dir[x] is a direction where p is on the line,
    # and constraint_sat[x, best_dir[x]] == 48 (so removing p would make it 47).

    # Let's track for each point p, how many points x have p as a "critical" point
    # (i.e., x's best direction has p on its line and is currently at 48).

    # This is complex. Let me try a simpler approach: just try to remove points one by one,
    # checking if the Nikodym property still holds.

    # But checking the full property is O(2401 * 50 * 48) = ~5.8M operations per removal.
    # With ~2000 removals, that's ~11.6B operations. Too slow.

    # Let's use the constraint_sat tracking.

    # For each point x, track:
    # sat_count[x] = number of directions d for which constraint_sat[x, d] == 48
    # The constraint for x is satisfied iff sat_count[x] > 0.

    # When we remove p, for each (x, d) where p is on the line through x in direction d:
    #   constraint_sat[x, d] -= 1
    #   if constraint_sat[x, d] was 48 and is now 47, sat_count[x] -= 1
    #   if constraint_sat[x, d] was 47 and is now 46, no change to sat_count
    #   etc.

    # So we need to know, for each point p, the list of (x, d) pairs where p is on the line.
    # This is the same as: for each direction d, for each point x, the 48 other points.
    # We already have line_members. We can invert it.

    # For each point p, for each direction d, the points x such that p is on the line through x in direction d
    # are the points x on the line through p in direction d (including p itself, but we exclude p).
    # Wait, if p is on the line through x in direction d, then x is on the line through p in direction d.
    # So the set of x is {x : x is on the line through p in direction d, x != p}.
    # This is exactly line_members[p, d, :] (the 48 other points).

    # So for each p and d, the affected (x, d) pairs are (x, d) for x in line_members[p, d, :].

    # Let's precompute this. We already have line_members.

    # Now, for greedy:
    # For each point p currently in N, compute its "cost" = number of points x that would become
    # unsatisfied if p is removed.
    # A point x becomes unsatisfied if sat_count[x] == 1 and p is on the line for x's only satisfying direction.
    # Or more generally, if removing p causes sat_count[x] to drop to 0.

    # Let's compute for each p:
    # critical_count[p] = number of x such that sat_count[x] == 1 and p is on the line for x's unique satisfying direction.

    # This is expensive to compute exactly. Let's approximate:
    # For each p, for each d, for each x in line_members[p, d, :]:
    #   if constraint_sat[x, d] == 48 and sat_count[x] == 1:
    #     critical_count[p] += 1

    # This is 2401 * 50 * 48 = 5.8M checks. Doable.

    # Let's implement this.

    # First, compute sat_count
    sat_count = np.zeros(total_points, dtype=np.int32)
    for x in range(total_points):
        for d in range(num_dirs):
            if constraint_sat[x, d] == q - 1:
                sat_count[x] += 1

    # Verify all are satisfied
    assert np.all(sat_count > 0), "Initial set is not a valid Nikodym set"

    # Compute critical_count for each point in N
    critical_count = np.zeros(total_points, dtype=np.int32)

    for p in range(total_points):
        if in_N[p] == 0:
            continue
        for d in range(num_dirs):
            for j in range(q - 1):
                x = line_members[p, d, j]
                if constraint_sat[x, d] == q - 1 and sat_count[x] == 1:
                    critical_count[p] += 1

    # Greedy removal
    # Use a priority queue: (critical_count, point_idx)
    # Remove the point with the smallest critical_count

    # We'll maintain the priority queue and update critical_count as we remove points

    # This is getting very complex. Let me try a simpler iterative approach:
    # Repeatedly find a point with critical_count == 0 and remove it.
    # If no such point exists, try to remove the point with the smallest critical_count.

    # But updating critical_count after each removal is expensive.

    # Given time constraints, let's just do a few rounds of greedy removal with full re-computation.

    # Each round:
    # 1. Compute sat_count and constraint_sat
    # 2. Compute critical_count for all points in N
    # 3. Remove all points with critical_count == 0
    # 4. Repeat until no more points can be removed or time runs out

    # Let's limit to a few rounds to stay within time budget.

    max_rounds = 50
    round_num = 0

    while round_num < max_rounds:
        if time.time() > deadline - safety_margin:
            break

        # Recompute constraint_sat and sat_count
        constraint_sat = np.zeros((total_points, num_dirs), dtype=np.int32)
        for point_idx in range(total_points):
            for d in range(num_dirs):
                count = 0
                for j in range(q - 1):
                    other_idx = line_members[point_idx, d, j]
                    if in_N[other_idx] == 1:
                        count += 1
                constraint_sat[point_idx, d] = count

        sat_count = np.zeros(total_points, dtype=np.int32)
        for x in range(total_points):
            for d in range(num_dirs):
                if constraint_sat[x, d] == q - 1:
                    sat_count[x] += 1

        # Check if all are satisfied
        if not np.all(sat_count > 0):
            # Restore and break
            break

        # Compute critical_count
        critical_count = np.zeros(total_points, dtype=np.int32)
        for p in range(total_points):
            if in_N[p] == 0:
                continue
            for d in range(num_dirs):
                for j in range(q - 1):
                    x = line_members[p, d, j]
                    if constraint_sat[x, d] == q - 1 and sat_count[x] == 1:
                        critical_count[p] += 1

        # Find points with critical_count == 0
        removable = [p for p in range(total_points) if in_N[p] == 1 and critical_count[p] == 0]

        if not removable:
            break

        # Remove all removable points
        for p in removable:
            in_N[p] = 0

        round_num += 1

    # Final validation
    # Check that all points have at least one satisfying direction
    for x in range(total_points):
        satisfied = False
        for d in range(num_dirs):
            all_in_N = True
            for j in range(q - 1):
                other_idx = line_members[x, d, j]
                if in_N[other_idx] == 0:
                    all_in_N = False
                    break
            if all_in_N:
                satisfied = True
                break
        if not satisfied:
            # Invalid, restore to initial set
            # Re-initialize
            in_N = np.ones(total_points, dtype=np.int8)
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                            in_y_special = (ya, yb) in y_special
                            if in_unital and not in_y_special:
                                idx = point_to_idx(xa, xb, ya, yb)
                                in_N[idx] = 0
            break

    # Build output array
    points = []
    for idx in range(total_points):
        if in_N[idx] == 1:
            xa, xb, ya, yb = idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
