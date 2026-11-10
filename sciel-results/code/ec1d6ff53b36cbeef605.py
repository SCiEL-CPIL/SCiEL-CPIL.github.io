# EVOLVE-BLOCK-START
import time
import numpy as np

def nikodym_set() -> np.ndarray:
    """
    Build a Nikodym set in F_49^2 (F_49 = F_7[alpha], alpha^2 = 3).
    Strategy: use the known Hermitian-based construction and then try to
    greedily increase the complement S (points removed from the unital) as long
    as the Nikodym condition remains satisfied.
    """
    start_time = time.time()
    DEADLINE = 600.0
    p = 7
    q = 49
    k_const = 3  # alpha^2 = 3, non-residue mod 7

    # Precompute: all 2401 points as (xa, xb, ya, yb)
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append((xa, xb, ya, yb))

    # Directions: v = (1, m) for m in F_49, plus (0, 1)
    # Represent m as (ma, mb)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, ma, mb))  # (vx, vya, vyb)
    directions.append((0, 0, 1))  # vertical direction

    n_dirs = len(directions)

    # For each point and each direction, precompute the "other" points on the line
    # line_through(point, dir) = {point + t * dir : t in F_49, t != 0}
    # We need to check if all those other points are in N.
    # Equivalent: for each point x and direction d, the set of other points on that line.
    # N is valid iff for each x, there exists a direction d such that no point
    # on the line {x + t*d : t != 0} is in S (complement).
    # i.e., for each x, there exists a direction d such that the line through x in dir d
    # contains NO S-points other than possibly x itself (but x is not in N, so x may be in S or not;
    # actually x must be in N for the condition to apply? No: "every point x of F_49^2 (in N or not)
    # must lie on a line... whose points other than x all belong to N".
    # So even if x is in S, x must have a line through it (in some direction) where all OTHER points are in N.
    # So for each x (whether in S or N), we need at least one direction d such that
    # {x + t*d : t in F_49, t != 0} ⊆ N, i.e., no S-point on that line other than x.
    # But if x is in S, x is not in N, so the condition is about the OTHER points.
    # So the constraint is: for each x, there exists a direction d such that
    # for all t != 0, x + t*d is NOT in S.

    # Let S be a set of indices. For each point x, define bad_dirs[x] = set of directions d
    # such that there exists t != 0 with x + t*d in S.
    # Then x is "covered" if bad_dirs[x] is not all directions.
    # We want to maximize |S| subject to: for all x, bad_dirs[x] != all directions.

    # Precompute for each point x and direction d, the list of other points on the line.
    # line_points[x_idx][d_idx] = list of point indices (excluding x)
    # There are 2401 points and 50 directions. Each line has 49 points, so 48 other points.
    # Total storage: 2401 * 50 * 48 ~ 5.7M entries. Might be heavy but let's try.

    # Instead, let's use a more memory-efficient approach.
    # For each direction d, the lines partition the plane. For each line, we can compute
    # which points are on it. Then for a point x, the directions d for which x's line in d
    # contains an S-point (other than x) are those d where the line L_{x,d} has S-count >= 1
    # (if x not in S) or S-count >= 2 (if x in S, since x itself is one S-point).

    # Let's precompute for each direction d, the line membership:
    # line_id[d][x_idx] = which line of direction d contains x
    # And line_members[d][line_id] = set of point indices on that line.

    # But 2401 * 50 line_ids is fine. Let's do it.

    # Point index: idx = ((xa * p + xb) * p + ya) * p + yb
    def point_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return (xa, xb, ya, yb)

    n_points = q * q

    # For each direction, compute line partition
    # Direction d = (vx, vy_a, vy_b)
    # Two points are on the same line with direction d iff their difference is parallel to d.
    # For vx != 0: x1 and x2 are on same line iff (x1a - x2a, x1b - x2b) = t * (vy_a, vy_b) for some t.
    # This means (x1a - x2a, x1b - x2b) is a scalar multiple of (vy_a, vy_b) in F_49.
    # For vx = 0, vy = (0,1): lines are vertical, so same x-coordinate.

    # Let's precompute line_id for each direction.
    # This is 50 * 2401 integers, which is fine.

    line_id = np.zeros((n_dirs, n_points), dtype=np.int32)
    line_members_list = [None] * n_dirs  # list of lists

    for d_idx, (vx, vya, vyb) in enumerate(directions):
        # Build lines
        if vx == 0:
            # Vertical lines: same x
            # x = (xa, xb)
            # There are 49 x-values, each line has 49 points (all y)
            lid = 0
            members = []
            for xa in range(p):
                for xb in range(p):
                    line_members_list[d_idx] = []
                    line_members_list[d_idx].append([])
                    for ya in range(p):
                        for yb in range(p):
                            idx = point_to_idx(xa, xb, ya, yb)
                            line_id[d_idx, idx] = lid
                            line_members_list[d_idx][-1].append(idx)
                    lid += 1
        else:
            # Non-vertical: vx = 1
            # For each point, its line is determined by its "intercept".
            # A line with direction (1, vya, vyb) through (xa, xb, ya, yb):
            # Points (xa + t, xb + 0*t, ya + t*vya, yb + t*vyb) for t in F_49.
            # Wait, the direction is (vx, vy_a, vy_b) where the x-component is (1, 0) in F_49?
            # Actually, the direction v = (1, m) means the x-coordinate change is 1 (in F_49)
            # and y-coordinate change is m (in F_49).
            # So v = ((1,0), (vya, vyb)).
            # So x changes by (1,0) each step, y changes by (vya, vyb).
            # A line is parameterized by a base point (x0, y0) and all points are
            # (x0 + t*(1,0), y0 + t*(vya, vyb)) for t in F_49.
            # Two points (x1, y1) and (x2, y2) are on the same line iff
            # x2 - x1 = t * (1,0) and y2 - y1 = t * (vya, vyb) for some t.
            # So x2a - x1a = t and x2b - x1b = 0, so x1b = x2b.
            # And y2a - y1a = t * vya, y2b - y1b = t * vyb.
            # If vya, vyb are not both zero, we can solve for t.
            # The line is determined by (xb, ya, yb) adjusted by x.
            # Actually, for a given direction, the lines are indexed by the "offset".
            # Let's just use a dict to group points.

            line_members_list[d_idx] = []
            line_id[d_idx] = -1
            line_map = {}
            for idx in range(n_points):
                xa, xb, ya, yb = idx_to_point(idx)
                # Find the "canonical" representative of the line.
                # We can compute t such that x becomes (0,0): t = -x.
                # t = -(xa, xb) = ((p-xa)%p, (p-xb)%p)
                ta = (p - xa) % p
                tb = (p - xb) % p
                # y at x=0: y0 = y - x * v
                # y0_a = ya - xa * vya (mod p) ... but need to be careful with field arithmetic.
                # x = xa + xb*alpha, v = vya + vyb*alpha
                # x * v = (xa*vya + xb*vyb*3) + (xa*vyb + xb*vya)*alpha
                # So y0 = y - x*v
                # y0_a = ya - (xa*vya + xb*vyb*k_const) mod p
                # y0_b = yb - (xa*vyb + xb*vya) mod p
                xv_a = (xa * vya + xb * vyb * k_const) % p
                xv_b = (xa * vyb + xb * vya) % p
                y0a = (ya - xv_a) % p
                y0b = (yb - xv_b) % p
                key = (xb, y0a, y0b)  # Wait, xb is fixed? No, x changes, so xb changes.
                # Actually, when we go along the line, x changes, so xb changes.
                # The line is determined by the y-intercept y0 when x=0.
                # But x=0 means xa=0 and xb=0.
                # So the key should be (y0a, y0b).
                key = (y0a, y0b)
                if key not in line_map:
                    line_map[key] = len(line_members_list[d_idx])
                    line_members_list[d_idx].append([])
                lid = line_map[key]
                line_id[d_idx, idx] = lid
                line_members_list[d_idx][lid].append(idx)

    # Initialize with the Hermitian-based S
    # S = points in unital U with y NOT in Y_special
    # U: Trace(x) + Norm(y) = 0 => 2*xa + ya^2 - yb^2 * k_const = 0 mod p
    # Y_special: y_a is a square mod p

    squares_mod_p = set((u * u) % p for u in range(p))

    S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k_const) % p == 0
                    in_y_special = ya in squares_mod_p
                    if in_unital and not in_y_special:
                        S.add(point_to_idx(xa, xb, ya, yb))

    # Now try to greedily add points to S
    # For each point x, track how many directions are "blocked" (have an S-point on the line other than x)
    # blocked_count[x] = number of directions d where line_{x,d} contains at least one S-point (other than x)
    # Initially, compute blocked_count for all x.

    blocked_count = np.zeros(n_points, dtype=np.int32)

    # For each point in S, for each direction, mark that all other points on that line have this direction blocked.
    # Actually, if x is in S, then for any other point y on the same line (direction d),
    # the line through y in direction d contains x (which is in S), so direction d is blocked for y.
    # Also, for x itself, the line through x in direction d contains no S-point other than x
    # (since we're checking "other than x"), so direction d is NOT blocked for x by itself.
    # But if there are two S-points on the line, then for each of them, the other is an S-point
    # on the line, so the direction is blocked for both.

    # Let's recompute: for each point x, blocked_count[x] = number of directions d such that
    # there exists y != x with y in S and y on the line through x in direction d.

    # Initialize by iterating over all pairs? That's O(|S|^2 * 50) which is too slow.
    # Instead, for each direction d, for each line L, if |S ∩ L| = c, then for each x in L,
    # the number of S-points on L other than x is c - 1 (if x in S) or c (if x not in S).
    # So direction d is blocked for x iff c - (1 if x in S else 0) > 0.

    # So for each direction d and each line L with c = |S ∩ L|:
    #   If c == 0: no points blocked by this direction on L.
    #   If c == 1: the one S-point is not blocked (by itself), others are blocked.
    #   If c >= 2: all points on L are blocked (since each has at least one other S-point).

    for d_idx in range(n_dirs):
        members_list = line_members_list[d_idx]
        for lid, members in enumerate(members_list):
            c = 0
            for idx in members:
                if idx in S:
                    c += 1
            if c == 0:
                continue
            elif c == 1:
                # Find the one S-point
                s_point = None
                for idx in members:
                    if idx in S:
                        s_point = idx
                        break
                for idx in members:
                    if idx != s_point:
                        blocked_count[idx] += 1
            else:
                for idx in members:
                    blocked_count[idx] += 1

    # Now, we can add a point x to S if after adding it, all points still have blocked_count < n_dirs.
    # When we add x to S, which points' blocked_count increase?
    # For each direction d, the line L = line_{x,d}. Let c = |S ∩ L| before adding x.
    # After adding x, c' = c + 1.
    # For points y in L, y != x:
    #   Before: blocked if c - (1 if y in S else 0) > 0
    #   After: blocked if c' - (1 if y in S else 0) > 0
    # So y's blocked_count increases by 1 iff before: c - (1 if y in S else 0) == 0 and after: c' - (1 if y in S else 0) > 0.
    # Case 1: y not in S. Before: c == 0. After: c+1 - 0 > 0, true. So increases iff c == 0.
    # Case 2: y in S. Before: c - 1 == 0 => c == 1. After: c+1 - 1 = c > 0, true. So increases iff c == 1.
    # Also, for x itself:
    #   Before: blocked if c > 0 (since x not in S, so c - 0 > 0).
    #   After: blocked if c+1 - 1 = c > 0.
    # So x's blocked_count changes from (1 if c>0 else 0) to (1 if c>0 else 0). No change!
    # Wait, that's not right. Let me re-examine.
    # For x, before adding: x not in S. Number of S-points on L other than x = c.
    # So direction d is blocked for x iff c > 0.
    # After adding x to S: number of S-points on L other than x = c (since x is in S now, but we count others).
    # Wait, "other than x" means we exclude x. So if x is in S, the count is still c (the other S-points).
    # So blocked status for x does not change when we add x to S! That's interesting.
    # So adding x only affects other points on the lines through x.

    # So the condition to add x is: for all directions d, let L = line_{x,d}, c = |S ∩ L \ {x}| (before adding x, x not in S, so c = |S ∩ L|).
    # After adding x, for each y in L, y != x:
    #   If y not in S: blocked_count[y] increases by 1 iff c == 0.
    #   If y in S: blocked_count[y] increases by 1 iff c == 1.
    # We need blocked_count[y] + delta < n_dirs for all such y.

    # Let's precompute for each point x and direction d, c = |S ∩ L_{x,d}|.
    # This is 2401 * 50 = 120,050 values. We can store in a 2D array.

    s_line_count = np.zeros((n_points, n_dirs), dtype=np.int32)
    for d_idx in range(n_dirs):
        members_list = line_members_list[d_idx]
        for lid, members in enumerate(members_list):
            c = 0
            for idx in members:
                if idx in S:
                    c += 1
            for idx in members:
                s_line_count[idx, d_idx] = c

    # Now try to add points
    # We'll do random restarts: try to add points in random order, skip if it violates the condition.
    # Keep the best S found.

    best_S = set(S)
    best_size = len(S)

    # Time check
    if time.time() - start_time > DEADLINE - 60:
        # Not enough time, return current best
        pass

    # Try multiple random orderings
    rng = np.random.default_rng(42)
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE - 30:
            break

        S_current = set(S)
        blocked_current = blocked_count.copy()
        s_line_count_current = s_line_count.copy()

        order = rng.permutation(n_points)

        for x in order:
            if time.time() - start_time > DEADLINE - 15:
                break

            if x in S_current:
                continue

            # Check if we can add x
            can_add = True
            for d_idx in range(n_dirs):
                c = s_line_count_current[x, d_idx]  # number of S-points on L_{x,d}
                # For each y in L_{x,d}, y != x:
                #   If y not in S_current: blocked increases by 1 iff c == 0
                #   If y in S_current: blocked increases by 1 iff c == 1
                # We need blocked_current[y] + delta < n_dirs
                # This is expensive to check for all y. Let's precompute the line members.
                # Actually, we can check: for direction d, let L = line_{x,d}.
                # Let n_non_s = number of y in L, y != x, y not in S_current, with c == 0
                # Let n_s = number of y in L, y != x, y in S_current, with c == 1
                # We need for all such y: blocked_current[y] < n_dirs - 1
                # i.e., min blocked_current[y] over such y > n_dirs - 2
                # This is still expensive. Let's just check each y.

                # Get line members for direction d_idx containing x
                lid = line_id[d_idx, x]
                members = line_members_list[d_idx][lid]
                for y in members:
                    if y == x:
                        continue
                    if y in S_current:
                        if c == 1:
                            if blocked_current[y] >= n_dirs - 1:
                                can_add = False
                                break
                    else:
                        if c == 0:
                            if blocked_current[y] >= n_dirs - 1:
                                can_add = False
                                break
                if not can_add:
                    break

            if can_add:
                S_current.add(x)
                # Update blocked_current and s_line_count_current
                for d_idx in range(n_dirs):
                    c = s_line_count_current[x, d_idx]
                    lid = line_id[d_idx, x]
                    members = line_members_list[d_idx][lid]
                    for y in members:
                        if y == x:
                            continue
                        if y in S_current:
                            if c == 1:
                                blocked_current[y] += 1
                        else:
                            if c == 0:
                                blocked_current[y] += 1
                        s_line_count_current[y, d_idx] += 1
                s_line_count_current[x, :] += 1  # x is now in S, so for x, count increases? No, s_line_count[x,d] is count of S-points on L_{x,d} other than x? No, it's total S-points on L_{x,d}.
                # Actually, s_line_count[x,d] should be |S ∩ L_{x,d}|. After adding x, it increases by 1.
                # But we already updated for other y. For x itself, we need to update.
                # Wait, in the loop above, we updated s_line_count_current[y,d] for y in members, y != x.
                # We need to update s_line_count_current[x,d] as well.
                # Let's just do it separately.
                for d_idx in range(n_dirs):
                    s_line_count_current[x, d_idx] += 1

        if len(S_current) > best_size:
            best_size = len(S_current)
            best_S = set(S_current)

    # Build N = all_points \ best_S
    N = []
    for idx in range(n_points):
        if idx not in best_S:
            xa, xb, ya, yb = idx_to_point(idx)
            N.append([[xa, xb], [ya, yb]])

    return np.array(N, dtype=np.int64)

# EVOLVE-BLOCK-END
