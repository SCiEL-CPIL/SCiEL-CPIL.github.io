# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 (F_49 = F_7[alpha], alpha^2 = 3).

    Strategy:
      1. Start from a known valid Nikodym set (the unital-based construction).
      2. Greedily add points to the complement S (i.e., remove them from N) while
         preserving the Nikodym property.
      3. Use efficient incremental validity checking.

    A set N is a Nikodym set if for every point x in F_q^2, there exists a
    direction v (either (1,m) for some m in F_q, or (0,1)) such that the
    punctured line {x + t*v : t in F_q, t != 0} is fully contained in N.

    Equivalently: for every point x, for every direction v, if x is NOT in N then
    there must exist some other direction v' such that the punctured line in
    direction v' through x is fully in N.

    Actually, the condition is: for EVERY point x, there EXISTS a direction v such
    that all other points on the line are in N. This must hold regardless of whether
    x itself is in N or not.

    So if we remove a point x from N (add to complement), we need to ensure that
    for x itself, there still exists a direction where all other points on that line
    are in N. AND for all other points y, their existing valid directions must still
    be valid (i.e., if y's only valid direction went through x, and x is now removed,
    y might lose its valid direction).

    This is complex. Let me think more carefully.

    For a point y, it is "satisfied" if there exists a direction v such that all
    points on the line {y + t*v : t != 0} are in N. Note: y itself doesn't need to
    be in N for y to be satisfied – only the OTHER points on the line need to be in N.

    Wait, re-reading: "every point x of F_49^2 (in N or not) must lie on a line
    {x + t*v : t in F_49}, with v = (1, m) for some m in F_49 or v = (0, 1), whose
    points other than x all belong to N."

    So for point x, we need a direction v such that {x + t*v : t in F_49, t != 0}
    subset N. The point x itself can be in or out of N.

    So when we remove a point p from N:
    - For p itself: we need some direction where all other points on the line are in N.
    - For any other point y: if y's valid direction(s) included p as one of the "other
      points", then that direction is no longer valid for y. We need y to have at least
      one remaining valid direction.

    This is a constraint satisfaction problem. Let me implement a greedy approach:
    - Start with the base Nikodym set.
    - For each candidate point to remove (add to complement), check if the Nikodym
      property is still satisfied.
    - If yes, remove it.

    To make this efficient, I'll precompute for each point and direction whether
    the punctured line is fully in N.

    Given the time budget of 630s, and the space being 2401 points with 50 directions
    (49 slopes + vertical = 50 directions), this is 120,050 line-point incidences.
    Checking validity of the whole set is 2401 * 50 = 120,050 checks, each O(49) =
    ~6 million operations. That's fast enough in Python if vectorized.

    Let me use a different approach: represent N as a boolean array of shape (49, 49)
    where index (i, j) corresponds to point with x = i, y = j (flattened field elements).

    Field elements: 0..48, where element = a + b*alpha with a = elem // 7, b = elem % 7.
    alpha^2 = 3.

    Addition: (a1 + b1*alpha) + (a2 + b2*alpha) = (a1+a2) + (b1+b2)*alpha, mod 7.
    So elem1 + elem2 = ((a1+a2)%7)*7 + ((b1+b2)%7).

    Multiplication by t: t * (a + b*alpha) = (ta) + (tb)*alpha, mod 7.
    So t*elem = ((t*a)%7)*7 + ((t*b)%7).

    Direction (1, m): line through x in direction (1, m) is {x + t*(1,m) : t in F_49}.
    x + t*(1,m) = (x + t*1, y + t*m) = (x + t, y + t*m).

    Direction (0, 1): line is {(x, y + t) : t in F_49}.

    Let me flatten: point (x, y) -> index = x * 49 + y.

    For the greedy approach, I'll:
    1. Build the base N.
    2. For each point (in some order), try to remove it from N.
    3. Check if the Nikodym property holds.
    4. If yes, keep it removed; if no, put it back.

    To check the Nikodym property efficiently after removing a point p:
    - The only points whose status might change are:
      a) p itself
      b) Points y for which p lies on the punctured line in some direction, AND p was
         the only point in N on that punctured line that made it valid... no wait.
    
    Actually, removing p from N can only INVALIDATE a direction for a point y if p was
    on the punctured line of y in that direction AND p was in N before (which it was,
    since we're removing it). After removal, p is not in N, so that direction is no
    longer valid for y (if it was relying on p being in N).

    Wait no: a direction v is valid for y if ALL points on the punctured line are in N.
    If p is on the punctured line of y in direction v, and p is removed from N, then
    direction v is no longer valid for y (unless... no, it's definitely not valid anymore
    because p is not in N).

    So removing p from N invalidates, for every other point y and every direction v
    such that p is on the punctured line of y in direction v, that direction v for y.

    For y to still be satisfied, y must have at least one OTHER direction v' that is
    still valid.

    This is getting complex. Let me just do a full validity check each time, but
    optimize it with numpy.

    Actually, let me reconsider. The base set has complement size 148. The greedy
    approach found 17 in the knowledge. Let me just do a simple greedy with full
    validity checks, but be smart about ordering and use numpy for speed.

    Let me precompute:
    - For each point idx and direction d, the list of other points on the line.
    - A validity matrix: valid[idx, d] = True if all other points on line(idx, d) are in N.

    When we remove a point p:
    - For each (idx, d) where p is on the punctured line of idx in direction d,
      valid[idx, d] becomes False (since p is no longer in N).
    - Check if all points still have at least one valid direction.

    This is O(49 * 50) per removal attempt in the worst case, which is manageable.

    Let me implement this.
    """
    p = 7
    q = 49
    alpha_sq = 3  # alpha^2 = 3

    # Precompute field operations
    # Element representation: elem = a * 7 + b, representing a + b*alpha
    def elem_add(e1, e2):
        a1, b1 = divmod(e1, p)
        a2, b2 = divmod(e2, p)
        return ((a1 + a2) % p) * p + ((b1 + b2) % p)

    def elem_mul_scalar(t, e):
        # t is an integer 0..6, e is a field element
        a, b = divmod(e, p)
        return ((t * a) % p) * p + ((t * b) % p)

    def elem_mul(e1, e2):
        a1, b1 = divmod(e1, p)
        a2, b2 = divmod(e2, p)
        # (a1 + b1*alpha)(a2 + b2*alpha) = a1*a2 + (a1*b2 + a2*b1)*alpha + b1*b2*alpha^2
        # = (a1*a2 + b1*b2*3) + (a1*b2 + a2*b1)*alpha
        ra = (a1 * a2 + b1 * b2 * alpha_sq) % p
        rb = (a1 * b2 + a2 * b1) % p
        return ra * p + rb

    # Precompute all field operations as lookup tables
    add_table = np.zeros((q, q), dtype=np.int32)
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        for j in range(q):
            add_table[i, j] = elem_add(i, j)
            mul_table[i, j] = elem_mul(i, j)

    # Precompute scalar multiplication table: scalar_mul[t][e] = t * e
    scalar_mul = np.zeros((p, q), dtype=np.int32)
    for t in range(p):
        for e in range(q):
            scalar_mul[t, e] = elem_mul_scalar(t, e)

    # Directions: 49 slopes m (for direction (1, m)) + 1 vertical (0, 1)
    # Total 50 directions
    # For direction (1, m): line through (x, y) is {(x + t, y + t*m) : t in F_49}
    # For direction (0, 1): line through (x, y) is {(x, y + t) : t in F_49}

    # Precompute for each point and direction, the set of other points on the line
    # line_points[idx, d] = list of the 48 other points (excluding idx itself)
    # Actually, the line has 49 points including x. The punctured line has 48 points.

    # Let me precompute line_members[idx, d] as a set or array of the 48 other points
    # For memory efficiency, I'll use a 3D array: line_other[idx, d, t] = the point at
    # offset t (t = 1..48, skipping t=0 which is x itself)

    # Actually, let me think about this differently.
    # For point idx = x * 49 + y, and direction (1, m):
    #   Point at offset t: (x + t, y + t*m) where t ranges over F_49
    #   The point at t=0 is (x, y) = idx itself.
    #   So the other 48 points are at t = 1, 2, ..., 48 (all nonzero t in F_49)

    # For direction (0, 1):
    #   Point at offset t: (x, y + t)
    #   t=0 gives (x, y) = idx.

    # Let me precompute: for each (x, y) and each direction, the 48 other points.
    # This is 2401 * 50 * 48 = 5,762,400 entries. Manageable.

    # But actually, for the greedy algorithm, I need to know: when I remove point p,
    # which (idx, d) pairs are affected? I.e., for which (idx, d) is p on the punctured
    # line of idx in direction d?

    # Equivalently: for a given point p, which points y have p on their punctured line
    # in direction d? That means p = y + t*d for some t != 0. So y = p - t*d for t != 0.
    # For direction (1, m): y = (p_x - t, p_y - t*m) for t in F_49\{0}.
    # For direction (0, 1): y = (p_x, p_y - t) for t in F_49\{0}.

    # So for each p and direction d, there are 48 points y such that p is on the
    # punctured line of y in direction d.

    # Let me precompute: affected[p, d] = list of 48 (idx) values such that p is on
    # the punctured line of idx in direction d.

    # Actually, let me just precompute the full incidence structure.
    # For each point idx and direction d, store the 48 other points on the line.

    # Let me use a different approach for efficiency. I'll represent N as a boolean
    # array of size 2401. For each point, I'll track how many valid directions it has.

    # valid_count[idx] = number of directions d such that all 48 other points on
    # line(idx, d) are in N.

    # When I remove point p from N:
    #   For each (idx, d) where p is on the punctured line of idx in direction d:
    #     If valid[idx, d] was True, it becomes False (since p is no longer in N).
    #     valid_count[idx] -= 1
    #   Check if any idx has valid_count[idx] == 0. If so, invalid.

    # When I add point p back to N:
    #   For each (idx, d) where p is on the punctured line of idx in direction d:
    #     Check if all other points on line(idx, d) are now in N (including p).
    #     If yes, valid[idx, d] becomes True, valid_count[idx] += 1.

    # Let me precompute the incidence: for each p and direction d, the list of idx
    # such that p is on the punctured line of idx in direction d.

    # There are 2401 * 50 = 120,050 such (p, d) pairs, each with 48 idx values.
    # Total: 5,762,400 entries.

    # Let me build this.
    # directions: 0..48 are slopes m (direction (1, m)), 49 is vertical (0, 1)

    # For direction d < 49 (slope m = d):
    #   p = (px, py). idx = (ix, iy) such that p = ix + t, iy + t*m for some t != 0.
    #   So ix = px - t, iy = py - t*m for t in F_49\{0}.
    #
    # For direction d = 49 (vertical):
    #   p = (px, py). idx = (ix, iy) such that p = (ix, iy + t) for some t != 0.
    #   So ix = px, iy = py - t for t in F_49\{0}.

    # Let me precompute affected[p, d] as an array of 48 indices.
    # Shape: (2401, 50, 48)
    # That's 2401 * 50 * 48 = 5,762,400 int32 values = ~23 MB. OK.

    affected = np.zeros((q * q, 50, 48), dtype=np.int32)
    for px in range(q):
        for py in range(q):
            p_idx = px * q + py
            for d in range(49):  # slope m = d
                for t_idx, t in enumerate(range(1, q)):  # t = 1..48
                    # ix = px - t, iy = py - t*d
                    ix = elem_add(px, q - t)  # px + (-t) = px + (49-t) mod... wait
                    # Actually, -t in F_49. Let me compute properly.
                    # ix = px - t = px + (-t) where -t = (49 - t) % 49? No, field subtraction.
                    # px - t: a1 = px//7, b1 = px%7; t has a_t = t//7, b_t = t%7
                    # ix = (a1 - a_t) % 7 * 7 + (b1 - b_t) % 7
                    a_px, b_px = divmod(px, p)
                    a_t, b_t = divmod(t, p)
                    ix = ((a_px - a_t) % p) * p + ((b_px - b_t) % p)
                    iy = ((a_px - (t * d) % p) % p) * p + ((b_px - ((t // p) * d) % p) % p)
                    # Wait, I need to compute t*d in F_49, then subtract from py.
                    td = mul_table[t, d]
                    a_td, b_td = divmod(td, p)
                    iy = ((a_px - a_td) % p) * p + ((b_px - b_td) % p)
                    # Wait, that's wrong. py is the y-coordinate, not px.
                    # Let me redo.
                    # p = (px, py). For direction (1, d), the line through idx=(ix, iy)
                    # contains p if p = (ix + t, iy + t*d) for some t != 0.
                    # So ix = px - t, iy = py - t*d.
                    # ix = px - t: 
                    #   px = (a_px, b_px), t = (a_t, b_t)
                    #   ix = ((a_px - a_t) % p, (b_px - b_t) % p)
                    # iy = py - t*d:
                    #   py = (a_py, b_py), t*d = (a_td, b_td)
                    #   iy = ((a_py - a_td) % p, (b_py - b_td) % p)
                    a_py, b_py = divmod(py, p)
                    ix = ((a_px - a_t) % p) * p + ((b_px - b_t) % p)
                    iy = ((a_py - a_td) % p) * p + ((b_py - b_td) % p)
                    affected[p_idx, d, t_idx] = ix * q + iy
            # Vertical direction (0, 1): d = 49
            for t_idx, t in enumerate(range(1, q)):
                # ix = px, iy = py - t
                a_t, b_t = divmod(t, p)
                a_py, b_py = divmod(py, p)
                ix = px
                iy = ((a_py - a_t) % p) * p + ((b_py - b_t) % p)
                affected[p_idx, 49, t_idx] = ix * q + iy

    # Now build the base Nikodym set.
    # The base construction: N = F_49^2 \ S, where S = {points in U with y not in Y}
    # U = {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace(x) = 2 * x_a (where x = x_a + x_b * alpha)
    # Norm(y) = y_a^2 - y_b^2 * 3
    # Y = {(u^2 mod 7, v) : u, v in F_7}

    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    N = np.ones(q * q, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            x = xa * p + xb
            for ya in range(p):
                for yb in range(p):
                    y = ya * p + yb
                    in_unital = (2 * xa + ya * ya - yb * yb * alpha_sq) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        N[x * q + y] = False  # excluded from N

    # Initialize valid_count and valid arrays
    valid = np.zeros((q * q, 50), dtype=bool)
    valid_count = np.zeros(q * q, dtype=np.int32)

    # Compute initial valid matrix
    # For each point idx and direction d, check if all 48 other points are in N.
    # I can do this by: for each (idx, d), count how many of the 48 other points are in N.
    # If count == 48, valid.

    # Let me precompute line_other[idx, d, t] = the t-th other point on line(idx, d)
    # Actually, I can derive this from affected. If p is on the punctured line of idx
    # in direction d, then idx is on the punctured line of p in direction d (by symmetry
    # of the line). Wait, no. The line through idx in direction d contains p iff the
    # line through p in direction d contains idx. Yes, that's symmetric.

    # So line_other[idx, d, t] = affected[p, d, t] where p is the point at offset t
    # from idx in direction d.

    # Actually, let me just directly compute for each (idx, d) the 48 other points.
    # For idx = (x, y), direction (1, m): other points are (x + t, y + t*m) for t=1..48.
    # For direction (0, 1): other points are (x, y + t) for t=1..48.

    # Let me build line_other[idx, d, t] for t = 0..47 (representing offsets 1..48).
    line_other = np.zeros((q * q, 50, 48), dtype=np.int32)
    for x in range(q):
        for y in range(q):
            idx = x * q + y
            for d in range(49):  # slope m = d
                for t_idx, t in enumerate(range(1, q)):
                    # Other point: (x + t, y + t*d)
                    a_x, b_x = divmod(x, p)
                    a_t, b_t = divmod(t, p)
                    ox = ((a_x + a_t) % p) * p + ((b_x + b_t) % p)
                    td = mul_table[t, d]
                    a_td, b_td = divmod(td, p)
                    a_y, b_y = divmod(y, p)
                    oy = ((a_y + a_td) % p) * p + ((b_y + b_td) % p)
                    line_other[idx, d, t_idx] = ox * q + oy
            # Vertical
            for t_idx, t in enumerate(range(1, q)):
                a_t, b_t = divmod(t, p)
                a_y, b_y = divmod(y, p)
                oy = ((a_y + a_t) % p) * p + ((b_y + b_t) % p)
                line_other[idx, 49, t_idx] = x * q + oy

    # Compute initial valid matrix
    for idx in range(q * q):
        for d in range(50):
            others = line_other[idx, d, :]
            if np.all(N[others]):
                valid[idx, d] = True
                valid_count[idx] += 1

    # Check initial validity
    assert np.all(valid_count > 0), "Base set is not a valid Nikodym set!"

    # Greedy: try to remove points from N (add to complement)
    # Order: try points that are in N, in some heuristic order.
    # Let me try all points in order, and for each, check if removal is valid.

    # For efficiency, I'll process points in a fixed order and try to remove each one.
    # When I remove point p:
    #   For each d in 0..49:
    #     For each idx in affected[p, d, :]:
    #       If valid[idx, d] is True:
    #         valid[idx, d] = False
    #         valid_count[idx] -= 1
    #   Check if all valid_count > 0.
    #   If yes, commit the removal (N[p] = False).
    #   If no, revert.

    # To make this faster, let me use a different strategy:
    # Process points in random order or by some heuristic.
    # Given time constraints, let me just try all 2401 points in order.

    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline with safety margin

    # Try to remove each point from N
    for p_idx in range(q * q):
        if time.time() > deadline - 30:  # 30 second safety margin
            break
        if not N[p_idx]:
            continue  # already removed

        # Try removing p_idx
        # Track changes to revert if invalid
        changes = []  # list of (idx, d, old_valid)
        for d in range(50):
            for idx in affected[p_idx, d, :]:
                if valid[idx, d]:
                    changes.append((idx, d))
                    valid[idx, d] = False
                    valid_count[idx] -= 1

        # Check validity
        if np.any(valid_count == 0):
            # Revert
            for idx, d in changes:
                valid[idx, d] = True
                valid_count[idx] += 1
        else:
            # Commit removal
            N[p_idx] = False

    # Build the result: points in N
    points = []
    for x in range(q):
        for y in range(q):
            if N[x * q + y]:
                xa, xb = divmod(x, p)
                ya, yb = divmod(y, p)
                points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
