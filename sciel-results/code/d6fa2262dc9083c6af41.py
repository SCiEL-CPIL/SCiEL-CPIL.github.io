# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2. The strategy is:
    1. Start with the known valid complement (from the algebraic construction).
    2. Use vectorized validation to greedily expand the complement set S,
       trying to add points one by one while maintaining the Nikodym property.
    3. Return the resulting Nikodym set N = F_49^2 \\ S.
    """
    start_time = time.time()
    deadline = 590.0  # safety margin before 630s kill

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3, non-residue mod 7

    # Precompute field multiplication table: mul[a, b] = a * b in F_49
    # Represent each field element as index 0..48: index = a + b*p
    def idx(a, b):
        return a + b * p

    def a_of(i):
        return i % p

    def b_of(i):
        return i // p

    # Build multiplication table
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = a_of(i), b_of(i)
        for j in range(q):
            aj, bj = a_of(j), b_of(j)
            # (ai + bi*alpha) * (aj + bj*alpha) = (ai*aj + bi*bj*k) + (ai*bj + bi*aj)*alpha
            res_a = (ai * aj + bi * bj * k) % p
            res_b = (ai * bj + bi * aj) % p
            mul_table[i, j] = idx(res_a, res_b)

    # Build all points of F_49^2 as flat indices: point_idx = x * q + y where x, y in 0..48
    total_points = q * q  # 2401

    # Precompute: for each direction type and each point, the set of other points on the line
    # Direction types: (1, m) for m in F_49 (49 directions) and (0, 1) (1 direction) = 50 total
    # For direction (1, m): line through point (x, y) is {(x + t, y + t*m) : t in F_49}
    # For direction (0, 1): line through point (x, y) is {(x, y + t) : t in F_49}

    # Precompute line membership: for each direction d and point p, list of other points on line
    # This is expensive to store fully, so we'll compute on the fly but vectorize

    # Precompute for each direction m (0..48) and each point, the 48 other points
    # Direction (1, m): point (x, y) -> other points are (x + t, y + t*m) for t != 0
    # Direction (0, 1): point (x, y) -> other points are (x, y + t) for t != 0

    # We'll precompute: for direction index d (0..48 is (1,m) with m=d, 49 is (0,1))
    # and point index pt, the 48 other point indices
    # That's 50 * 2401 * 48 = 5,762,400 integers - about 23 MB. Feasible.

    # Actually, let's think about this differently for speed.
    # For validation: a point x is "covered" by direction d if all other 48 points on line_d(x) are in N.
    # Equivalently, none of those 48 points are in S (the complement/excluded set).

    # Let's precompute the line structure more efficiently.
    # For each direction d and point x, we need to check if any of the 48 other points are in S.
    # We can precompute for each direction d, a mapping from point x to the 48 other points.

    # Let's use a different approach: for each direction d, build an array of shape (total_points, 48)
    # where line_pts[d][x] = the 48 other points on the line through x in direction d.

    # Actually, for the greedy algorithm, we need to check: if we add point s to S,
    # does it break the Nikodym property for any point?
    # Point x is valid if there exists a direction d such that all 48 other points on line_d(x) are in N.
    # Adding s to S means s is no longer in N. So point x is broken if for ALL directions d,
    # line_d(x) contains s (and s != x) and all other 47 points were in N.
    # Actually more precisely: x is still valid if there exists d such that s is not on line_d(x) (or s == x),
    # OR s is on line_d(x) but that's okay because... wait no.
    # If s is on line_d(x) and s != x, then adding s to S means line_d(x) now has a point in S,
    # so direction d no longer works for x. But x might still work with another direction.

    # So the check is: after adding s to S, is every point x still covered by at least one direction?
    # This is expensive to check naively. Let's use a counter approach:
    # For each point x, count how many directions d have all 48 other points in N.
    # Initially (with the base S), we compute these counts.
    # When we add s to S, for each point x that has s on its line in direction d (s != x),
    # we decrement the count for (x, d). If the count for x drops to 0, x is no longer covered.

    # Let's precompute: for each point s and each direction d, which points x have s on line_d(x) with s != x.
    # For direction (1, m): s = (sx, sy) is on line through x = (x, y) in direction (1, m) iff
    #   sx = x + t and sy = y + t*m for some t != 0
    #   => t = sx - x, and we need sy = y + (sx-x)*m, i.e., y = sy - (sx-x)*m
    #   So for given s and m, the points x are: x = sx - t, y = sy - t*m for t in F_49 \ {0}
    #   That's 48 points for each (s, m).
    # For direction (0, 1): s = (sx, sy) is on line through x = (x, y) in direction (0,1) iff
    #   sx = x and sy = y + t for some t != 0
    #   => x = sx, y = sy - t for t in F_49 \ {0}
    #   That's 48 points.

    # So for each s (2401 points) and each direction d (50 directions), there are 48 points x affected.
    # Total: 2401 * 50 * 48 = 5,762,400 entries.

    # Let's precompute: affected[s, d] = array of 48 point indices x such that s is on line_d(x), s != x.
    # Actually, we also need to handle the case where s == x: in that case, s is the point itself,
    # and it's not "other than x", so it doesn't affect the count.

    # Let me restructure. Precompute:
    # For each direction d (0..49) and each point x (0..2400), the 48 other points on line_d(x).
    # line_other[d, x, j] for j in 0..47.

    # Direction d=0..48: direction (1, m) where m = d
    # Direction d=49: direction (0, 1)

    # Let's build this.
    line_other = np.zeros((50, total_points, 48), dtype=np.int32)

    for d in range(50):
        for x_idx in range(total_points):
            x = x_idx // q  # x-coordinate in F_49
            y = x_idx % q   # y-coordinate in F_49

            if d < 49:
                m = d
                # Points: (x + t, y + t*m) for t in F_49 \ {0}
                for t_idx in range(q):
                    if t_idx == 0:
                        continue
                    nx = (x + t_idx) % q
                    ny = (y + mul_table[t_idx, m]) % q
                    other_idx = nx * q + ny
                    # Find the slot
                    slot = t_idx - 1 if t_idx > 0 else t_idx  # just need 48 slots
                    # Actually let's just use a counter
                    # This is getting complicated, let me use a list
                    pass

    # This is getting too complex and slow. Let me use a simpler but still vectorized approach.

    # Simpler approach: 
    # 1. Start with the known valid S (complement).
    # 2. Build the set of all points not in S (i.e., N).
    # 3. For each point x, compute the set of valid directions (those where all 48 other points are in N).
    # 4. Greedily try to add points to S (remove from N), checking validity.

    # Even simpler: use the known S as starting point, then try random points to add to S.
    # For each candidate point s to add to S:
    #   - Check if removing s from N (adding to S) keeps the Nikodym property.
    #   - To check: for every point x, is there still at least one direction d where all 48 other points are in N' = N \ {s}?
    #   - A point x is only affected if s is on line_d(x) for some d (s != x).
    #   - For those affected (x, d) pairs, we need to recheck if direction d still works for x.
    #   - But x might have other valid directions, so we need to check if x has ANY valid direction.

    # Let me precompute the valid directions for each point in the initial configuration.

    # Build the initial S (complement) from the algebraic construction.
    S = set()
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add((u * u) % p)

    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = ya in y_special
                    if in_unital and not in_y_special:
                        x_idx = xa + xb * p
                        y_idx = ya + yb * p
                        S.add(x_idx * q + y_idx)

    # N is everything not in S
    # valid_dirs[x] = set of directions d where all 48 other points on line_d(x) are in N
    # For efficiency, let's precompute which points are on each line.

    # Let me precompute: for each direction d and point x, the 48 other points.
    # I'll do this more carefully.

    # Precompute field addition: add[a, b] = a + b in F_49
    # Since F_49 elements are indices 0..48, addition is:
    # (a_a + b_a, a_b + b_b) mod p
    # index = a_a + a_b * p
    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = a_of(i), b_of(i)
        for j in range(q):
            aj, bj = a_of(j), b_of(j)
            add_table[i, j] = ((ai + aj) % p) + ((bi + bj) % p) * p

    # Precompute line_other properly
    line_other = np.full((50, total_points, 48), -1, dtype=np.int32)

    for d in range(50):
        for x_idx in range(total_points):
            x = x_idx // q
            y = x_idx % q
            count = 0
            if d < 49:
                m = d
                for t in range(1, q):
                    # t is a field element index 1..48
                    nx = add_table[x, t]
                    ny = add_table[y, mul_table[t, m]]
                    line_other[d, x_idx, count] = nx * q + ny
                    count += 1
            else:
                # Direction (0, 1)
                for t in range(1, q):
                    nx = x  # x + t*0 = x
                    ny = add_table[y, t]
                    line_other[d, x_idx, count] = nx * q + ny
                    count += 1

    # Now compute valid directions for each point in the initial configuration
    N_mask = np.ones(total_points, dtype=bool)
    for s in S:
        N_mask[s] = False

    # valid_count[x] = number of directions d where all 48 other points are in N
    valid_count = np.zeros(total_points, dtype=np.int32)
    valid_dir_mask = np.zeros((total_points, 50), dtype=bool)

    for x_idx in range(total_points):
        for d in range(50):
            others = line_other[d, x_idx, :]
            if np.all(N_mask[others]):
                valid_count[x_idx] += 1
                valid_dir_mask[x_idx, d] = True

    # Check initial validity
    assert np.all(valid_count > 0), "Initial configuration is not a valid Nikodym set!"

    # Now greedily try to expand S
    # For each candidate point s to add to S:
    #   - It must not already be in S
    #   - After adding s to S (removing from N), every point x must still have valid_count[x] > 0
    #   - Points x that are affected are those where s is on line_d(x) for some d (s != x)
    #   - For each affected (x, d) pair where valid_dir_mask[x, d] was True:
    #       - Check if all other 47 points (excluding s) on line_d(x) are still in N
    #       - If yes, direction d still works; if no, decrement valid_count[x]

    # Precompute: for each point s, which (x, d) pairs are affected
    # affected[s] = list of (x, d) where s is on line_d(x) and s != x
    # For direction d and point s, the affected points x are those where s = x + t*v_d for some t != 0
    # i.e., x = s - t*v_d for t != 0

    # Let's precompute this
    # For each s and d, find all x such that s is in line_other[d, x, :]
    # This is equivalent to: x is on the line through s in direction d, x != s

    # For direction (1, m): x = s - t*(1, m) = (sx - t, sy - t*m) for t != 0
    # For direction (0, 1): x = s - t*(0, 1) = (sx, sy - t) for t != 0

    # affected[s, d] = list of x indices
    # Total size: 2401 * 50 * 48 = 5,762,400
    affected = np.full((total_points, 50, 48), -1, dtype=np.int32)

    for s_idx in range(total_points):
        sx = s_idx // q
        sy = s_idx % q
        for d in range(50):
            count = 0
            if d < 49:
                m = d
                for t in range(1, q):
                    # x = s - t*(1, m) = (sx - t, sy - t*m)
                    neg_t = (q - t) % q  # -t in F_49
                    nx = add_table[sx, neg_t]
                    ny = add_table[sy, mul_table[neg_t, m]]
                    affected[s_idx, d, count] = nx * q + ny
                    count += 1
            else:
                for t in range(1, q):
                    neg_t = (q - t) % q
                    nx = sx
                    ny = add_table[sy, neg_t]
                    affected[s_idx, d, count] = nx * q + ny
                    count += 1

    # Greedy expansion
    # Try all points not in S, in some order
    candidates = [i for i in range(total_points) if i not in S]
    # Shuffle for diversity
    np.random.seed(42)
    np.random.shuffle(candidates)

    improved = True
    while improved and time.time() - start_time < deadline - 60:
        improved = False
        for s in candidates:
            if time.time() - start_time > deadline - 60:
                break
            if s in S:
                continue
            # Try adding s to S
            # Check if this breaks any point
            broken = False
            # For each direction d and each x in affected[s, d, :]:
            #   if valid_dir_mask[x, d] is True, check if direction d still works for x after removing s
            #   i.e., check if all other points on line_d(x) except s are in N
            for d in range(50):
                xs = affected[s, d, :]
                for x in xs:
                    if x == -1:
                        continue
                    if not valid_dir_mask[x, d]:
                        continue
                    # Direction d currently works for x. After removing s from N,
                    # check if it still works: all other 47 points on line_d(x) (excluding s) must be in N
                    others = line_other[d, x, :]
                    # Check all others except s
                    # We can check: all others are in N, except possibly s
                    # Since s is being removed, we need all(others \ {s}) in N
                    # Equivalently: all(others in N) AND (s not in others OR s in N)
                    # But s is not in N (we're adding it to S), so we need s not in others
                    # Wait, s IS in others (that's why x is affected). So we need all other 47 points in N.
                    # Since currently all 48 are in N (that's why valid_dir_mask[x,d] is True),
                    # and we're removing s, we just need to verify that the other 47 are still in N.
                    # They were in N before, and we're only removing s, so they're still in N.
                    # So direction d still works for x!
                    # Wait, that means adding s to S never breaks the Nikodym property?
                    # No, that can't be right. Let me reconsider.
                    
                    # Oh wait, I think I'm confusing myself. Let me re-examine.
                    # valid_dir_mask[x, d] = True means all 48 other points on line_d(x) are in N.
                    # If we add s to S, s is no longer in N.
                    # If s is one of those 48 other points, then after adding s to S,
                    # not all 48 other points are in N, so direction d no longer works for x.
                    # So we DO need to invalidate direction d for x.
                    
                    # But x might have other valid directions, so x is still covered.
                    # We need to check: after invalidating d for x, does x still have at least one valid direction?
                    
                    # So the check is: for each affected (x, d), if valid_dir_mask[x, d] is True,
                    #   we tentatively invalidate it. Then check if valid_count[x] > 0.
                    # If any x has valid_count[x] == 0 after all invalidations, the addition is invalid.
                    
                    # Let me redo this properly.
                    pass
                # I'll handle the full check outside the inner loop
            # Let me restructure: compute all (x, d) that would be invalidated, then check
            # which x's would drop to 0 valid directions
            
            # Collect all (x, d) pairs that would be invalidated
            to_invalidate = set()
            for d in range(50):
                xs = affected[s, d, :]
                for x in xs:
                    if x == -1:
                        continue
                    if valid_dir_mask[x, d]:
                        to_invalidate.add((x, d))
            
            # Check if any x would have 0 valid directions after invalidation
            x_counts = {}
            for (x, d) in to_invalidate:
                x_counts[x] = x_counts.get(x, 0) + 1
            
            valid_addition = True
            for x, cnt in x_counts.items():
                if valid_count[x] - cnt <= 0:
                    valid_addition = False
                    break
            
            if valid_addition:
                # Add s to S
                S.add(s)
                N_mask[s] = False
                for (x, d) in to_invalidate:
                    valid_dir_mask[x, d] = False
                    valid_count[x] -= 1
                improved = True
                # Remove s from candidates
                candidates.remove(s)
                # We can break and restart for a fresh pass
                break

    # Build the result: N = all points not in S
    points = []
    for i in range(total_points):
        if i not in S:
            x = i // q
            y = i % q
            xa, xb = a_of(x), b_of(x)
            ya, yb = a_of(y), b_of(y)
            points.append([[xa, xb], [ya, yb]])

    result = np.array(points, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
