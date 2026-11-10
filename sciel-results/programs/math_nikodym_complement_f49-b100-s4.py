# EVOLVE-BLOCK-START
import numpy as np
import time

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 15.0


def _build_tables():
    """Build field arithmetic, line membership, and the baseline Nikodym set."""
    p = _P
    # alpha^2 = 3 (3 is a quadratic non-residue mod 7)
    k = 3

    # Field addition: (a,b)+(c,d) = ((a+c)%p, (b+d)%p)
    add = np.zeros((p * p, p * p, 2), dtype=np.int64)
    for a in range(p):
        for b in range(p):
            i = a * p + b
            for c in range(p):
                for d in range(p):
                    j = c * p + d
                    add[i, j] = [(a + c) % p, (b + d) % p]

    # Field multiplication: (a+b*alpha)(c+d*alpha) = (ac + bd*k, ad + bc)
    mul = np.zeros((p * p, p * p, 2), dtype=np.int64)
    for a in range(p):
        for b in range(p):
            i = a * p + b
            for c in range(p):
                for d in range(p):
                    j = c * p + d
                    mul[i, j] = [(a * c + b * d * k) % p, (a * d + b * c) % p]

    # Inverse table (0 has no inverse; leave as 0)
    inv = np.zeros(p * p, dtype=np.int64)
    for i in range(1, p * p):
        for j in range(1, p * p):
            if mul[i, j, 0] == 1 and mul[i, j, 1] == 0:
                inv[i] = j
                break

    # Build line membership: for each direction d (0..q, 0=vertical), each offset,
    # the q points on that line.
    # Vertical line through (x, y0): points (x, y0 + t) for t in F_q
    # Slanted line through (x0, y0) with slope m: points (x0 + t, y0 + m*t)
    # Represent each line by (direction_index, offset) where:
    #   direction 0: vertical, offset = x
    #   direction m+1: slanted slope m, offset = y - m*x (the y-intercept at x=0)
    # total lines = 1 + q = 50 directions, each with q lines => 50*49 = 2450 lines
    # Each line has q=49 points.

    # line_points: list of arrays of shape (q,) with point indices in F_q^2
    # point index in F_q^2: ((xa*p+xb)*q + ya*p+yb)
    def pt_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p * p + ya * p + yb)

    line_points = []  # list of np.array of point indices
    for xa in range(p):
        for xb in range(p):
            x = xa * p + xb
            # vertical line
            pts = []
            for t in range(p * p):
                ta, tb = t // p, t % p
                ya, yb = (0 + ta) % p, (0 + tb) % p
                pts.append(pt_idx(xa, xb, ya, yb))
            line_points.append(np.array(pts, dtype=np.int64))
    # slanted lines
    for m in range(p * p):
        ma, mb = m // p, m % p
        for x0a in range(p):
            for x0b in range(p):
                for y0a in range(p):
                    for y0b in range(p):
                        x0 = x0a * p + x0b
                        y0 = y0a * p + y0b
                        # offset = y0 - m*x0 ... but we index by intercept at x=0
                        # Actually a slanted line is determined by slope m and one point.
                        # Two points on same slanted line iff y - m*x is constant (in F_q).
                        # offset = y0 - m*x0
                        off_a = (y0a - (ma * x0a + mb * x0b * k)) % p
                        off_b = (y0b - (ma * x0b + mb * x0a)) % p
                        # We need unique lines; (m, offset) pairs. offset = (off_a, off_b).
                        # But iterating x0,y0 gives duplicates. Instead iterate offset directly.
                        pass
    # Redo slanted lines by (m, offset) directly
    line_points = []
    for xa in range(p):
        for xb in range(p):
            x = xa * p + xb
            pts = []
            for t in range(p * p):
                ta, tb = t // p, t % p
                ya, yb = (ta) % p, (tb) % p
                pts.append(pt_idx(xa, xb, ya, yb))
            line_points.append(np.array(pts, dtype=np.int64))
    for m in range(p * p):
        ma, mb = m // p, m % p
        for oa in range(p):
            for ob in range(p):
                # offset (oa, ob) = y - m*x at x=0 means the line: y = m*x + (oa,ob)
                pts = []
                for t in range(p * p):
                    ta, tb = t // p, t % p
                    # x = t, y = m*t + offset
                    ya = (ma * ta + mb * tb * k + oa) % p
                    yb = (ma * tb + mb * ta + ob) % p
                    xa, xb = ta, tb
                    pts.append(pt_idx(xa, xb, ya, yb))
                line_points.append(np.array(pts, dtype=np.int64))

    n_lines = len(line_points)  # 49 + 49*49 = 2450

    # For each point, list of lines containing it (should be exactly 50)
    point_lines = [[] for _ in range(_Q2)]
    for li, pts in enumerate(line_points):
        for pi in pts:
            point_lines[pi].append(li)

    # Baseline: S = { (x,y) in U : y not in Y }
    # U: 2*xa + ya^2 - yb^2*k == 0 (mod p)
    # Y: ya is a square mod p (i.e., ya in {0,1,2,4})
    squares = set()
    for u in range(p):
        squares.add((u * u) % p)

    S = np.zeros(_Q2, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        if ya not in squares:
                            S[pt_idx(xa, xb, ya, yb)] = True

    return p, k, add, mul, inv, line_points, point_lines, S


def nikodym_set() -> np.ndarray:
    p, k, add, mul, inv, line_points, point_lines, S = _build_tables()
    q = _Q
    q2 = _Q2

    # S is the complement (excluded set). N = all points not in S.
    # For each point x, find a covering line (direction, line) such that all other
    # points on that line are in N (i.e., not in S).
    # A point x is "covered" by line L if x in L and (L \ {x}) subset of N.
    # Equivalently, L contains x and no other point of S.
    # So x is covered iff there exists a line L containing x such that
    #   S_count_on_L == 1 (only x itself) OR x not in S and S_count_on_L == 0.

    # Precompute for each line: which points of S are on it.
    # line_S: list of sets of point indices in S on that line
    line_S = []
    for pts in line_points:
        s_on_line = [pi for pi in pts if S[pi]]
        line_S.append(s_on_line)

    # For each point, the set of lines that cover it (currently)
    # covered[x] = set of line indices that cover x
    # A line L covers x iff x in L and all other points of L are in N.
    # i.e., S points on L are subset of {x} if x in S, or empty if x not in S.

    def covers(line_idx, x):
        s_on = line_S[line_idx]
        if S[x]:
            return len(s_on) == 1 and s_on[0] == x
        else:
            return len(s_on) == 0

    # For each point, which lines cover it
    covered_by = [set() for _ in range(q2)]
    for li in range(len(line_points)):
        pts = line_points[li]
        s_on = line_S[li]
        # Determine which points on this line are covered by it
        if len(s_on) == 0:
            # all points on line are in N, so line covers all its points
            for pi in pts:
                covered_by[pi].add(li)
        elif len(s_on) == 1:
            x = s_on[0]
            # line covers x (the only S point on it)
            covered_by[x].add(li)
        # else: line has >=2 S points, covers no one

    # Check baseline validity: every point must have >=1 covering line
    # (We trust the baseline is valid, but verify quickly.)
    # For efficiency, we'll track for each point the number of covering lines.
    cover_count = np.array([len(c) for c in covered_by], dtype=np.int64)

    # Now greedily add points to S (complement) to maximize |S|.
    # A point z can be added to S iff for every point x, if x currently has
    # cover_count[x] == 1 (only one covering line), then that line must not
    # be "broken" by adding z. Adding z to S breaks a line L iff z in L and
    # L currently has 0 S points (so after adding z, L has 1 S point = z, and
    # L now covers only z, not other points on L).
    # More precisely: adding z to S changes line_S for every line containing z.
    # For a line L containing z: before, S_count(L) = c. After, c+1.
    # Points on L that were covered by L: if c==0, all points on L were covered by L.
    #   After adding z, only z is covered by L. So all other points on L lose L
    #   as a covering line. They remain valid only if they have another covering line.
    #   So adding z is valid iff for every line L containing z with S_count(L)==0,
    #   every point x on L, x != z, has cover_count[x] >= 2.
    #   (If S_count(L)>=1, adding z makes it >=2, so L covered no one before
    #    and covers no one after; no change to coverage of other points, and z
    #    itself: if z was in N, it had some covering line; after adding z to S,
    #    z needs a covering line with exactly z as the only S point. That's a
    #    different condition. Let's handle z's own coverage too.)

    # This is getting complex. Let me use a simpler but correct incremental check.
    # Given the small size (2401 points, 2450 lines), I'll do the following:
    # Maintain S as a boolean array. For each candidate z not in S, check validity
    # of S ∪ {z} by verifying the Nikodym property. To speed up, only re-check
    # points affected: points on lines containing z, plus z itself.

    # Precompute for each point z, the list of lines containing z.
    # (point_lines already has this.)

    # For a fast validity check of S ∪ {z}:
    # A point x is valid iff exists line L containing x with S_count(L ∪ {z}) == 
    #   (1 if x in S∪{z} else 0) and the S-point on L (if any) is x.
    # S_count(L ∪ {z}) = S_count(L) + (1 if z in L else 0).

    # Let me just do a full re-verification but optimized, and use a deadline.

    def is_valid(S_arr):
        """Check Nikodym property. Returns True if valid."""
        # For each line, compute S count
        for li in range(len(line_points)):
            pts = line_points[li]
            s_count = 0
            s_point = -1
            for pi in pts:
                if S_arr[pi]:
                    s_count += 1
                    s_point = pi
                    if s_count > 1:
                        break
            # Determine coverage for points on this line
            if s_count == 0:
                # all points covered by this line
                for pi in pts:
                    if not covered_mark[pi]:
                        covered_mark[pi] = True
            elif s_count == 1:
                if not covered_mark[s_point]:
                    covered_mark[s_point] = True
            # else: no one covered by this line
        # Check all points covered
        for pi in range(q2):
            if not covered_mark[pi]:
                return False
        return True

    covered_mark = np.zeros(q2, dtype=bool)

    # Verify baseline
    covered_mark[:] = False
    assert is_valid(S), "Baseline is not a valid Nikodym set!"

    # Greedy: try to add each point to S, in some order, if valid.
    # Order: try points not currently in S. Use a heuristic: points that are
    # "easy" to add (fewer lines with S_count==0 containing them).
    
    # Precompute for each point z not in S, the number of lines containing z
    # with S_count == 0 (these are the lines that would be "broken").
    # A line L has S_count==0 iff no point of S is on L.
    
    # Compute S_count for each line
    line_S_count = np.zeros(len(line_points), dtype=np.int64)
    for li in range(len(line_points)):
        pts = line_points[li]
        cnt = 0
        for pi in pts:
            if S[pi]:
                cnt += 1
        line_S_count[li] = cnt

    # For each point z, count lines containing z with line_S_count == 0
    # (only consider z not in S)
    candidates = []
    for z in range(q2):
        if S[z]:
            continue
        # count lines with S_count==0 that contain z
        zero_lines = 0
        for li in point_lines[z]:
            if line_S_count[li] == 0:
                zero_lines += 1
        candidates.append((zero_lines, z))
    candidates.sort()

    # Greedy add
    for zero_lines, z in candidates:
        if time.time() > _DEADLINE - _MARGIN:
            break
        # Try adding z to S
        S[z] = True
        # Update line_S_count for lines containing z
        affected_lines = point_lines[z]
        for li in affected_lines:
            line_S_count[li] += 1
        # Check validity
        covered_mark[:] = False
        valid = is_valid(S)
        if valid:
            # Keep z in S
            continue
        else:
            # Revert
            S[z] = False
            for li in affected_lines:
                line_S_count[li] -= 1

    # Build output: N = all points not in S
    pts_list = []
    for z in range(q2):
        if not S[z]:
            xa, xb = z // (p * p), z % (p * p)
            # z = ((xa*p+xb)*p*p + ya*p+yb) ... wait, my pt_idx was ((xa*p+xb)*q + ya*p+yb)
            # Let me recompute. pt_idx(xa,xb,ya,yb) = ((xa*p+xb)*p*p + ya*p+yb)
            # So z = (x * p*p + y) where x = xa*p+xb, y = ya*p+yb
            x = z // (p * p)
            y = z % (p * p)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])

    result = np.array(pts_list, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
