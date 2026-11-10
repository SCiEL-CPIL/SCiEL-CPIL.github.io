# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using greedy complement growth.
    
    S = complement of N. For validity, every point x must have a direction d
    such that line L(x,d) contains at most 1 point of S (which must be x if x in S).
    We greedily add points to S while maintaining validity.
    """
    p = 7
    q = p * p
    total_points = q * q  # 2401
    num_dirs = q + 1  # 50 directions: 49 slopes + vertical
    
    deadline = time.time() + 600.0
    safety_margin = 30.0  # seconds before hard limit
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Element represented as (a, b) meaning a + b*alpha
    # Multiplication: (a,b)*(c,d) = (ac+bd*3, ad+bc)
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_mul(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*3) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    def field_inv(a):
        # Inverse of a + b*alpha in F_7[alpha]/(alpha^2-3)
        # (a+b*alpha)^-1 = (a - b*alpha) / (a^2 - 3b^2)
        denom = (a[0]*a[0] - 3*b[1]*b[1]) % p
        if denom == 0:
            return (0, 0)  # shouldn't happen for nonzero
        inv_denom = pow(denom, p-2, p)
        return ((a[0]*inv_denom) % p, (-b[1]*inv_denom) % p)
    
    # Directions: (1, m) for m in F_49, and (0, 1)
    # Represent directions as indices 0..49
    # Direction d: if d < 49, slope = d (as field element index), direction vector = (1, m)
    # Direction 49: vertical, direction vector = (0, 1)
    
    # Precompute field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Precompute line membership: for each point and direction, list of points on that line
    # Point index: idx = xa * q + xb * p + ya * p + yb  -> too complex
    # Simpler: point = (x, y) where x, y are field elements
    # Point index: idx = x[0]*q + x[1]*p + y[0]*p + y[1]  ... let me use a flat index
    
    # Let's index points as: idx = xa * p^3 + xb * p^2 + ya * p + yb
    # No wait, simpler: idx = (xa * p + xb) * q + (ya * p + yb) = xa*p^3 + xb*p^2 + ya*p + yb
    # Since p=7, q=49: idx = xa*343 + xb*49 + ya*7 + yb
    
    def point_to_idx(xa, xb, ya, yb):
        return xa * p * p * p + xb * p * p + ya * p + yb
    
    def idx_to_point(idx):
        ya = idx % p
        idx //= p
        xb = idx % p
        idx //= p
        xa = idx % p
        return (xa, xb, ya, yb)
    
    # Precompute lines: for each point idx and direction d, store the list of other 48 points
    # This is 2401 * 50 * 48 integers = ~5.7M integers, manageable
    
    # Actually, for efficiency, let's precompute for each (point, direction) the sorted list of points on that line
    # And for each point, for each direction, the count of S-points on that line (excluding self)
    
    # Line through point P in direction V: {P + t*V : t in F_49}
    # For direction (1, m): line is {(x + t, y + t*m) : t in F_49}
    # For direction (0, 1): line is {(x, y + t) : t in F_49}
    
    # Precompute all line memberships
    # lines[point_idx][dir_idx] = list of 49 point indices (including self)
    
    print(f"Precomputing lines... {time.time() - (deadline - 600):.1f}s elapsed")
    
    # For each direction, precompute the line each point belongs to
    # Actually, let's precompute: for each direction d, a mapping from point to line_id,
    # and for each line_id, the list of points on it.
    
    # Direction d < 49: slope m = field_elements[d], vector = (1, m)
    # Direction 49: vector = (0, 1)
    
    # For direction (1, m): two points (x1,y1) and (x2,y2) are on the same line iff
    # (x2-x1) = t and (y2-y1) = t*m for some t, i.e., y2-y1 = m*(x2-x1)
    # So the line is determined by (x, y - m*x) ... in field arithmetic
    
    # Let me precompute line_id for each (point, direction)
    # line_id = hash of the line, or just use a canonical representative
    
    # For direction (1, m): line through (x, y) has invariant y - m*x (in F_49)
    # For direction (0, 1): line through (x, y) has invariant x
    
    # Precompute: for each direction, for each point, which line it's on
    # And for each line, which points are on it
    
    # This is getting complex. Let me use a simpler approach:
    # For each point, precompute the 48 other points on each of the 50 lines through it.
    
    # lines_through[point_idx][dir_idx] = array of 48 point indices (excluding self)
    # Memory: 2401 * 50 * 48 * 8 bytes = ~46 MB, OK
    
    print(f"Computing line memberships... {time.time() - (deadline - 600):.1f}s")
    
    # Precompute field addition table for speed
    # add_table[i][j] = field_elements[i] + field_elements[j] as index
    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = field_elements[i]
        for j in range(q):
            aj, bj = field_elements[j]
            add_table[i, j] = (ai + aj) % p * p + (bi + bj) % p
    
    # mul_table[i][j] = field_elements[i] * field_elements[j] as index
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = field_elements[i]
        for j in range(q):
            aj, bj = field_elements[j]
            mul_table[i, j] = ((ai*aj + bi*bj*3) % p) * p + ((ai*bj + bi*aj) % p)
    
    # Point index: x_idx = xa*p + xb (0..48), y_idx = ya*p + yb (0..48)
    # Full point index = x_idx * q + y_idx (0..2400)
    
    # For direction d (0..48): slope = field_elements[d], vector = (1, slope)
    # Line through point (x, y) in direction d: {(x + t, y + t*slope) : t in F_49}
    # For direction 49: vector = (0, 1), line: {(x, y + t) : t in F_49}
    
    # Precompute lines_through: shape (2401, 50, 48)
    lines_through = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    
    for x_idx in range(q):
        xa, xb = x_idx // p, x_idx % p
        for y_idx in range(q):
            ya, yb = y_idx // p, y_idx % p
            p_idx = x_idx * q + y_idx
            
            for d in range(num_dirs):
                if d < q:
                    # Direction (1, m) where m = field_elements[d]
                    ma, mb = field_elements[d]
                    for t_idx in range(q):
                        if t_idx == 0:
                            continue  # t=0 gives the point itself
                        ta, tb = field_elements[t_idx]
                        # t * (1, m) = (t, t*m)
                        # t*m in field
                        tm_idx = mul_table[t_idx, d]
                        nx_idx = add_table[x_idx, t_idx]
                        ny_idx = add_table[y_idx, tm_idx]
                        other_idx = nx_idx * q + ny_idx
                        # Find position in the 48-element array
                        pos = t_idx - 1 if t_idx > 0 else q - 1
                        lines_through[p_idx, d, pos] = other_idx
                else:
                    # Direction (0, 1)
                    for t_idx in range(q):
                        if t_idx == 0:
                            continue
                        nx_idx = x_idx
                        ny_idx = add_table[y_idx, t_idx]
                        other_idx = nx_idx * q + ny_idx
                        pos = t_idx - 1 if t_idx > 0 else q - 1
                        lines_through[p_idx, d, pos] = other_idx
    
    print(f"Lines precomputed. {time.time() - (deadline - 600):.1f}s")
    
    # Now greedy: start with S = empty set (N = all points)
    # S is a boolean array of size 2401
    S = np.zeros(total_points, dtype=bool)
    
    # For each point, for each direction, track count of S-points on that line (excluding self)
    # line_count[point_idx][dir_idx] = number of S-points on line through point_idx in direction dir_idx, excluding point_idx itself
    line_count = np.zeros((total_points, num_dirs), dtype=np.int32)
    
    # When we add point p to S:
    # For each direction d, for each other point q on line L(p, d):
    #   line_count[q][d] += 1
    # Also, for p itself: line_count[p][d] = number of other S-points on L(p,d)
    #   But we can compute this from the other points' perspective... 
    #   Actually, line_count[p][d] should count S-points on L(p,d) excluding p.
    #   When p is added to S, for each q on L(p,d) (q != p), line_count[q][d] increases by 1.
    #   And line_count[p][d] = sum over q on L(p,d), q!=p of S[q]
    
    # Validity check for point x:
    # If x in S: need some d with line_count[x][d] == 0
    # If x not in S: need some d with line_count[x][d] == 0 (since x not in S, all S-points on L(x,d) are "other")
    # Wait, for x not in S: line_count[x][d] counts S-points on L(x,d) excluding x. Since x not in S, this is all S-points on L(x,d).
    # We need line_count[x][d] == 0 for some d.
    # For x in S: line_count[x][d] counts S-points on L(x,d) excluding x. We need this to be 0 for some d.
    # So in both cases: valid iff min_d line_count[x][d] == 0.
    
    # A point x is "blocked" if for ALL directions d, line_count[x][d] >= 1.
    # S is valid iff no point is blocked.
    
    # Greedy: try to add each point to S, check if it creates any blocked points.
    # When adding p to S:
    #   For each direction d, for each q on L(p,d), q != p: line_count[q][d] += 1
    #   Also need to set line_count[p][d] properly.
    #   line_count[p][d] = number of S-points on L(p,d) excluding p = sum_{q in L(p,d), q!=p} S[q]
    #   Before adding p, line_count[p][d] was already tracking this (since p was not in S).
    #   After adding p, line_count[p][d] stays the same (it excludes p).
    #   But we need to check if p becomes blocked: min_d line_count[p][d] == 0?
    #   And check if any q becomes blocked: was line_count[q][d] == 0 before, now it's 1, so q might become blocked
    #   if for all other directions d', line_count[q][d'] >= 1.
    
    # To check if q becomes blocked after incrementing line_count[q][d] from 0 to 1:
    #   q is blocked iff for all d', line_count[q][d'] >= 1.
    #   Before the change, line_count[q][d] was 0, so q was NOT blocked.
    #   After, line_count[q][d] is 1. q is blocked iff for all d' != d, line_count[q][d'] >= 1.
    #   i.e., min_{d' != d} line_count[q][d'] >= 1.
    
    # This is expensive to check for all affected q. Let me think of a better way.
    
    # Alternative: maintain for each point the minimum line_count over all directions.
    # min_count[x] = min_d line_count[x][d]
    # x is blocked iff min_count[x] >= 1.
    # When we increment line_count[q][d] from 0 to 1, min_count[q] might increase from 0 to 1
    # (if d was the only direction with count 0).
    
    # Let me track zero_count[x] = number of directions d with line_count[x][d] == 0.
    # x is blocked iff zero_count[x] == 0.
    
    zero_count = np.full(total_points, num_dirs, dtype=np.int32)
    # Initially all line_counts are 0, so zero_count = 50 for all points.
    
    # When adding p to S:
    # For each direction d:
    #   For each q on L(p,d), q != p:
    #     old_lc = line_count[q][d]
    #     line_count[q][d] = old_lc + 1
    #     if old_lc == 0:
    #       zero_count[q] -= 1
    #   # p's line_count for direction d: line_count[p][d] is the count of S-points on L(p,d) excl p
    #   # Before adding p, this was already tracked. After adding p, it's the same.
    #   # But wait: line_count[p][d] should reflect S-points on L(p,d) excluding p.
    #   # Since p is now in S, and we're counting S-points excluding p, it's the same as before.
    #   # However, we also need to check: does p have zero_count[p] > 0?
    #   # line_count[p][d] for each d: this is the number of OTHER S-points on L(p,d).
    #   # Before p was added, line_count[p][d] tracked exactly this.
    #   # So zero_count[p] is unchanged by adding p.
    #   # But we need zero_count[p] > 0 for p to be valid.
    
    # After adding p, check:
    # 1. zero_count[p] > 0 (p is not blocked)
    # 2. For all affected q: zero_count[q] > 0 (q is not blocked)
    
    # If any condition fails, revert.
    
    # Let me also track which points are affected for potential revert.
    
    # Greedy with randomization: shuffle point order, try each.
    # If a point can't be added, skip it.
    
    # To make this more effective, use multiple passes with different orderings.
    
    rng = np.random.default_rng(42)
    point_order = np.arange(total_points)
    
    # Track which points are in S
    # For revert, we need to know which (q, d) pairs were modified.
    
    def try_add(p):
        """Try to add point p to S. Returns True if successful."""
        if S[p]:
            return True
        # Check if p would be blocked: zero_count[p] > 0?
        if zero_count[p] == 0:
            return False
        
        # Collect all modifications
        mods = []  # list of (q, d, old_lc)
        
        for d in range(num_dirs):
            # Get the 48 other points on L(p, d)
            others = lines_through[p, d]
            for q in others:
                q = int(q)
                old_lc = int(line_count[q, d])
                if old_lc == 0:
                    mods.append((q, d, 0, True))  # zero_count will decrease
                else:
                    mods.append((q, d, old_lc, False))
                line_count[q, d] = old_lc + 1
        
        # Check if any q became blocked
        valid = True
        for (q, d, old_lc, was_zero) in mods:
            if was_zero:
                zero_count[q] -= 1
                if zero_count[q] == 0:
                    valid = False
                    break
        
        if not valid:
            # Revert
            for (q, d, old_lc, was_zero) in mods:
                line_count[q, d] = old_lc
                if was_zero:
                    zero_count[q] += 1
            return False
        
        # Also check p itself
        if zero_count[p] == 0:
            # Revert
            for (q, d, old_lc, was_zero) in mods:
                line_count[q, d] = old_lc
                if was_zero:
                    zero_count[q] += 1
            return False
        
        S[p] = True
        return True
    
    # Multiple passes
    max_time = deadline - safety_margin
    num_passes = 0
    improved = True
    
    while time.time() < max_time and improved:
        num_passes += 1
        improved = False
        rng.shuffle(point_order)
        
        for p in point_order:
            if time.time() > max_time:
                break
            if not S[p]:
                if try_add(int(p)):
                    improved = True
    
    print(f"Greedy done in {num_passes} passes. |S| = {S.sum()}, |N| = {total_points - S.sum()}")
    print(f"Time used: {time.time() - (deadline - 600):.1f}s")
    
    # Build the result: N = complement of S
    n_points = total_points - S.sum()
    result = np.zeros((n_points, 2, 2), dtype=np.int64)
    
    idx = 0
    for x_idx in range(q):
        xa, xb = x_idx // p, x_idx % p
        for y_idx in range(q):
            ya, yb = y_idx // p, y_idx % p
            p_idx = x_idx * q + y_idx
            if not S[p_idx]:
                result[idx, 0, 0] = xa
                result[idx, 0, 1] = xb
                result[idx, 1, 0] = ya
                result[idx, 1, 1] = yb
                idx += 1
    
    return result

# EVOLVE-BLOCK-END
