# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 (where F_49 = F_7[alpha], alpha^2 = 3).
    
    Strategy: Use a greedy approach to maximize the number of excluded points (complement of N).
    For each point x in F_49^2, we need at least one direction v (either (1,m) for some m in F_49
    or (0,1)) such that all other points on the line x + t*v (t != 0) are in N.
    Equivalently, for each point x, at least one of the 50 lines through x (49 with slope m, 
    1 vertical) must have all its other 48 points in N.
    
    We greedily exclude points: a point x can be excluded if for every direction v through x,
    at least one other point on that line is already excluded. We process points in order of
    "constraint tightness" - points that are on fewer lines with all-other-points-in-N first.
    """
    p = 7
    q = p * p  # 49
    alpha_sq = 3  # alpha^2 = 3 mod 7
    
    # Field arithmetic helpers for F_49 = F_7[alpha] where alpha^2 = 3
    def field_add(a, b):
        # a, b are (a0, a1) pairs
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    def field_mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = (a0*b0 + a1*b1*3) + (a0*b1 + a1*b0)*alpha
        return ((a[0] * b[0] + a[1] * b[1] * alpha_sq) % p,
                (a[0] * b[1] + a[1] * b[0]) % p)
    
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    def field_inv(a):
        # Inverse of a0 + a1*alpha in F_49
        # (a0 + a1*alpha)^(-1) = (a0 - a1*alpha) / (a0^2 - 3*a1^2)
        norm = (a[0] * a[0] - alpha_sq * a[1] * a[1]) % p
        if norm == 0:
            raise ValueError("Zero has no inverse")
        norm_inv = pow(norm, p - 2, p)
        return ((a[0] * norm_inv) % p, (-a[1] * norm_inv) % p)
    
    def field_mul_scalar(a, c):
        # Multiply field element by scalar c in F_7
        return ((a[0] * c) % p, (a[1] * c) % p)
    
    # Enumerate all field elements as indices 0..48
    # Element i corresponds to (i // p, i % p) i.e., (i//7, i%7) = a + b*alpha
    def idx_to_elem(i):
        return (i // p, i % p)
    
    def elem_to_idx(e):
        return e[0] * p + e[1]
    
    # Precompute field multiplication table
    mul_table = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        for j in range(q):
            ei = idx_to_elem(i)
            ej = idx_to_elem(j)
            prod = field_mul(ei, ej)
            mul_table[i, j] = elem_to_idx(prod)
    
    # Precompute field addition table
    add_table = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        for j in range(q):
            ei = idx_to_elem(i)
            ej = idx_to_elem(j)
            prod = field_add(ei, ej)
            add_table[i, j] = elem_to_idx(prod)
    
    # All points in F_49^2: indexed by (x_idx, y_idx) where x_idx, y_idx in 0..48
    # Total: 49 * 49 = 2401 points
    # Point index: pt_idx = x_idx * q + y_idx
    
    def pt_to_idx(x_idx, y_idx):
        return x_idx * q + y_idx
    
    def idx_to_pt(pt_idx):
        x_idx = pt_idx // q
        y_idx = pt_idx % q
        return x_idx, y_idx
    
    # Directions: v = (1, m) for m in F_49 (49 directions) plus v = (0, 1) (1 direction)
    # Total: 50 directions
    # For direction m (slope), line through point (x, y) with direction (1, m):
    #   Points are (x + t, y + t*m) for t in F_49
    # For vertical direction (0, 1):
    #   Points are (x, y + t) for t in F_49
    
    # For each point, compute the 50 lines through it (excluding the point itself, 48 other points per line)
    # line[pt_idx][dir_idx] = set of other 48 points on that line
    
    # Actually, let's think differently. For each point x and direction v, the line is {x + t*v : t in F_49}.
    # The other points are {x + t*v : t in F_49, t != 0}, which is 48 points.
    
    # Precompute for each point and each direction, the list of other points on the line
    # This is a 2401 x 50 x 48 array, which is 2401*50*48 = 5,762,400 entries - manageable
    
    # Directions: m_idx in 0..48 for slope m = field element, and dir 49 for vertical
    # For direction (1, m): line through (x, y) has points (x + t, y + t*m) for t in F_49, t != 0
    # For direction (0, 1): line through (x, y) has points (x, y + t) for t in F_49, t != 0
    
    # Let's precompute the lines more efficiently
    # For a given point (x_idx, y_idx) and slope m_idx:
    #   Other points: for t_idx in 1..48 (t != 0):
    #     x' = x + t, y' = y + t*m
    #     x'_idx = add_table[x_idx, t_idx]
    #     y'_idx = add_table[y_idx, mul_table[t_idx, m_idx]]
    
    # For vertical:
    #   Other points: for t_idx in 1..48:
    #     x' = x, y' = y + t
    #     x'_idx = x_idx
    #     y'_idx = add_table[y_idx, t_idx]
    
    # Build line membership: for each point, for each of 50 directions, list of 48 other points
    # line_pts[pt_idx][dir_idx] = array of 48 point indices
    
    # This will take some memory: 2401 * 50 * 48 * 8 bytes = ~46 MB - OK
    
    # Actually, let's use a different approach for the greedy algorithm.
    # 
    # Greedy: We want to exclude as many points as possible.
    # A point x can be excluded if for EVERY direction v through x, at least one of the other 48 points
    # on that line is already excluded.
    #
    # We process points in some order. When we decide to exclude point x, we check all 50 lines through x.
    # For each line, we need at least one other point on that line to be excluded.
    #
    # Alternative view: Let E be the set of excluded points. For each point x not in E, there must be
    # at least one direction v such that ALL other 48 points on the line x + t*v are in E.
    # Wait, that's the opposite. Let me re-read.
    #
    # "every point x of F_49^2 (in N or not) must lie on a line {x + t*v : t in F_49}, with v = (1, m) 
    # for some m in F_49 or v = (0, 1), whose points other than x all belong to N."
    #
    # So for every point x (whether in N or not), there exists a direction v such that all other points
    # on the line through x in direction v are in N.
    #
    # If x is in N, this means: there's a line through x where all OTHER points are in N.
    # If x is NOT in N (i.e., x is excluded), this means: there's a line through x where all OTHER points
    # are in N (x itself is not in N, but the other 48 points are).
    #
    # So for an excluded point x, we need at least one direction v where the other 48 points on that line
    # are ALL in N (i.e., NOT excluded).
    #
    # This is a constraint satisfaction problem. Let's use a greedy approach:
    # 
    # Start with all points in N (no exclusions). Then try to exclude points one by one.
    # When we exclude point x, we must ensure that for x itself, there's still a valid direction.
    # But excluding x might break the validity of other points that relied on x being in N.
    #
    # Better approach: Use a greedy algorithm where we try to exclude points in an order that minimizes
    # disruption. 
    #
    # Let's think about it differently. For each point x, define the set of "valid directions" for x as
    # those directions v where all other 48 points on the line are in N. Initially, all 50 directions are valid
    # for every point. When we exclude a point y, for every point x on any line with y, we need to check if
    # that line was a valid direction for x - if it was, and y was the only "in N" point that mattered... 
    # wait, no. Excluding y means y is NOT in N. So for any line that contains y, if that line was previously
    # valid for some point x (meaning all other points were in N), now it might not be valid if y was one of
    # the "other points" for x.
    #
    # Let me reconsider. For point x and direction v, the line is L(x, v) = {x + t*v : t in F_49}.
    # The "other points" are L(x, v) \ {x}. This line is valid for x if all points in L(x, v) \ {x} are in N.
    #
    # When we exclude point y (remove y from N), for any point x and direction v such that y is in L(x, v) \ {x},
    # the line L(x, v) is no longer valid for x (because y is no longer in N).
    #
    # So the algorithm:
    # 1. Start with all points in N.
    # 2. For each point x, maintain the set of valid directions.
    # 3. Try to exclude points one by one. A point x can be excluded if after exclusion, x still has at least
    #    one valid direction, AND no other point loses its last valid direction.
    # 4. To make this efficient, process points in an order that's likely to work.
    
    # Given the complexity, let's use a simpler greedy: 
    # Try to exclude points in a fixed order. For each point, check if it can be excluded (i.e., after exclusion,
    # it still has a valid direction, and we don't break other points). If yes, exclude it.
    
    # To make this fast, we need efficient data structures.
    
    # Let's precompute for each point x and direction v, the set of other points on that line.
    # other_pts[x_idx, v_idx] = set of point indices on line through x in direction v, excluding x
    
    # x_idx in 0..2400, v_idx in 0..49
    # Each set has 48 elements
    
    # This is 2401 * 50 sets of 48 elements. Let's store as a 3D array: other_pts[x_idx, v_idx, 0:48]
    
    other_pts = np.zeros((q*q, 50, q-1), dtype=np.int64)  # 2401 * 50 * 48
    
    for x_idx in range(q):
        for y_idx in range(q):
            pt = pt_to_idx(x_idx, y_idx)
            # Directions 0..48: slope m = field element with index d
            for d in range(q):
                m_idx = d
                for t_idx in range(1, q):  # t != 0
                    # x' = x + t, y' = y + t*m
                    x_new = add_table[x_idx, t_idx]
                    y_new = add_table[y_idx, mul_table[t_idx, m_idx]]
                    pt_new = pt_to_idx(x_new, y_new)
                    other_pts[pt, d, t_idx - 1] = pt_new
            # Direction 49: vertical
            for t_idx in range(1, q):
                x_new = x_idx
                y_new = add_table[y_idx, t_idx]
                pt_new = pt_to_idx(x_new, y_new)
                other_pts[pt, 49, t_idx - 1] = pt_new
    
    # Now, greedy exclusion
    # in_N[pt] = True if point is in N
    in_N = np.ones(q*q, dtype=bool)
    
    # valid_dir_count[pt] = number of valid directions for point pt
    # A direction v is valid for pt if all other_pts[pt, v, :] are in N
    valid_dir_count = np.zeros(q*q, dtype=np.int64)
    
    # Initialize: all directions are valid for all points
    valid_dir_count[:] = 50
    
    # For efficient updates, we need to know, for each point y, which (x, v) pairs are affected
    # when y is excluded. y is in other_pts[x, v, :] for various (x, v).
    
    # Let's build a reverse index: for each point y, list of (x, v) such that y is in other_pts[x, v, :]
    # affected_by[y] = list of (x, v)
    
    # This could be large. Each point y is on 50 lines (one for each direction through y), and on each line
    # there are 49 other points. So y affects 50 * 49 = 2450 (x, v) pairs? No, wait.
    # 
    # y is in other_pts[x, v, :] iff y is on the line through x in direction v, and y != x.
    # For a fixed y and direction v, how many x have y in other_pts[x, v, :]?
    # The line through y in direction v has 49 points. For each x on this line (x != y), y is in other_pts[x, v, :].
    # So 48 such x for each v. Total: 50 * 48 = 2400 (x, v) pairs.
    
    # Let's build affected_by as a list of lists
    affected_by = [[] for _ in range(q*q)]
    for x in range(q*q):
        for v in range(50):
            for t in range(q-1):
                y = other_pts[x, v, t]
                affected_by[y].append((x, v))
    
    # Now greedy: try to exclude points in some order
    # Order: try points that are "least critical" - those with more valid directions initially?
    # Or just go in order 0, 1, 2, ...
    
    # For efficiency, let's use a deadline
    start_time = time.time()
    deadline = start_time + 500  # 500 seconds out of 630
    
    # Try to exclude points in order
    # For each point, check if it can be excluded
    excluded = []
    
    for pt in range(q*q):
        if time.time() > deadline:
            break
        
        if not in_N[pt]:
            continue
        
        # Check if pt can be excluded: after exclusion, pt must still have at least 1 valid direction
        # And no other point should lose its last valid direction
        
        # First, check pt's valid directions after exclusion
        # pt's valid directions are those v where all other_pts[pt, v, :] are in N
        # After excluding pt, this doesn't change (pt's validity depends on OTHER points, not itself)
        # Wait, pt's valid directions depend on other points being in N, not on pt itself.
        # So excluding pt doesn't affect pt's own valid directions.
        
        # But excluding pt affects other points: for each (x, v) in affected_by[pt], 
        # if v was valid for x, it might no longer be valid.
        
        # Check if pt has at least 1 valid direction (it should, since it's currently in N and valid)
        # Actually, we need to verify pt currently has at least 1 valid direction
        pt_valid = False
        for v in range(50):
            all_in_N = True
            for t in range(q-1):
                if not in_N[other_pts[pt, v, t]]:
                    all_in_N = False
                    break
            if all_in_N:
                pt_valid = True
                break
        
        if not pt_valid:
            continue
        
        # Now check if excluding pt breaks any other point
        # For each (x, v) in affected_by[pt], if v is currently valid for x and this is x's last valid direction,
        # then excluding pt would break x.
        
        can_exclude = True
        for x, v in affected_by[pt]:
            if x == pt:
                continue
            if not in_N[x]:
                continue
            # Check if v is currently valid for x
            # v is valid for x if all other_pts[x, v, :] are in N
            # Since pt is currently in N, and pt is in other_pts[x, v, :], 
            # v is valid for x only if all other points (including pt) are in N
            # After excluding pt, v will no longer be valid for x
            # So we need to check: is v currently valid for x, and is it x's last valid direction?
            
            # Check if v is currently valid for x
            v_valid_for_x = True
            for t in range(q-1):
                if not in_N[other_pts[x, v, t]]:
                    v_valid_for_x = False
                    break
            
            if not v_valid_for_x:
                continue
            
            # v is currently valid for x. After excluding pt, it won't be.
            # Check if x has another valid direction
            x_has_other = False
            for v2 in range(50):
                if v2 == v:
                    continue
                v2_valid = True
                for t in range(q-1):
                    if not in_N[other_pts[x, v2, t]]:
                        v2_valid = False
                        break
                if v2_valid:
                    x_has_other = True
                    break
            
            if not x_has_other:
                can_exclude = False
                break
        
        if can_exclude:
            in_N[pt] = False
            excluded.append(pt)
    
    # Build the result: N is the set of points still in N
    points = []
    for pt in range(q*q):
        if in_N[pt]:
            x_idx, y_idx = idx_to_pt(pt)
            xa, xb = idx_to_elem(x_idx)
            ya, yb = idx_to_elem(y_idx)
            points.append([[xa, xb], [ya, yb]])
    
    if not points:
        # Fallback: return all points (trivially valid, but worst score)
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
