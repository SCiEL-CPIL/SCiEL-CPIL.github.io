# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy removal heuristic.
    Starts with the full set and removes points that are not essential for the Nikodym property.
    """
    p = 7
    q = p * p
    k = 3 # alpha^2 = 3

    # Field operations in F_p[alpha]
    def f_add(a, b):
        return (a[0] + b[0]) % p, (a[1] + b[1]) % p
    
    def f_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        return (a0*b0 + a1*b1*k) % p, (a0*b1 + a1*b0) % p
    
    def f_inv(a):
        if a[0] == 0 and a[1] == 0:
            return (0, 0)
        # Norm is a0^2 - k*a1^2
        norm = (a[0]*a[0] - k*a[1]*a[1]) % p
        # Inverse of norm in F_p
        inv_norm = pow(norm, p-2, p)
        return (a[0]*inv_norm) % p, (-a[1]*inv_norm) % p

    # Precompute all points and directions
    # Point index: ((xa*p + xb)*p + ya)*p + yb
    # Directions: (1, m) for m in F_q, and (0, 1)
    # Total directions: q + 1
    
    # Map (a, b) to index
    def idx(a, b):
        return (a[0]*p + a[1])*p + b[0]*p + b[1]
        
    def unidx(i):
        yb = i % p
        i //= p
        ya = i % p
        i //= p
        xb = i % p
        i //= p
        xa = i % p
        return (xa, xb), (ya, yb)

    # Precompute line membership for each direction
    # lines[dir_idx][point_idx] = set of points on the line through point_idx in direction dir_idx
    # Actually, for each point and direction, we need to know if the other points on the line are in N.
    # Let's precompute for each point x and direction v, the set of other points on the line.
    
    # dirs: list of (vx, vy)
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append(((1, 0), (ma, mb))) # (1, m)
    dirs.append(((0, 1), (0, 0))) # (0, 1) is vertical, wait. v=(0,1) means x is const, y varies.
    # Wait, v=(1, m) means x=t, y=tm+x0. 
    # v=(0, 1) means x=x0, y=t+y0.
    # The problem says: v = (1, m) for some m in F_49 OR v = (0, 1).
    # So there are q+1 directions.
    
    # Precompute line members
    # line_points[dir_idx][point_idx] = list of point indices on the line (excluding the point itself? No, including)
    # We need to check if ALL other points are in N.
    
    # To speed up, we can represent the set N as a boolean array of size q^2.
    # For a point x to be removable, for EVERY direction v, the line L(x, v) \ {x} must be subset of N.
    # This is equivalent to: for every direction v, count of points in N on L(x, v) is q-1.
    
    # Let's precompute for each direction v, the lines.
    # There are q lines per direction.
    # For each line, we store the list of q points.
    
    # lines_by_dir[dir_idx] = list of q lines, each line is a list of q point indices.
    lines_by_dir = []
    for vx, vy in dirs:
        lines = []
        # A line is determined by its intersection with a "base" set of lines.
        # For v=(1, m), lines are determined by y-intercept? No, x-intercept?
        # y - m*x = c. 
        # For v=(0, 1), lines are determined by x-intercept. x = c.
        
        # Let's just iterate over all points and group them.
        # But that's O(q^2 * q) which is 2401 * 49 ~ 117k operations per direction? No.
        # Grouping:
        # For v=(1, m): key is (y - m*x) in F_q.
        # For v=(0, 1): key is x in F_q.
        
        # Let's compute the key for each point.
        keys = np.zeros(q*q, dtype=int)
        
        if vx[0] == 1: # (1, m)
            # key = y - m*x
            # y - m*x = (ya, yb) - m*(xa, xb)
            for i in range(q*q):
                (xa, xb), (ya, yb) = unidx(i)
                mx = f_mul(((1, 0), m), (xa, xb)) # m * (xa, xb)
                # Wait, m is (ma, mb).
                # Let's use the actual m from the loop.
                pass
            
            # Vectorized approach for keys might be tricky with field ops.
            # Let's precompute field ops tables.
            # f_mul_table[a_idx][b_idx] = c_idx
            # f_sub_table[a_idx][b_idx] = c_idx
            # f_add_table[a_idx][b_idx] = c_idx
            
            # This will be memory heavy but fast.
            # q^2 = 2401. 2401x2401 table is ~5.7M entries. int32 is 23MB. OK.
            
            pass # We will build tables outside the loop
        else: # (0, 1)
            # key = x
            pass
        
        lines_by_dir.append(None) # Placeholder

    # Build field operation tables
    # Indices 0..2400
    # add_table[i, j] = (i+j) mod q (in field sense)
    # sub_table[i, j] = (i-j) mod q
    # mul_table[i, j] = (i*j) mod q
    
    # To save memory and time, we might just use the unidx/idx functions if we are careful.
    # But for greedy removal, we need to update counts for lines.
    
    # Let's try a different approach:
    # 1. Start with N = All points.
    # 2. For each point x, calculate "removability".
    #    x is removable if for all v, line(x, v) \ {x} subset N.
    #    Initially, all points are removable? No.
    #    If N is full, then for any x, line(x, v) \ {x} is full, so it is in N.
    #    So initially ALL points are removable.
    #    We can remove any point.
    #    Once we remove a point x, it might make other points non-removable.
    
    # Strategy:
    # Maintain count_in_N_on_line[dir_idx][line_idx]
    # Initially q for all.
    # A point x is removable if for all dir_idx, count_in_N_on_line[dir_idx][line_of_x(dir_idx)] == q-1.
    # (Since x is in N, the count includes x. If count is q-1, it means all OTHERS are in N).
    
    # When we remove x:
    # For each dir_idx, let L be the line containing x.
    # Decrement count_in_N_on_line[dir_idx][L] by 1.
    # This might make points on L non-removable.
    
    # We need to efficiently find which points become non-removable.
    # A point y becomes non-removable if there exists a dir_idx such that count_in_N_on_line[dir_idx][line_of_y(dir_idx)] drops below q-1.
    # Since we only decrement by 1, the count drops from q-1 to q-2.
    # So y becomes non-removable if it was on a line with count q-1 in some direction, and we removed a point from that line.
    
    # Data structures:
    # counts[dir_idx, line_idx]
    # For each point x, we need to know line_idx for each dir_idx.
    # line_idx_of_point[dir_idx, point_idx]
    
    # Precomputation:
    # 1. Field tables (add, sub, mul) for F_49.
    # 2. For each direction, map point to line index.
    
    # Let's implement field tables.
    # q = 49.
    # add_table = np.zeros((q, q), dtype=int)
    # etc.
    
    # Time budget check
    start_time = time.time()
    deadline = start_time + 600
    
    # 1. Build field tables
    # Map (a,b) to i
    # i = a*p + b
    # Wait, the problem says pair (a,b) stands for a + b*alpha.
    # My previous idx used 4 coords for a point in F_49^2.
    # Field element index: 0..48.
    
    # Field element index: a*p + b
    def f_idx(a, b): return a*p + b
    def f_unidx(i): return (i // p, i % p)
    
    add_tbl = np.zeros((q, q), dtype=np.int32)
    sub_tbl = np.zeros((q, q), dtype=np.int32)
    mul_tbl = np.zeros((q, q), dtype=np.int32)
    
    for i in range(q):
        ai, bi = f_unidx(i)
        for j in range(q):
            aj, bj = f_unidx(j)
            # Add
            ri = (ai + aj) % p
            rj = (bi + bj) % p
            add_tbl[i, j] = f_idx(ri, rj)
            # Sub
            si = (ai - aj) % p
            sj = (bi - bj) % p
            sub_tbl[i, j] = f_idx(si, sj)
            # Mul
            mi = (ai*aj + bi*bj*k) % p
            mj = (ai*bj + bi*aj) % p
            mul_tbl[i, j] = f_idx(mi, mj)
            
    # 2. Map points in F_49^2 to indices 0..2400
    # Point index: x_idx * q + y_idx
    # x_idx, y_idx in 0..48
    
    # 3. Precompute line indices for each direction
    # Directions:
    # (1, m) for m in F_49. q directions.
    # (0, 1). 1 direction.
    # Total q+1 directions.
    
    # For dir (1, m): Line is y - m*x = c.
    # c = y - m*x.
    # For a point (x, y), c_idx = sub_tbl[y_idx, mul_tbl[m_idx, x_idx]]
    # Line index is c_idx (0..48).
    
    # For dir (0, 1): Line is x = c.
    # c = x.
    # Line index is x_idx (0..48).
    
    # line_idx[dir_idx, point_idx]
    # point_idx = x_idx * q + y_idx
    # dir_idx: 0..q-1 correspond to (1, m) with m_idx = dir_idx.
    # dir_idx = q corresponds to (0, 1).
    
    line_idx = np.zeros((q+1, q*q), dtype=np.int32)
    
    # For each point
    for x_idx in range(q):
        for y_idx in range(q):
            p_idx = x_idx * q + y_idx
            
            # Direction (1, m)
            for m_idx in range(q):
                # c = y - m*x
                mx = mul_tbl[m_idx, x_idx]
                c = sub_tbl[y_idx, mx]
                line_idx[m_idx, p_idx] = c
                
            # Direction (0, 1)
            line_idx[q, p_idx] = x_idx
            
    # 4. Initialize counts
    # counts[dir_idx, line_idx] = q
    counts = np.full((q+1, q), q, dtype=np.int32)
    
    # 5. Greedy removal
    # We maintain a set of "removable" points.
    # Initially all points are removable.
    # We want to remove as many as possible.
    # Removing a point x:
    #   For each dir, L = line_idx[dir, x].
    #   counts[dir, L] -= 1.
    #   If counts[dir, L] becomes q-2 (from q-1), then all points y on L (y != x) might become non-removable.
    #   Specifically, y is non-removable if ANY of its lines has count < q-1.
    
    # To track removability efficiently:
    # For each point y, let bad_count[y] = number of directions where counts[dir, line_idx[dir, y]] < q-1.
    # y is removable iff bad_count[y] == 0.
    
    bad_count = np.zeros(q*q, dtype=np.int32)
    removable = np.ones(q*q, dtype=bool)
    
    # We need to update bad_count when counts change.
    # When counts[dir, L] drops from q-1 to q-2:
    #   For all y on L, bad_count[y] += 1.
    #   If bad_count[y] becomes 1, removable[y] = False.
    
    # We need a fast way to iterate over points on a line.
    # points_on_line[dir_idx, line_idx] = list of point indices.
    
    # Precompute points_on_line
    # This will take some memory but is necessary for speed.
    # (q+1) * q lists, each of length q.
    # Total q^2 * (q+1) ~ 117k integers. Very small.
    
    points_on_line = []
    for d in range(q+1):
        lines = [[] for _ in range(q)]
        for p_idx in range(q*q):
            l = line_idx[d, p_idx]
            lines[l].append(p_idx)
        points_on_line.append(lines)
        
    # Now the greedy loop
    # We want to pick a point to remove.
    # Any removable point works.
    # However, removing a point that is on many "critical" lines (lines with count q-1) might be bad?
    # Actually, we just want to maximize removals.
    # A simple strategy: pick the first removable point.
    # Or: pick the point that is removable and has the highest "impact"?
    # Let's just pick the first one for simplicity, or random.
    # Random might be better to avoid getting stuck in local minima?
    # But since we are removing points, the set of removable points shrinks.
    # Let's try picking the point with the smallest index first.
    
    # To speed up finding removable points, we can keep a list of removable indices.
    # When a point becomes non-removable, we remove it from the list?
    # Or we just scan. Scanning 2400 points is fast.
    
    removed = np.zeros(q*q, dtype=bool)
    num_removed = 0
    
    while True:
        if time.time() > deadline - 10:
            break
            
        # Find a removable point not yet removed
        # We can scan the array.
        # To make it more interesting, maybe pick one that is "least useful"?
        # But for now, let's just pick the first one.
        
        candidate = -1
        for i in range(q*q):
            if removable[i] and not removed[i]:
                candidate = i
                break
                
        if candidate == -1:
            break
            
        # Remove candidate
        removed[candidate] = True
        num_removed += 1
        
        # Update counts and bad_count
        for d in range(q+1):
            L = line_idx[d, candidate]
            counts[d, L] -= 1
            if counts[d, L] == q - 2:
                # This line was critical. All other points on it are now non-removable (unless they were already).
                # Actually, they become non-removable if this was their FIRST bad line.
                # We increment bad_count for all points on this line.
                for y in points_on_line[d][L]:
                    if y == candidate:
                        continue
                    bad_count[y] += 1
                    if bad_count[y] == 1:
                        removable[y] = False
                        
    # Construct the result
    # N is the set of points NOT removed.
    # points = [[x_a, x_b], [y_a, y_b]]
    
    res = []
    for i in range(q*q):
        if not removed[i]:
            x_idx = i // q
            y_idx = i % q
            xa, xb = f_unidx(x_idx)
            ya, yb = f_unidx(y_idx)
            res.append([[xa, xb], [ya, yb]])
            
    # If nothing was removed (should not happen), return full set
    if not res:
        # Fallback to full set? No, k>=1.
        # If we removed all, N is empty, which is invalid.
        # But we start with all removable, so we remove at least one?
        # Wait, if we remove one, others become non-removable.
        # Can we remove ALL?
        # If N is empty, condition "points other than x all belong to N" is vacuously true?
        # "whose points other than x all belong to N". If N is empty, there are no points other than x in the line?
        # No, the line has q points. x is one. There are q-1 others.
        # If N is empty, the others are NOT in N. So condition fails.
        # So N cannot be empty.
        
        # What is the minimum size of N?
        # If we remove too many, we break the condition.
        # Our algorithm stops when no removable points are left.
        # So N will be valid.
        
        # If res is empty, it means we removed everything?
        # That would imply we removed the last point while it was still removable.
        # If N has 1 point x, is x removable?
        # Line through x has q-1 other points. They must be in N. But N={x}. So others not in N.
        # So x is NOT removable.
        # So we should stop with at least 1 point?
        # Actually, if N={x}, bad_count[x] should be > 0.
        # So removable[x] would be False.
        # So the loop would stop.
        
        # So res should not be empty.
        pass

    return np.array(res, dtype=np.int64)

# EVOLVE-BLOCK-END
