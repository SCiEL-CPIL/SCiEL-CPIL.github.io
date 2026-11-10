import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    n_points = p ** 4  # 2401
    n_dirs = p * p + 1  # 50
    k = 3

    start_time = time.time()
    deadline = start_time + 570  # 60s safety margin

    # Field operations
    def field_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        c0 = (a0 * b0 + a1 * b1 * k) % p
        c1 = (a0 * b1 + a1 * b0) % p
        return (c0, c1)
        
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)

    def field_to_int(f):
        return f[0] * p + f[1]
        
    def int_to_field(i):
        return (i // p, i % p)

    # Encode/Decode
    def encode(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
        
    def decode(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        xa = idx // p
        return (xa, xb, ya, yb)

    # Precompute line members for each direction and line id
    # line_members[d][lid] = np.array of point indices
    line_members = [None] * n_dirs
    # Also need for each point y and direction d, which line lid it belongs to?
    # Or just iterate through all lines in direction d and check if y is in them?
    # Better: precompute point_to_line[d][y] = lid
    point_to_line = [None] * n_dirs
    
    for d in range(n_dirs):
        line_members[d] = [None] * q
        point_to_line[d] = np.zeros(n_points, dtype=np.int32)
        
        if d < n_dirs - 1:
            ma = d // p
            mb = d % p
            m = (ma, mb)
            # Calculate line id for each point
            line_ids = np.zeros(n_points, dtype=np.int32)
            for idx in range(n_points):
                xa, xb, ya, yb = decode(idx)
                x = (xa, xb)
                y = (ya, yb)
                mx = field_mul(m, x)
                c = field_add(y, field_neg(mx))
                lid = field_to_int(c)
                line_ids[idx] = lid
                point_to_line[d][idx] = lid
                
            # Group points by line id
            for lid in range(q):
                mask = line_ids == lid
                line_members[d][lid] = np.where(mask)[0]
        else:
            # Vertical lines: x is constant
            # line id is x
            for idx in range(n_points):
                xa, xb, ya, yb = decode(idx)
                lid = field_to_int((xa, xb))
                point_to_line[d][idx] = lid
                
            for lid in range(q):
                # Find points with this x
                # xa = lid // p, xb = lid % p
                xa_target = lid // p
                xb_target = lid % p
                indices = []
                for idx in range(n_points):
                    xa, xb, ya, yb = decode(idx)
                    if xa == xa_target and xb == xb_target:
                        indices.append(idx)
                line_members[d][lid] = np.array(indices, dtype=np.int32)

    def run_greedy_addition(order):
        # C[x, m] = count of points in S on line through x with slope m
        C = np.zeros((n_points, n_dirs), dtype=np.int32)
        count_0 = np.full(n_points, n_dirs, dtype=np.int32)
        count_1 = np.zeros(n_points, dtype=np.int32)
        in_S = np.zeros(n_points, dtype=bool)
        
        # Initially, all x not in S, count_0=50, count_1=0.
        # Valid if count_0 > 0. All valid.
        
        for y in order:
            if in_S[y]:
                continue
                
            # Check if adding y keeps all points valid
            # We need to simulate changes
            # Changes: for each d, for each x in line(d, y), C[x, d] += 1
            # Update count_0, count_1
            
            # Store old values to revert if necessary
            # We only need to revert if we find an invalid point
            # To be efficient, we can do a dry run check first?
            # Or just apply and check, then revert if bad.
            
            # Let's collect the points that will be affected
            # affected_points = set()
            # changes = [] # (x, d, old_c0, old_c1, old_count0, old_count1)
            
            valid = True
            # We need to check validity of ALL points that are affected
            # A point x is affected if it shares a line with y in some direction d.
            # This is ALL points? No.
            # For a fixed d, x is affected if x is on the same line as y.
            # Union over all d of lines through y is the whole space?
            # Yes, for any x != y, there is a unique line through x and y with some slope.
            # And for vertical, if x has same x-coord.
            # So ALL points are affected?
            # Wait. C[x, d] changes only if x is on the line through y with slope d.
            # For a specific d, only 49 points are affected.
            # Total affected points is union of 50 lines of length 49.
            # Is it the whole space?
            # For any x, is there a d such that x is on line(y, d)?
            # Yes, the line connecting x and y has a specific slope (or vertical).
            # So yes, ALL points x have their C[x, d_x] incremented for the specific d_x connecting them.
            # So we must update C for all 2401 points?
            # That makes it O(N^2) per addition. 2401 * 2401 = 5.7M ops.
            # 2401 additions * 5.7M = 13 Billion ops. Too slow in Python.
            
            # Optimization:
            # We don't need to update C for all points if we can check validity incrementally?
            # Validity of x depends on count_0[x] and count_1[x].
            # count_0[x] decreases if C[x, d] goes 0->1.
            # count_1[x] increases if C[x, d] goes 0->1.
            # count_1[x] decreases if C[x, d] goes 1->2.
            
            # If we add y, for each x, exactly ONE direction d_x is affected.
            # So for each x, we only look at C[x, d_x].
            
            # Let's precompute for each pair (y, x), the direction d_xy.
            # d_xy is the slope of line yx.
            # If x == y, all d are affected?
            # If x == y, y is on ALL lines through y.
            # So for x=y, C[y, d] increases for ALL d.
            
            # Precompute direction_matrix[y][x] = d
            # 2401x2401 matrix of directions.
            # Memory: 2401 * 2401 * 2 bytes ~ 11MB. OK.
            
            # Let's precompute this outside the loop.
            
            if not hasattr(run_greedy_addition, 'dir_mat'):
                run_greedy_addition.dir_mat = np.zeros((n_points, n_points), dtype=np.int32)
                for y in range(n_points):
                    for x in range(n_points):
                        if x == y:
                            # Special case: all directions
                            # We can't store 50 values in one int.
                            # Handle x==y separately in the loop.
                            run_greedy_addition.dir_mat[y][x] = -1
                        else:
                            # Find slope m such that x is on line through y with slope m
                            # Or vertical
                            xa_y, xb_y, ya_y, yb_y = decode(y)
                            xa_x, xb_x, ya_x, yb_x = decode(x)
                            
                            # Check vertical
                            if (xa_y, xb_y) == (xa_x, xb_x):
                                run_greedy_addition.dir_mat[y][x] = n_dirs - 1
                            else:
                                # m = (y_x - y_y) / (x_x - x_y)
                                # y_x = x, y_y = y in field terms
                                # dx = x_x - x_y
                                # dy = y_x - y_y
                                # m = dy / dx
                                dx = field_sub((xa_x, xb_x), (xa_y, xb_y))
                                dy = field_sub((ya_x, yb_x), (ya_y, yb_y))
                                # m = dy * inv(dx)
                                # Need field_inv
                                # F_49 is small, precompute inverses
                                pass
                # Precompute inverses
                inv = [0] * q
                for i in range(q):
                    a, b = int_to_field(i)
                    # (a+b*alpha)(c+d*alpha) = (ac+3bd) + (ad+bc)*alpha
                    # Need (ac+3bd)=1, (ad+bc)=0
                    # If a != 0, c = 1/a, d = -b/a / (1 - 3/a^2)?
                    # Simpler: brute force for 49 elements
                    for j in range(q):
                        c, d = int_to_field(j)
                        c0 = (a*c + b*d*k) % p
                        c1 = (a*d + b*c) % p
                        if c0 == 1 and c1 == 0:
                            inv[i] = j
                            break
                run_greedy_addition.inv = inv
                
                # Recalculate dir_mat with inv
                for y in range(n_points):
                    for x in range(n_points):
                        if x == y:
                            run_greedy_addition.dir_mat[y][x] = -1
                        else:
                            xa_y, xb_y, ya_y, yb_y = decode(y)
                            xa_x, xb_x, ya_x, yb_x = decode(x)
                            
                            if (xa_y, xb_y) == (xa_x, xb_x):
                                run_greedy_addition.dir_mat[y][x] = n_dirs - 1
                            else:
                                dx = field_sub((xa_x, xb_x), (xa_y, xb_y))
                                dy = field_sub((ya_x, yb_x), (ya_y, yb_y))
                                dx_int = field_to_int(dx)
                                dy_int = field_to_int(dy)
                                m_int = (dy_int * run_greedy_addition.inv[dx_int]) % q
                                run_greedy_addition.dir_mat[y][x] = m_int

            dir_mat = run_greedy_addition.dir_mat
            inv = run_greedy_addition.inv
            
            # Back to greedy loop
            # For y:
            # x=y: all d affected.
            # x!=y: d = dir_mat[y][x] affected.
            
            # We need to check if any x becomes invalid.
            # We can do this in one pass over all x.
            
            # Prepare for revert
            # We will store the old state of count_0, count_1 for all x?
            # That's 2401 * 2 ints. Small.
            old_count_0 = count_0.copy()
            old_count_1 = count_1.copy()
            old_in_S = in_S.copy()
            
            # Apply changes
            for x in range(n_points):
                if x == y:
                    # All d affected
                    for d in range(n_dirs):
                        c = C[x, d]
                        if c == 0:
                            count_0[x] -= 1
                            count_1[x] += 1
                        elif c == 1:
                            count_1[x] -= 1
                        C[x, d] = c + 1
                else:
                    d = dir_mat[y][x]
                    c = C[x, d]
                    if c == 0:
                        count_0[x] -= 1
                        count_1[x] += 1
                    elif c == 1:
                        count_1[x] -= 1
                    C[x, d] = c + 1
                    
            # Check validity
            # x is valid if:
            # (in_S[x] and count_1[x] > 0) or (not in_S[x] and count_0[x] > 0)
            # Note: in_S[y] is now True (simulated)
            # For x != y, in_S[x] is unchanged.
            
            for x in range(n_points):
                if x == y:
                    is_in = True
                else:
                    is_in = in_S[x]
                
                if is_in:
                    if count_1[x] == 0:
                        valid = False
                        break
                else:
                    if count_0[x] == 0:
                        valid = False
                        break
            
            if valid:
                in_S[y] = True
                # C and counts are already updated
            else:
                # Revert
                count_0 = old_count_0
                count_1 = old_count_1
                in_S = old_in_S
                # Revert C
                for x in range(n_points):
                    if x == y:
                        for d in range(n_dirs):
                            C[x, d] -= 1
                    else:
                        d = dir_mat[y][x]
                        C[x, d] -= 1

        return in_S

    best_S = None
    best_size = 0
    
    # Multiple restarts
    while time.time() < deadline:
        order = list(range(n_points))
        random.shuffle(order)
        S = run_greedy_addition(order)
        size = np.sum(S)
        if size > best_size:
            best_size = size
            best_S = S.copy()
            
    # Convert best_S to N (complement)
    if best_S is None:
        # Fallback: N = All
        N_mask = np.ones(n_points, dtype=bool)
    else:
        N_mask = ~best_S
        
    # If N is empty, fallback
    if not np.any(N_mask):
        N_mask = np.ones(n_points, dtype=bool)
        
    N_points = []
    for idx in range(n_points):
        if N_mask[idx]:
            xa, xb, ya, yb = decode(idx)
            N_points.append([[xa, xb], [ya, yb]])
            
    if not N_points:
        N_points = [[[0,0],[0,0]]]
        
    return np.array(N_points, dtype=np.int64)
