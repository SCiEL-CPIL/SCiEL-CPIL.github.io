import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy:
    1. Formulate the problem as finding a large set S such that for every point x,
       there is at least one direction d where the line L_{x,d} contains no other points of S.
    2. Use a greedy algorithm with bitsets to track 'zero-directions' (directions where 
       count of other S-points is 0).
    3. A point x can be added to S if for all y != x, the direction connecting x and y 
       is not the ONLY zero-direction for y.
    4. Use multiple random restarts to find a large S.
    """
    p = 7
    q = p * p
    total = q * q
    deadline = time.time() + 580
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    k_const = 3
    
    # Flat index: pt = (xa, xb, ya, yb) -> xa + xb*p + ya*p^2 + yb*p^3
    def flat_to_pt(idx):
        xa = idx % p
        xb = (idx // p) % p
        ya = (idx // (p * p)) % p
        yb = (idx // (p * p * p)) % p
        return (xa, xb, ya, yb)

    # Directions: 0..48 are slopes m in F_49, 49 is vertical
    n_dirs = q + 1
    
    # Precompute direction between any two points
    # dir_between[x, y] = direction index
    # If x == y, direction is 0 (dummy)
    dir_between = np.zeros((total, total), dtype=np.uint8)
    
    # Helper to compute direction from x to y
    # Vertical if x_coord == y_coord
    # Else slope m = (y_y - x_y) / (x_x - y_x) ... wait, line is y - y0 = m(x - x0)
    # m = (y_y - x_y) / (x_x - y_x) is not quite right for vector (1, m).
    # Line through (x0, y0) with direction (1, m) contains (x, y) if y - y0 = m * (x - x0).
    # So m = (y - y0) / (x - x0).
    
    def get_dir(x_idx, y_idx):
        xa, xb, ya, yb = flat_to_pt(x_idx)
        xa2, xb2, ya2, yb2 = flat_to_pt(y_idx)
        
        # Check vertical: x == x2
        if xa == xa2 and xb == xb2:
            return n_dirs - 1 # Vertical direction index 49
        
        # Compute dx = x2 - x
        dxa = (xa2 - xa) % p
        dxb = (xb2 - xb) % p
        # dx is not zero
        
        # Compute dy = y2 - y
        dya = (ya2 - ya) % p
        dyb = (yb2 - yb) % p
        
        # m = dy / dx
        # Need inverse of dx in F_49
        # dx = dxa + dxb * alpha
        # inv_dx = ?
        # (a+b*alpha)(c+d*alpha) = (ac+3bd) + (ad+bc)alpha = 1
        # ac+3bd = 1, ad+bc = 0
        # c = -bd, a(-bd)+b(-bd) ... no.
        # a c + 3 b d = 1
        # a d + b c = 0 => c = -a d / b (if b!=0)
        # If b == 0, dx = a. inv = 1/a.
        # If b != 0:
        # c = -a d / b
        # a(-a d / b) + 3 b d = 1
        # d (3b - a^2/b) = 1
        # d (3b^2 - a^2) / b = 1
        # d = b / (3b^2 - a^2)
        # c = -a / (3b^2 - a^2)
        
        # Precompute inverses? Or just do it on fly? 
        # 2401^2 / 2 * 2401 calls is too slow.
        # We need a faster way.
        
        # Alternative: Precompute lines.
        # We already have line_id logic in the previous code.
        # Let's reuse the line_id precomputation to build dir_between.
        return -1 # Placeholder

    # Let's use the line_id approach from the current program to build dir_between
    # This is O(N * D) to build, which is fine.
    
    # Directions: v = (1, m) for m in F_49, and v = (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, 0, ma, mb))
    directions.append((0, 1, 0, 0))
    
    # line_id[pt, d] = line identifier
    line_id = np.zeros((total, n_dirs), dtype=np.int32)
    
    for d in range(n_dirs):
        vxa, vxb, vya, vyb = directions[d]
        for pt_idx in range(total):
            xa, xb, ya, yb = flat_to_pt(pt_idx)
            if vxa == 0:  # v = (0, 1)
                lid = xa + xb * p
            else:
                # v = (1, m)
                mxa = (vya * xa + vyb * xb * k_const) % p
                mxb = (vya * xb + vyb * xa) % p
                lda = (ya - mxa) % p
                ldb = (yb - mxb) % p
                lid = lda + ldb * p
            line_id[pt_idx, d] = lid

    # Build dir_between
    # For each d, all points with same line_id[pt, d] are on a line with direction d.
    # We can fill dir_between for these points.
    # This is O(N * D) if we do it smartly, or O(N^2) if naive.
    # Naive: For each d, group points by lid. For each group, set dir_between[i, j] = d.
    # Sum of group sizes squared can be large.
    # Group size is 49. 50 directions. 2401 points.
    # 50 * (2401/49) * 49^2 = 50 * 49 * 2401 = ~5.8M operations. This is fine.
    
    for d in range(n_dirs):
        # Get unique line ids and their points
        lids = line_id[:, d]
        unique_lids = np.unique(lids)
        for lid in unique_lids:
            pts = np.where(lids == lid)[0]
            # For all pairs i, j in pts, dir_between[i, j] = d
            # Use broadcasting
            for i in pts:
                for j in pts:
                    if i != j:
                        dir_between[i, j] = d

    # Greedy Algorithm with Bitsets
    best_S_size = 0
    best_S_mask = np.zeros(total, dtype=bool)
    
    rng = np.random.default_rng(42)
    
    # Number of restarts
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        S_mask = np.zeros(total, dtype=bool)
        
        # Z_mask[y] is a bitmask of directions where count of other S-points is 0.
        # Initially all directions are 0-count.
        # Bit i is set if direction i is "zero".
        # n_dirs = 50. We need 64-bit int.
        all_ones = (1 << n_dirs) - 1
        Z_mask = np.full(total, all_ones, dtype=np.uint64)
        Z_size = np.full(total, n_dirs, dtype=np.uint8)
        
        # Random order of points
        order = rng.permutation(total)
        
        for x in order:
            if S_mask[x]:
                continue
                
            # Check if x can be added
            # For all y != x, let d = dir_between[x, y]
            # If bit d is set in Z_mask[y], then this direction is currently "zero" for y.
            # Adding x will make it non-zero.
            # If Z_size[y] == 1 and bit d is set, then y will have 0 zero-directions -> Invalid.
            
            # Vectorized check
            # dirs = dir_between[x, :] (shape total)
            # We need to check y where y != x.
            # Let's create a mask for y != x.
            
            dirs = dir_between[x, :]
            # dirs[x] is 0 (dummy), but we skip x.
            
            # Check condition:
            # valid[y] = NOT ( (Z_mask[y] has bit dirs[y]) AND (Z_size[y] == 1) )
            
            # Bit check: (Z_mask >> dirs) & 1
            # Note: dirs is uint8, Z_mask is uint64.
            # Need to handle shift properly.
            
            # Create a mask for y != x
            y_indices = np.arange(total)
            y_indices = y_indices[y_indices != x]
            
            if len(y_indices) == 0:
                # Should not happen
                continue
                
            dirs_y = dirs[y_indices]
            masks_y = Z_mask[y_indices]
            sizes_y = Z_size[y_indices]
            
            # Check which y have bit set
            # (masks_y >> dirs_y) & 1
            # This works with numpy if types are compatible.
            has_bit = (masks_y >> dirs_y) & 1
            
            # If has_bit is 1 and size is 1, it's invalid
            invalid_mask = (has_bit == 1) & (sizes_y == 1)
            
            if np.any(invalid_mask):
                continue
                
            # Valid to add x
            S_mask[x] = True
            
            # Update Z_mask and Z_size for affected y
            # Affected y are those where has_bit is 1
            affected_indices = y_indices[has_bit == 1]
            
            if len(affected_indices) > 0:
                # Clear the bit
                bits_to_clear = 1 << dirs[affected_indices]
                Z_mask[affected_indices] ^= bits_to_clear
                Z_size[affected_indices] -= 1
        
        # Check if this S is better
        current_size = np.sum(S_mask)
        if current_size > best_S_size:
            best_S_size = current_size
            best_S_mask = S_mask.copy()
            
    # Build N = all points NOT in S
    N_indices = [i for i in range(total) if not best_S_mask[i]]
    
    points = np.array([[[flat_to_pt(i)[0], flat_to_pt(i)[1]],
                        [flat_to_pt(i)[2], flat_to_pt(i)[3]]] for i in N_indices],
                      dtype=np.int64)
    
    return points
