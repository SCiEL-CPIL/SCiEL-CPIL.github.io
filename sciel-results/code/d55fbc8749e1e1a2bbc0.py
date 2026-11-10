# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 620.0
START_TIME = time.time()


def _time_left() -> float:
    return DEADLINE - (time.time() - START_TIME)


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 where F_49 = F_7[alpha], alpha^2 = 3.
    
    Uses a direct construction based on the unital, then greedily removes
    points from N (adds to complement S) while maintaining validity.
    Optimized with bitsets for fast validity checks.
    """
    p = 7
    k = 3  # alpha^2 = 3
    q = p * p  # 49

    # Field operations on pairs (a, b) representing a + b*alpha
    def field_mul(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        a3 = (a1 * a2 + k * b1 * b2) % p
        b3 = (a1 * b2 + a2 * b1) % p
        return (a3, b3)

    def field_add(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 + a2) % p, (b1 + b2) % p)

    def field_sub(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 - a2) % p, (b1 - b2) % p)

    # Enumerate all points in F_49^2 as index 0..2400
    # idx = ((xa * 7 + xb) * 7 + ya) * 7 + yb
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))

    n_total = len(all_points)  # 2401

    def point_to_int(pt):
        (xa, xb), (ya, yb) = pt
        return ((xa * 7 + xb) * 7 + ya) * 7 + yb

    def int_to_point(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return ((xa, xb), (ya, yb))

    # Directions: (1, m) for all m in F_49 (49 directions) plus (0, 1) (1 direction)
    # Total: 50 directions
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))  # (1, m)
    directions.append(((0, 1), (0, 0)))  # (0, 1)

    n_dirs = len(directions)  # 50
    mask_50 = (1 << n_dirs) - 1

    # Precompute line masks for each direction and point
    # line_mask[d][i] is a bitmask of all points j != i on the line through i with direction d
    line_masks = []
    for d_idx in range(n_dirs):
        vx, vy = directions[d_idx]

        # Map from line invariant to list of point indices
        line_map = {}

        for pt_idx, pt in enumerate(all_points):
            (x, y) = pt
            # Compute line invariant
            if vx == (1, 0):  # direction (1, m) where m = vy
                # Line: y - m*x = const
                mx = field_mul(vy, x)
                c = field_sub(y, mx)
            else:  # direction (0, 1)
                # Line: x = const
                c = x

            if c not in line_map:
                line_map[c] = []
            line_map[c].append(pt_idx)

        # Build masks
        masks = [0] * n_total
        for c, pts_on_line in line_map.items():
            # Create a mask for all points on this line
            line_mask_val = 0
            for j in pts_on_line:
                line_mask_val |= (1 << j)
            
            # For each point i on the line, the mask of OTHER points is line_mask_val ^ (1 << i)
            for i in pts_on_line:
                masks[i] = line_mask_val ^ (1 << i)
        
        line_masks.append(masks)

    # Base construction: unital-based
    # Squares mod 7: {0, 1, 2, 4}
    squares = set()
    for u in range(p):
        squares.add((u * u) % p)

    # S_base: points in unital where y_a is NOT a square
    S_bits = 0
    for pt_idx, pt in enumerate(all_points):
        (xa, xb), (ya, yb) = pt
        in_unital = (2 * xa + ya * ya - k * yb * yb) % p == 0
        in_y_special = ya in squares
        if in_unital and not in_y_special:
            S_bits |= (1 << pt_idx)

    # Precompute valid_dirs for initial S
    valid_dirs = [0] * n_total
    for j in range(n_total):
        mask = 0
        for d in range(n_dirs):
            if (S_bits & line_masks[d][j]) == 0:
                mask |= (1 << d)
        valid_dirs[j] = mask

    # Precompute c_disrupts[c][j]
    # c_disrupts[c][j] is a bitmask of directions d such that c is in line_masks[d][j]
    c_disrupts = []
    for c in range(n_total):
        disrupts = [0] * n_total
        c_bit = 1 << c
        for j in range(n_total):
            mask = 0
            for d in range(n_dirs):
                if line_masks[d][j] & c_bit:
                    mask |= (1 << d)
            disrupts[j] = mask
        c_disrupts.append(disrupts)

    if _time_left() < 100:
        # Not enough time, return base construction
        points = []
        for pt_idx in range(n_total):
            if not (S_bits & (1 << pt_idx)):
                pt = all_points[pt_idx]
                (xa, xb), (ya, yb) = pt
                points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)

    # Greedily add points to S (remove from N)
    candidates = [i for i in range(n_total) if not (S_bits & (1 << i))]
    
    # To make it faster, we can sort candidates by how "safe" they are.
    # A point c is safer if it disrupts fewer points' unique witnesses.
    # Let's just iterate.
    
    i = 0
    while i < len(candidates):
        if _time_left() < 50:
            break
            
        c = candidates[i]
        
        # Check if c is still valid (not in S)
        if S_bits & (1 << c):
            i += 1
            continue
            
        # Check if c can be added
        valid = True
        for j in range(n_total):
            # If valid_dirs[j] has no bit that is NOT in c_disrupts[c][j], then c blocks j
            # i.e., valid_dirs[j] is a subset of c_disrupts[c][j]
            if (valid_dirs[j] & (mask_50 ^ c_disrupts[c][j])) == 0:
                valid = False
                break
                
        if valid:
            # Add c to S
            S_bits |= (1 << c)
            
            # Update valid_dirs
            for j in range(n_total):
                valid_dirs[j] &= (mask_50 ^ c_disrupts[c][j])
                
        i += 1

    # Build final result
    points = []
    for pt_idx in range(n_total):
        if not (S_bits & (1 << pt_idx)):
            pt = all_points[pt_idx]
            (xa, xb), (ya, yb) = pt
            points.append([[xa, xb], [ya, yb]])

    arr = np.array(points, dtype=np.int64)
    return arr

# EVOLVE-BLOCK-END
