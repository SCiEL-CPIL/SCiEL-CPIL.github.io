import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    # Precompute all lines in the allowed directions
    # Directions: (1, m) for m in F_49, and (0, 1)
    # Represent field elements as (a, b) pairs where value = a + b*alpha, alpha^2 = 3
    
    # Field operations for F_49 = F_7[alpha], alpha^2 = 3
    def add(x, y):
        return ((x[0] + y[0]) % p, (x[1] + y[1]) % p)
    
    def sub(x, y):
        return ((x[0] - y[0]) % p, (x[1] - y[1]) % p)
    
    def mul(x, y):
        # (a + b*alpha)(c + d*alpha) = ac + (ad+bc)*alpha + bd*alpha^2 = (ac + 3bd) + (ad+bc)*alpha
        a, b = x
        c, d = y
        return ((a*c + 3*b*d) % p, (a*d + b*c) % p)
    
    # Generate all field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Precompute all lines for each direction
    # For direction (1, m): line through point (x0, y0) is {(x0 + t, y0 + t*m) : t in F_49}
    # For direction (0, 1): line through point (x0, y0) is {(x0, y0 + t) : t in F_49}
    
    # We'll represent points as indices 0..2400
    # Point index: xa + xb*q + ya*q*q + yb*q*q*q... let's use a flat index
    # idx = xa + xb*q + ya*q*q + yb*q*q*q
    # But q=49, so idx = xa + xb*49 + ya*2401 + yb*2401*49... that's too big
    # Actually: idx = xa + xb*q + ya*q*q + yb*q*q*q doesn't work since q*q*q = 117649 > 2401
    # Let's use: idx = (xa, xb, ya, yb) -> xa + xb*q + ya*q*q + yb*q*q... no
    # Simple: idx = xa + xb*q + ya*q*q + yb*q*q*q is wrong
    # Correct: there are p^4 = 2401 points, so idx = xa + xb*p + ya*p*p + yb*p*p*p
    # p=7, so idx = xa + xb*7 + ya*49 + yb*343, range 0..2400
    
    def point_to_idx(xa, xb, ya, yb):
        return xa + xb*p + ya*p*p + yb*p*p*p
    
    def idx_to_point(idx):
        xa = idx % p
        xb = (idx // p) % p
        ya = (idx // (p*p)) % p
        yb = (idx // (p*p*p)) % p
        return (xa, xb, ya, yb)
    
    # Precompute lines: for each direction and each point, which points are on the line
    # But that's 50 directions * 2401 points * 49 points = too much to store explicitly
    # Instead, for each point, precompute the set of points on each of the 50 lines through it
    
    # Actually, for efficiency, let's precompute for each direction d and each point x,
    # the list of other points on the line through x in direction d
    
    # Directions: 49 directions of type (1, m) + 1 direction (0, 1) = 50 directions
    # m ranges over F_49, i.e., (ma, mb) for ma, mb in 0..6
    
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, ma, mb))  # direction (1, m) where m = (ma, mb)
    directions.append((0, 0, 1))  # direction (0, 1)
    
    # For each point and each direction, compute the other 48 points on the line
    # This is 2401 * 50 * 48 = ~5.7 million entries, manageable
    
    # Let's precompute: lines_through[point_idx][dir_idx] = list of 48 other point indices
    lines_through = {}
    for idx in range(total_points):
        xa, xb, ya, yb = idx_to_point(idx)
        point = ((xa, xb), (ya, yb))
        lines_through[idx] = []
        for d in directions:
            dx, dma, dmb = d
            other_points = []
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    # t = (ta, tb), new point = x + t*v
                    # v = (dx, dma + dmb*alpha)
                    # t*v = (ta*dx + 3*tb*dma, ta*dmb + tb*dma) ... wait
                    # t = ta + tb*alpha, v = dx + (dma + dmb*alpha)*alpha = dx + dma*alpha + dmb*alpha^2 = (dx + 3*dmb) + dma*alpha
                    # t*v = (ta + tb*alpha)((dx + 3*dmb) + dma*alpha)
                    # = ta*(dx+3*dmb) + ta*dma*alpha + tb*alpha*(dx+3*dmb) + tb*dma*alpha^2
                    # = ta*(dx+3*dmb) + ta*dma*alpha + tb*(dx+3*dmb)*alpha + tb*dma*3
                    # = (ta*(dx+3*dmb) + 3*tb*dma) + (ta*dma + tb*(dx+3*dmb))*alpha
                    tva = (ta*(dx + 3*dmb) + 3*tb*dma) % p
                    tvb = (ta*dma + tb*(dx + 3*dmb)) % p
                    nxa = (xa + tva) % p
                    nyb = (yb + tvb) % p  # wait, y coordinate
                    # Point is (x, y) = ((xa,xb), (ya,yb))
                    # New point: x + t*v_x, y + t*v_y where v = (v_x, v_y)
                    # v_x = (dx, 0) if dx=1, or (0,0) if dx=0... 
                    # Actually v = (1, m) means v_x = 1, v_y = m
                    # So new_x = x + t*1 = x + t
                    # new_y = y + t*m
                    # For direction (0,1): v_x = 0, v_y = 1
                    # new_x = x, new_y = y + t
                    
                    if dx == 1:
                        # new_x = x + t
                        nx_a = (xa + ta) % p
                        nx_b = (xb + tb) % p
                        # new_y = y + t*m, m = (dma, dmb)
                        # t*m = (ta + tb*alpha)(dma + dmb*alpha) = (ta*dma + 3*tb*dmb) + (ta*dmb + tb*dma)*alpha
                        tma = (ta*dma + 3*tb*dmb) % p
                        tmb = (ta*dmb + tb*dma) % p
                        ny_a = (ya + tma) % p
                        ny_b = (yb + tmb) % p
                    else:  # dx == 0
                        nx_a = xa
                        nx_b = xb
                        # new_y = y + t*1 = y + t
                        ny_a = (ya + ta) % p
                        ny_b = (yb + tb) % p
                    
                    other_points.append(point_to_idx(nx_a, nx_b, ny_a, ny_b))
            lines_through[idx].append(other_points)
    
    # Now, validity check: for each point x, check if there exists a direction d such that
    # all points in lines_through[x][d] are NOT in S (i.e., are in N)
    # S is represented as a boolean array of length 2401
    
    def is_valid(S):
        """S is a set of indices (boolean array). Check Nikodym validity."""
        for x in range(total_points):
            found = False
            for d in range(len(directions)):
                all_in_N = True
                for y in lines_through[x][d]:
                    if S[y]:
                        all_in_N = False
                        break
                if all_in_N:
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Start with S = empty (N = all points), which is always valid
    S = np.zeros(total_points, dtype=bool)
    
    # Try to add points to S one by one
    # For each point, try adding it to S and check if validity is maintained
    # Use a greedy approach with randomization
    
    deadline = time.time() + 580  # 580 second deadline
    
    # Precompute for each point, which other points are "critical" - i.e., if we add x to S,
    # which other points' validity might be affected?
    # A point y's validity depends on lines through y. If x is on a line through y, and that line
    # was the only valid line for y, then adding x to S breaks y's validity.
    
    # For efficiency, track for each point y, how many valid directions it has
    # valid_count[y] = number of directions d such that all other points on line y,d are in N
    
    def compute_valid_counts(S):
        counts = np.zeros(total_points, dtype=np.int32)
        for y in range(total_points):
            count = 0
            for d in range(len(directions)):
                all_in_N = True
                for z in lines_through[y][d]:
                    if S[z]:
                        all_in_N = False
                        break
                if all_in_N:
                    count += 1
            counts[y] = count
        return counts
    
    valid_counts = compute_valid_counts(S)
    
    # Now greedily add points to S
    # A point x can be added to S if:
    # 1. x itself still has at least one valid direction after adding x to S
    #    (i.e., there's a direction d where all OTHER points on the line are in N, and x is not on those lines... 
    #     wait, x IS on the line, but we're checking if the other points are in N)
    #    Actually, when we add x to S, for x's own validity: we need a direction d where all points in lines_through[x][d] are in N.
    #    This is independent of x being in S or not, since lines_through[x][d] doesn't include x.
    #    So x's validity doesn't change when we add x to S!
    # 2. For every other point y that has x on one of its lines, if that line was the only valid line for y,
    #    then y's validity breaks.
    
    # So: x can be added to S iff for all y such that x is in lines_through[y][d] for some d,
    # valid_counts[y] > 1 (i.e., y has at least 2 valid directions, so losing one doesn't make it 0)
    # OR y has valid_counts[y] == 1 but the only valid direction for y doesn't have x on it... 
    # wait, if x is on the line through y in direction d, and that's y's only valid direction, then adding x to S breaks y.
    
    # More precisely: x can be added to S iff for all y and all d such that x ∈ lines_through[y][d],
    # it's NOT the case that valid_counts[y] == 1 and d is y's only valid direction.
    
    # Let's precompute: for each point x, which (y, d) pairs have x in lines_through[y][d]
    # affected_by[x] = list of (y, d) where x is on the line through y in direction d
    
    affected_by = [[] for _ in range(total_points)]
    for y in range(total_points):
        for d in range(len(directions)):
            for x in lines_through[y][d]:
                affected_by[x].append((y, d))
    
    # Also need to know, for each y, which direction is its only valid one if valid_counts[y] == 1
    # Let's track valid_dirs[y] = set of valid directions for y
    
    def compute_valid_dirs(S):
        vdirs = [set() for _ in range(total_points)]
        for y in range(total_points):
            for d in range(len(directions)):
                all_in_N = True
                for z in lines_through[y][d]:
                    if S[z]:
                        all_in_N = False
                        break
                if all_in_N:
                    vdirs[y].add(d)
        return vdirs
    
    valid_dirs = compute_valid_dirs(S)
    
    # Now greedy: repeatedly find a point x that can be safely added to S
    # x can be added iff for all (y, d) in affected_by[x], valid_dirs[y] - {d} is non-empty
    # i.e., |valid_dirs[y]| > 1 for all such (y,d), OR d is not in valid_dirs[y]
    
    def can_add(x, valid_dirs):
        for y, d in affected_by[x]:
            if d in valid_dirs[y] and len(valid_dirs[y]) == 1:
                return False
        return True
    
    # Greedy loop
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try points in random order
        order = list(range(total_points))
        random.shuffle(order)
        for x in order:
            if S[x]:
                continue
            if can_add(x, valid_dirs):
                # Add x to S
                S[x] = True
                # Update valid_dirs for all affected points
                for y, d in affected_by[x]:
                    if d in valid_dirs[y]:
                        valid_dirs[y].discard(d)
                        if len(valid_dirs[y]) == 0:
                            # This shouldn't happen if can_add was correct
                            pass
                changed = True
                break  # Restart with fresh order after a change
    
    # Final validity check
    if not is_valid(S):
        # Fallback: return all points in N
        S = np.zeros(total_points, dtype=bool)
    
    # Build result
    points = []
    for idx in range(total_points):
        if not S[idx]:
            xa, xb, ya, yb = idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
