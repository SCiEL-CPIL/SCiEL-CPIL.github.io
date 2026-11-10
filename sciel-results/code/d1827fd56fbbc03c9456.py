# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (q=49, p=7) by greedily maximizing the complement.
    
    Field: F_49 = F_7[alpha] with alpha^2 = 3.
    A point is [[xa, xb], [ya, yb]] representing (xa+xb*alpha, ya+yb*alpha).
    
    Nikodym condition: For every point x in F_49^2, there exists a direction v
    (either (1,m) for some m in F_49, or (0,1)) such that all points x + t*v
    for t in F_49, t != 0, are in N (i.e., not in the complement C).
    
    We greedily build C (the complement) by adding points that maintain validity.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    # Precompute all points as flat indices 0..2400
    # Point index = ((xa * p + xb) * q) + (ya * p + yb)
    # Or more simply: xa*p^3 + xb*p^2 + ya*p + yb
    
    deadline = time.time() + 550  # safety margin
    
    # Field arithmetic helpers
    def to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * q) + (ya * p + yb)
    
    def from_idx(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return (xa, xb, ya, yb)
    
    # Precompute lines: for each point and each direction, list the other points on that line
    # Directions: (1, m) for m in F_49 (49 directions) + (0, 1) (1 direction) = 50 directions
    # For direction (1, m): line through x is {x + t*(1,m) : t in F_49}
    # For direction (0, 1): line through x is {x + t*(0,1) : t in F_49}
    
    # Precompute all lines as sets of point indices
    # lines[point][dir_idx] = list of other points on that line through point
    
    # Actually, for efficiency, let's precompute for each point and direction,
    # the set of other points on the line.
    
    # There are 2401 points and 50 directions, so 120050 line memberships to precompute.
    # Each line has 49 points, so 48 "other" points per (point, direction) pair.
    
    # Let's precompute: for each direction, and each point, the 48 other points on the line.
    
    # Directions: index 0..48 are (1, m) for m in F_49, index 49 is (0, 1)
    # m is represented as (ma, mb)
    
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, ma, mb))  # (dx_a, dy_a, dy_b) where v = (1, ma+mb*alpha)
    directions.append((0, 1, 0))  # v = (0, 1)
    
    num_dirs = len(directions)  # 50
    
    # Precompute line_others[point_idx][dir_idx] = list of 48 other point indices
    line_others = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    
    for dir_idx in range(num_dirs):
        dx_a, dy_a, dy_b = directions[dir_idx]
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        point_idx = to_idx(xa, xb, ya, yb)
                        count = 0
                        for t in range(q):
                            if t == 0:
                                continue
                            # t = ta + tb*alpha
                            ta = t % p
                            tb = t // p
                            # new_x = x + t * dx
                            # new_y = y + t * (dy_a + dy_b * alpha)
                            if dx_a == 1:
                                # new_xa = xa + t_a, new_xb = xb + t_b
                                nxa = (xa + ta) % p
                                nxb = (xb + tb) % p
                            else:
                                nxa = xa
                                nxb = xb
                            # new_y = y + t * dy where dy = dy_a + dy_b*alpha
                            # t * dy = (ta + tb*alpha) * (dy_a + dy_b*alpha)
                            # = ta*dy_a + (ta*dy_b + tb*dy_a)*alpha + tb*dy_b*alpha^2
                            # = ta*dy_a + (ta*dy_b + tb*dy_a)*alpha + tb*dy_b*3
                            # = (ta*dy_a + 3*tb*dy_b) + (ta*dy_b + tb*dy_a)*alpha
                            ny_a = (ta * dy_a + 3 * tb * dy_b) % p
                            ny_b = (ta * dy_b + tb * dy_a) % p
                            nya = (ya + ny_a) % p
                            nyb = (yb + ny_b) % p
                            other_idx = to_idx(nxa, nxb, nya, nyb)
                            line_others[point_idx, dir_idx, count] = other_idx
                            count += 1
    
    # Now greedy: start with C = empty, try to add points one by one
    # A point p can be added to C if for every point x in F_49^2,
    # there exists a direction d such that none of the other points on line x+d are in C.
    
    # This is expensive to check naively. Let's use a smarter approach.
    
    # For each point x and direction d, let's track how many of the 48 other points are in C.
    # x is covered by direction d if count[x][d] == 0.
    # x is covered if there exists d with count[x][d] == 0.
    
    # When we add point p to C, it affects count[x][d] for all x,d where p is on the line through x in direction d.
    # p is on the line through x in direction d iff x is on the line through p in direction d.
    # So for each direction d, and for each of the 48 other points x on the line through p in direction d,
    # we increment count[x][d].
    
    # Maintain: count[x][d] = number of points in C on the line through x in direction d (excluding x itself)
    # x is valid if min_d count[x][d] == 0
    
    count = np.zeros((total_points, num_dirs), dtype=np.int32)
    in_C = np.zeros(total_points, dtype=bool)
    
    # Initially all counts are 0, so all points are covered.
    
    # Greedy: try to add points in some order. Try all points, add if possible.
    # For efficiency, iterate multiple passes.
    
    best_complement_size = 0
    best_C = set()
    
    # Try to add points. We'll iterate through points in a fixed order and try to add them.
    # If adding a point would make some x uncovered, skip it.
    
    # For checking if adding p is valid:
    # After adding p, for each direction d, and for each x on the line through p in direction d (x != p),
    # count[x][d] increases by 1.
    # We need to ensure that for all x, there still exists d with count[x][d] == 0.
    # This is equivalent to: for all x on any line through p, the new count[x][d] > 0 for all d 
    # would be bad. But we only need to check x that are on lines through p.
    
    # Actually, only points x that are on some line through p (in some direction) are affected.
    # For a given direction d, the points x affected are the 48 other points on the line through p in direction d.
    # For those x, count[x][d] increases by 1.
    # x becomes uncovered if after the increase, count[x][d'] > 0 for all d'.
    # So x becomes uncovered if before the increase, count[x][d'] == 0 for only d = d' (the affected direction)
    # and count[x][d'] was 0, now becomes 1, and all other d' had count > 0.
    
    # So: x becomes uncovered iff before adding p, x had exactly one direction d0 with count[x][d0] == 0,
    # and d0 is the direction such that p is on the line through x in direction d0.
    
    # Let's precompute for each point x, the set of directions where count[x][d] == 0.
    # Initially all 50 directions have count 0.
    
    # For efficiency, let's just do a straightforward check but with early termination.
    
    # Actually, let's just try a simpler greedy: iterate through all 2401 points in order,
    # and for each, check if adding it keeps all points covered. If yes, add it.
    
    # The check: for each direction d, for each of the 48 other points x on line through p in direction d,
    # check if x would become uncovered. x becomes uncovered if for all d', count[x][d'] > 0 after update.
    # After update, count[x][d] increases by 1 (for the specific d), other d' unchanged.
    # So x becomes uncovered iff before update, count[x][d] == 0 and for all d' != d, count[x][d'] > 0.
    
    # Precompute for each x, the number of directions with count[x][d] == 0.
    zero_count = np.full(total_points, num_dirs, dtype=np.int32)  # initially all 50
    
    # For the check when adding p:
    # For each direction d (0..49):
    #   For each x in line_others[p][d] (48 points):
    #     if zero_count[x] == 1 and count[x][d] == 0:
    #       # x has only direction d with count 0, and we're about to make it 1
    #       # x would become uncovered -> invalid
    #       return False
    # If we get here, it's valid.
    
    # Then update: for each d, for each x in line_others[p][d]:
    #   if count[x][d] == 0: zero_count[x] -= 1
    #   count[x][d] += 1
    #   in_C[p] = True
    
    # This is O(50 * 48) = 2400 operations per point, times 2401 points = ~5.7M operations per pass.
    # With multiple passes, should be fast enough.
    
    # Let's do multiple passes until no more points can be added or deadline.
    
    C = set()
    
    for pass_num in range(50):
        if time.time() > deadline:
            break
        added_any = False
        for p in range(total_points):
            if time.time() > deadline:
                break
            if p in C:
                continue
            # Check if adding p is valid
            valid = True
            for d in range(num_dirs):
                if zero_count[p] == 0:
                    # p itself would be uncovered? No, p's own count doesn't matter for p's coverage.
                    # Actually, p's coverage depends on count[p][d] for all d.
                    # But we're checking other points x.
                    pass
                for i in range(q - 1):
                    x = line_others[p, d, i]
                    if zero_count[x] == 1 and count[x, d] == 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                # Add p to C
                C.add(p)
                added_any = True
                # Update counts
                for d in range(num_dirs):
                    for i in range(q - 1):
                        x = line_others[p, d, i]
                        if count[x, d] == 0:
                            zero_count[x] -= 1
                        count[x, d] += 1
        
        if not added_any:
            break
    
    # Build the complement set C, and N = all points not in C
    # Return N as array of shape (k, 2, 2)
    
    N_points = []
    for idx in range(total_points):
        if idx not in C:
            xa, xb, ya, yb = from_idx(idx)
            N_points.append([[xa, xb], [ya, yb]])
    
    return np.array(N_points, dtype=np.int64)


# EVOLVE-BLOCK-END
