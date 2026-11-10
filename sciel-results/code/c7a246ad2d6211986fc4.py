# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total = q * q  # 2401

    # Precompute field operations and line memberships
    # alpha^2 = 3 in F_7
    k_const = 3

    # Precompute for each slope m (0..48, where 48 represents vertical), 
    # and each point x (0..2400), the set of other points on the line through x in direction m.
    # Direction (1, m) for m != 48, direction (0, 1) for m = 48 (vertical).
    # We represent the line points as a numpy array of booleans or a sorted list.
    
    # Point index: idx = (xa * p + xb) * q + (ya * p + yb)
    # Or more simply: idx = xa * p^3 + xb * p^2 + ya * p + yb
    
    def point_idx(xa, xb, ya, yb):
        return (xa * p + xb) * q + (ya * p + yb)
    
    def idx_to_coords(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // q) % p
        xa = idx // (q * p)
        return xa, xb, ya, yb

    # Precompute lines: for each direction d (0..q, where q=48 is vertical),
    # and each point x, the set of points on the line through x in direction d (excluding x).
    # Actually, for the Nikodym condition, for a point x and direction v, 
    # the punctured line is {x + t*v : t in F_q, t != 0}.
    # We need to check if all these points are in N (i.e., NOT in S).
    
    # Let's precompute for each point x and each direction d, the list of other points on the line.
    # There are 2401 points and 49 directions (48 slopes + vertical).
    # That's 2401 * 49 = 117,649 (point, direction) pairs.
    # For each, we store up to 48 other points.
    
    # Directions: m = 0, 1, ..., 48 (where 48 means vertical, i.e., v = (0,1))
    # For m = 0..47: v = (1, m) where m is represented as (ma, mb)
    # For m = 48: v = (0, 1)
    
    # Precompute field addition and multiplication tables for F_49 = F_7[alpha], alpha^2 = 3
    # Element: (a, b) represents a + b*alpha
    def add_f(a1, b1, a2, b2):
        return ((a1 + a2) % p, (b1 + b2) % p)
    
    def mul_f(a1, b1, a2, b2):
        # (a1 + b1*alpha) * (a2 + b2*alpha) = a1*a2 + (a1*b2 + a2*b1)*alpha + b1*b2*alpha^2
        # = (a1*a2 + b1*b2*k_const) + (a1*b2 + a2*b1)*alpha
        a = (a1 * a2 + b1 * b2 * k_const) % p
        b = (a1 * b2 + a2 * b1) % p
        return (a, b)
    
    # Precompute line points: lines[x][d] = array of point indices on the line through x in direction d, excluding x
    # Use a list of arrays
    lines = np.zeros((total, q + 1, q - 1), dtype=np.int32)
    
    # For each point x
    for idx in range(total):
        xa, xb, ya, yb = idx_to_coords(idx)
        
        # For each direction
        for d in range(q + 1):
            if d < q:
                # v = (1, m) where m = (d // p, d % p)
                ma, mb = d // p, d % p
                # v = (1, ma + mb*alpha)
                # t = (ta, tb) for ta, tb in 0..6, not both 0
                # x + t*v = (xa + ta*1, xb + ta*ma + tb*mb*... wait
                # t*v = (ta, tb) * (1, ma + mb*alpha) = (ta, ta*(ma+mb*alpha) + tb)
                # Wait, t*v where v=(1, m): t = t0 + t1*alpha, v = (1, m)
                # t*v = (t0+t1*alpha, (t0+t1*alpha)*m) = (t0+t1*alpha, t0*m + t1*m*alpha)
                # m = ma + mb*alpha
                # (t0+t1*alpha)*(ma+mb*alpha) = t0*ma + t0*mb*alpha + t1*ma*alpha + t1*mb*alpha^2
                # = (t0*ma + t1*mb*3) + (t0*mb + t1*ma)*alpha
                # So t*v = (t0 + t1*alpha, (t0*ma + 3*t1*mb) + (t0*mb + t1*ma)*alpha)
                # x + t*v = (xa + t0, xb + t1, ya + t0*ma + 3*t1*mb, yb + t0*mb + t1*ma)
                cnt = 0
                for t0 in range(p):
                    for t1 in range(p):
                        if t0 == 0 and t1 == 0:
                            continue
                        nx_a = (xa + t0) % p
                        nx_b = (xb + t1) % p
                        ny_a = (ya + t0 * ma + 3 * t1 * mb) % p
                        ny_b = (yb + t0 * mb + t1 * ma) % p
                        lines[idx, d, cnt] = point_idx(nx_a, nx_b, ny_a, ny_b)
                        cnt += 1
            else:
                # Vertical: v = (0, 1)
                # t*v = (0, t) where t = (t0, t1)
                # x + t*v = (xa, xb, ya + t0, yb + t1)
                cnt = 0
                for t0 in range(p):
                    for t1 in range(p):
                        if t0 == 0 and t1 == 0:
                            continue
                        ny_a = (ya + t0) % p
                        ny_b = (yb + t1) % p
                        lines[idx, d, cnt] = point_idx(xa, xb, ny_a, ny_b)
                        cnt += 1
    
    # Now, for a given complement S (set of points NOT in N), the Nikodym condition is:
    # For every point x:
    #   If x in S: there exists d such that all points on the line through x in direction d (except x) are in N,
    #              i.e., none of lines[x][d] are in S.
    #   If x not in S: there exists d such that all points on the line through x in direction d (except x) are in N,
    #              i.e., none of lines[x][d] are in S.
    # Wait, re-reading: "every point x of F_q^2 (in N or not) must lie on a line ... whose points other than x all belong to N"
    # So for ALL x, there exists a direction d such that the punctured line through x in direction d is contained in N.
    # Equivalently, for all x, exists d such that no point in lines[x][d] is in S.
    
    # So S is valid iff for every x, there exists d such that lines[x][d] ∩ S = ∅.
    
    # Strategy: Start with a known valid S from the seed, then try to expand S.
    # Seed S: points in unital U where y is NOT in Y_special.
    
    # Compute seed S
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add((u * u % p, v))
    
    S = set()
    for idx in range(total):
        xa, xb, ya, yb = idx_to_coords(idx)
        in_unital = (2 * xa + ya * ya - yb * yb * k_const) % p == 0
        in_y_special = (ya, yb) in y_special
        if in_unital and not in_y_special:
            S.add(idx)
    
    seed_size = len(S)
    best_S = set(S)
    best_size = seed_size
    
    start_time = time.time()
    deadline = start_time + 500  # 500s safety margin
    
    # Try to expand S by adding random points
    # For a point p to be added to S, we need to check that for every x in the new S,
    # there still exists a direction d with lines[x][d] ∩ S_new = ∅.
    
    # Precompute for each point x, the set of directions that are "clean" for x given current S.
    # A direction d is clean for x if lines[x][d] ∩ S = ∅.
    
    def is_valid_S(S_set):
        """Check if S_set is a valid complement."""
        for x in range(total):
            # Check if there exists d such that lines[x][d] ∩ S_set = ∅
            found = False
            for d in range(q + 1):
                line_pts = lines[x, d, :]
                if not any(lp in S_set for lp in line_pts):
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Optimize: for each x, precompute which directions are clean
    def get_clean_directions(x, S_set):
        clean = []
        for d in range(q + 1):
            line_pts = lines[x, d, :]
            if not any(lp in S_set for lp in line_pts):
                clean.append(d)
        return clean
    
    # First verify seed is valid
    if not is_valid_S(best_S):
        # Should not happen, but fallback
        pass
    
    # Try random expansions
    all_points = list(range(total))
    import random
    random.seed(42)
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Try to add 1-5 random points
        num_add = random.randint(1, 5)
        candidates = random.sample([p for p in all_points if p not in best_S], min(num_add, len([p for p in all_points if p not in best_S])))
        
        new_S = set(best_S)
        for p in candidates:
            new_S.add(p)
        
        if len(new_S) <= best_size:
            continue
        
        if is_valid_S(new_S):
            best_S = new_S
            best_size = len(best_S)
    
    # Construct N = all points not in S
    N = [idx for idx in range(total) if idx not in best_S]
    
    # Convert to array of shape (k, 2, 2)
    result = np.zeros((len(N), 2, 2), dtype=np.int64)
    for i, idx in enumerate(N):
        xa, xb, ya, yb = idx_to_coords(idx)
        result[i, 0, 0] = xa
        result[i, 0, 1] = xb
        result[i, 1, 0] = ya
        result[i, 1, 1] = yb
    
    return result


# EVOLVE-BLOCK-END
