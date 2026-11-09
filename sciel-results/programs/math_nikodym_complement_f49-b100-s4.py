# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 630.0
START_TIME = time.time()


def _time_left() -> float:
    return DEADLINE - (time.time() - START_TIME)


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 where F_49 = F_7[alpha], alpha^2 = 3.
    Points are [[x_a, x_b], [y_a, y_b]] representing x = x_a + x_b*alpha,
    y = y_a + y_b*alpha.
    
    Strategy: Use the unital construction to get a valid Nikodym set with
    complement size 148, then attempt to add more points to the complement
    S while maintaining validity.
    
    The base construction:
    - Unital U: points where 2*x_a + y_a^2 - 3*y_b^2 ≡ 0 (mod 7)
    - Special set Y: y_a is a square mod 7 (y_a in {0,1,2,4})
    - S_base = U \\ B where B = U ∩ {y in Y}
    - N_base = F_49^2 \\ S_base
    
    We then try to greedily add points to S (remove from N) while ensuring
    the Nikodym property holds.
    """
    p = 7
    k = 3  # alpha^2 = 3

    # Precompute all field elements as (a, b) pairs
    # Field element: a + b*alpha, where a, b in 0..6
    
    # Multiplication table: field_mult[(a1,b1)][(a2,b2)] = (a3,b3)
    # (a1+b1*alpha)*(a2+b2*alpha) = a1*a2 + (a1*b2+a2*b1)*alpha + b1*b2*alpha^2
    # = (a1*a2 + 3*b1*b2) + (a1*b2+a2*b1)*alpha (mod 7)
    
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
    
    # Enumerate all points in F_49^2
    # Each point is ((xa, xb), (ya, yb))
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))
    
    n_total = len(all_points)  # 2401
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Directions: (1, m) for all m in F_49, plus (0, 1)
    # A direction v = (vx, vy) where vx, vy are field elements
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))  # (1, m)
    directions.append(((0, 1), (0, 0)))  # (0, 1)
    
    n_dirs = len(directions)  # 50
    
    # For each point and each direction, precompute the set of other points on the line
    # line_other[point_idx][dir_idx] = set of indices of other points on the line
    
    # Actually, let's precompute for each direction, the line containing each point
    # and the other points on that line
    
    # line_membership[dir_idx][point_idx] = frozenset of indices on same line
    # This uses too much memory. Instead, let's compute on the fly or use a different approach.
    
    # Better approach: For each direction, group points by which line they're on
    # For direction (1, m): lines are parametrized by y - m*x = c
    # For direction (0, 1): lines are x = c
    
    # Let's precompute for each direction and each point, the list of other points on its line
    # Store as a flat structure for speed
    
    # For efficiency, let's represent points as integers 0..2400
    # Point ((xa,xb),(ya,yb)) -> index = xa*49 + xb*7 + ya*7 + yb? 
    # Actually: idx = ((xa * 7 + xb) * 7 + ya) * 7 + yb
    
    def point_to_int(pt):
        (xa, xb), (ya, yb) = pt
        return ((xa * 7 + xb) * 7 + ya) * 7 + yb
    
    def int_to_point(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return ((xa, xb), (ya, yb))
    
    # Build mapping
    pt_int_map = {pt: point_to_int(pt) for pt in all_points}
    int_pt_map = {point_to_int(pt): pt for pt in all_points}
    
    # Precompute line neighbors for each direction
    # For each direction d, and each point i, line_neighbors[d][i] = list of other point indices on same line
    
    line_neighbors = []
    for d_idx in range(n_dirs):
        vx, vy = directions[d_idx]
        
        # For each point, find the line it's on
        # For direction (1, m): line through point p has constant (y - m*x)
        # For direction (0, 1): line through point p has constant x
        
        # Map from line_id to list of points on that line
        line_map = {}
        
        for pt_idx, pt in enumerate(all_points):
            (x, y) = pt
            # Compute line invariant
            if vx == (1, 0):  # direction (1, m) where m = vy
                # Line: y - m*x = const
                # m*x: field_mul(m, x)
                mx = field_mul(vy, x)
                # y - m*x
                c = field_sub(y, mx)
            else:  # direction (0, 1)
                # Line: x = const
                c = x
            
            if c not in line_map:
                line_map[c] = []
            line_map[c].append(pt_idx)
        
        # For each point, the other points on its line
        neigh = [[] for _ in range(n_total)]
        for c, pts_on_line in line_map.items():
            for i in pts_on_line:
                neigh[i] = [j for j in pts_on_line if j != i]
        
        line_neighbors.append(neigh)
    
    if _time_left() < 300:
        # Not enough time for optimization, just return base construction
        pass
    
    # Base construction: unital-based
    squares = set()
    for u in range(p):
        squares.add((u * u) % p)
    # squares = {0, 1, 2, 4}
    
    # S_base: points in unital where y_a is NOT a square
    S_base = set()
    for pt_idx, pt in enumerate(all_points):
        (xa, xb), (ya, yb) = pt
        in_unital = (2 * xa + ya * ya - k * yb * yb) % p == 0
        in_y_special = ya in squares
        if in_unital and not in_y_special:
            S_base.add(pt_idx)
    
    # N_base = all points not in S_base
    # Verify base is valid (should be, by construction)
    
    # Now try to add more points to S
    S = set(S_base)
    
    # For a point to be added to S, we need to check that for that point,
    # there exists a direction where all other points on the line are in N (not in S)
    
    # Actually, the condition is: for EVERY point x (in N or S), there exists a direction v
    # such that all other points on the line through x with direction v are in N.
    
    # So if we add a point to S, we need to ensure:
    # 1. The added point itself has a witness direction
    # 2. All other points that previously relied on a direction that now has this new S-point
    #    on their line still have some valid witness direction
    
    # This is complex. Let's use a simpler approach: try adding points one at a time
    # and verify the full property after each addition.
    
    # For verification, we need to check for every point x in F_49^2:
    # exists direction v such that for all t != 0, x + t*v not in S
    # i.e., for all y in line(x, v) \ {x}, y not in S
    
    def is_valid_N(S_set):
        """Check if N = F_49^2 \ S_set is a valid Nikodym set."""
        for pt_idx in range(n_total):
            # Check if there's a valid witness direction
            valid = False
            for d_idx in range(n_dirs):
                # Check all other points on the line
                ok = True
                for other_idx in line_neighbors[d_idx][pt_idx]:
                    if other_idx in S_set:
                        ok = False
                        break
                if ok:
                    valid = True
                    break
            if not valid:
                return False
        return True
    
    if _time_left() < 250:
        # Return base construction
        points = []
        for pt_idx in range(n_total):
            if pt_idx not in S:
                pt = all_points[pt_idx]
                (xa, xb), (ya, yb) = pt
                points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)
    
    # Try to add points to S greedily
    # Candidates: points not currently in S
    candidates = [i for i in range(n_total) if i not in S]
    
    # Shuffle candidates for variety (but deterministic seed)
    np.random.seed(42)
    np.random.shuffle(candidates)
    
    added_count = 0
    for cand in candidates:
        if _time_left() < 150:
            break
        
        # Try adding cand to S
        S_trial = S | {cand}
        
        # Quick check: does cand have a witness?
        cand_valid = False
        for d_idx in range(n_dirs):
            ok = True
            for other_idx in line_neighbors[d_idx][cand]:
                if other_idx in S_trial:
                    ok = False
                    break
            if ok:
                cand_valid = True
                break
        
        if not cand_valid:
            continue
        
        # Full verification (expensive, but necessary)
        if is_valid_N(S_trial):
            S = S_trial
            added_count += 1
    
    # Build final result
    points = []
    for pt_idx in range(n_total):
        if pt_idx not in S:
            pt = all_points[pt_idx]
            (xa, xb), (ya, yb) = pt
            points.append([[xa, xb], [ya, yb]])
    
    arr = np.array(points, dtype=np.int64)
    return arr


# EVOLVE-BLOCK-END
