import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Returns an integer array of shape (k, 2, 2) representing points in N.
    """
    p = 7
    q = p * p
    
    # Field operations for F_49 = F_7[alpha], alpha^2 = 3
    # Element represented as (a, b) -> a + b*alpha
    # We map each element to an integer index 0..48
    # index = a * 7 + b
    
    # Precompute field addition and multiplication tables
    # Actually, we just need to generate lines.
    
    # Helper to add two field elements
    def add_f(a, b):
        # a, b are tuples (a_a, a_b), (b_a, b_b)
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def mul_f(a, b):
        # (a1 + a2*alpha) * (b1 + b2*alpha) = a1b1 + (a1b2+a2b1)alpha + a2b2*alpha^2
        # alpha^2 = 3
        a1, a2 = a
        b1, b2 = b
        c0 = (a1 * b1 + a2 * b2 * 3) % p
        c1 = (a1 * b2 + a2 * b1) % p
        return (c0, c1)
    
    # Generate all field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    elem_to_idx = {elem: i for i, elem in enumerate(field_elements)}
    
    # Points in F_49^2
    # Point is (x, y) where x, y in F_49
    # Total q^2 points
    total_points = q * q
    
    # Precompute lines
    # A line is defined by a point (x0, y0) and a direction (1, m) or (0, 1)
    # Directions: m in F_49 (49 directions) + vertical (1 direction) = 50 directions
    
    # For each point, we need to know which lines pass through it.
    # Instead, for the greedy algorithm, we need:
    # When we select a point x, we need to update all points y on the same lines as x.
    
    # Let's precompute for each point index (0..2400), the list of other points on the same lines.
    # Or better: for each point, list of (line_id). And for each line, list of points.
    
    # Let's create a list of all lines.
    # Line represented by set of point indices.
    
    lines = []
    # To avoid duplicates, we can define a canonical line.
    # A line is uniquely determined by its set of points.
    # We can just generate all lines and use a set of frozensets to deduplicate, 
    # but that might be slow.
    # Alternatively, iterate over all points and all slopes, generate the line, 
    # and only add it if the "minimum" point in the line is the current point?
    # Or just generate all and deduplicate at the end.
    
    # Total lines = q^2 * (q+1) / q = q(q+1) = 49 * 50 = 2450 lines.
    # Each line has q=49 points.
    
    # Let's generate lines properly.
    # For each point (x0, y0) and each slope m in F_49:
    #   Line: y = m(x - x0) + y0
    #   Points: (x, y) for x in F_49
    # For vertical lines: x = x0
    #   Points: (x0, y) for y in F_49
    
    # We will store lines as lists of point indices.
    
    line_points = []
    seen_lines = set()
    
    for x0_a in range(p):
        for x0_b in range(p):
            x0 = (x0_a, x0_b)
            for y0_a in range(p):
                for y0_b in range(p):
                    y0 = (y0_a, y0_b)
                    p_idx = elem_to_idx[x0] * q + elem_to_idx[y0]
                    
                    # Slopes m in F_49
                    for m_a in range(p):
                        for m_b in range(p):
                            m = (m_a, m_b)
                            pts = []
                            for x_a in range(p):
                                for x_b in range(p):
                                    x = (x_a, x_b)
                                    # y = m(x - x0) + y0
                                    dx = add_f(x, (p - x0_a % p, p - x0_b % p)) # x - x0
                                    # Actually simpler: dx = (x[0]-x0[0])%p, (x[1]-x0[1])%p
                                    dx = ((x[0] - x0[0]) % p, (x[1] - x0[1]) % p)
                                    mx = mul_f(m, dx)
                                    y = add_f(mx, y0)
                                    pt_idx = elem_to_idx[x] * q + elem_to_idx[y]
                                    pts.append(pt_idx)
                            
                            # Canonicalize line to avoid duplicates
                            # Sort points
                            pts_sorted = tuple(sorted(pts))
                            if pts_sorted not in seen_lines:
                                seen_lines.add(pts_sorted)
                                line_points.append(list(pts_sorted))
                    
                    # Vertical line
                    pts = []
                    for y_a in range(p):
                        for y_b in range(p):
                            y = (y_a, y_b)
                            pt_idx = elem_to_idx[x0] * q + elem_to_idx[y]
                            pts.append(pt_idx)
                    pts_sorted = tuple(sorted(pts))
                    if pts_sorted not in seen_lines:
                        seen_lines.add(pts_sorted)
                        line_points.append(list(pts_sorted))
    
    # Now we have all lines.
    # For each point, find which lines it belongs to.
    point_lines = [[] for _ in range(total_points)]
    for i, line in enumerate(line_points):
        for pt in line:
            point_lines[pt].append(i)
    
    # Greedy Algorithm
    # We want to maximize |S|.
    # avail_lines[pt] = number of lines through pt that are "free" (contain no points in S).
    # Initially all 50.
    # When we add pt to S, we decrement avail_lines for all other points on all lines through pt.
    
    def run_greedy(seed):
        random.seed(seed)
        avail = np.full(total_points, len(point_lines[0]), dtype=np.int32)
        in_S = np.zeros(total_points, dtype=bool)
        
        # To break ties or order, we can use a priority queue or just random selection.
        # Let's try: pick random point with avail > 0.
        # Or: pick point with minimum avail > 0.
        
        # Optimization: maintain a list of active points.
        # Since total points is small (2401), we can just scan.
        
        S = []
        active = set(range(total_points))
        
        while active:
            # Find point with min avail > 0?
            # Or random?
            # Let's try min avail (most constrained first)
            min_avail = 1000
            best_pt = -1
            # Random tie breaker
            candidates = []
            for pt in active:
                if avail[pt] > 0:
                    if avail[pt] < min_avail:
                        min_avail = avail[pt]
                        candidates = [pt]
                    elif avail[pt] == min_avail:
                        candidates.append(pt)
            
            if not candidates:
                break
            
            pt = random.choice(candidates)
            
            # Add pt to S
            in_S[pt] = True
            S.append(pt)
            active.remove(pt)
            
            # Update neighbors
            # For each line through pt, for each other point y in line, decrement avail[y]
            for line_idx in point_lines[pt]:
                for y in line_points[line_idx]:
                    if y != pt and y in active:
                        avail[y] -= 1
                        if avail[y] == 0:
                            # y can no longer be added
                            # We don't remove from active immediately to keep logic simple?
                            # No, if avail[y]==0, it's not a candidate.
                            # But it might become >0? No, avail only decreases.
                            # So we can remove it from active to save time?
                            # Yes.
                            active.remove(y)
        
        return len(S)

    # Run multiple times with different seeds to find best S
    # Time limit: 630s.
    # Each run is fast. Let's estimate.
    # Worst case |S| ~ 1000.
    # 1000 * 50 lines * 49 points = 2.45M updates.
    # Python loop might take ~0.1s per run.
    # We can run ~1000 runs in 100s.
    
    start_time = time.time()
    best_size = 0
    best_S = []
    
    # We need to return the points of N, which is complement of S.
    # So we need to store the actual S set.
    
    def run_greedy_return_S(seed):
        random.seed(seed)
        avail = np.full(total_points, len(point_lines[0]), dtype=np.int32)
        in_S = np.zeros(total_points, dtype=bool)
        
        S = []
        active = set(range(total_points))
        
        while active:
            min_avail = 1000
            candidates = []
            for pt in active:
                if avail[pt] > 0:
                    if avail[pt] < min_avail:
                        min_avail = avail[pt]
                        candidates = [pt]
                    elif avail[pt] == min_avail:
                        candidates.append(pt)
            
            if not candidates:
                break
            
            pt = random.choice(candidates)
            
            in_S[pt] = True
            S.append(pt)
            active.remove(pt)
            
            for line_idx in point_lines[pt]:
                for y in line_points[line_idx]:
                    if y != pt and y in active:
                        avail[y] -= 1
                        if avail[y] == 0:
                            active.remove(y)
        
        return S

    num_runs = 1000
    run_id = 0
    while time.time() - start_time < 600:
        S = run_greedy_return_S(run_id)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
        run_id += 1
        
        # If we find a very large set, maybe stop early?
        # Max possible S?
        # Each point in S "uses up" one line for its neighbors.
        # Rough bound: q^2 / q = q? No.
        # If S is large, lines are blocked.
        
    # Construct N from best_S
    in_S = np.zeros(total_points, dtype=bool)
    for pt in best_S:
        in_S[pt] = True
    
    points = []
    for i in range(total_points):
        if not in_S[i]:
            x_idx = i // q
            y_idx = i % q
            x = field_elements[x_idx]
            y = field_elements[y_idx]
            points.append([[x[0], x[1]], [y[0], y[1]]])
    
    return np.array(points, dtype=np.int64)

# Run once to check? No, the function is called by evaluator.
# We just define it.
