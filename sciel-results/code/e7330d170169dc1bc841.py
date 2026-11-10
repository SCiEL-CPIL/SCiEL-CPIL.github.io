# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using the Hermitian Unital strategy.
    
    Strategy:
    1. Identify the Hermitian Unital U in F_49^2.
    2. We want to find S subset of U such that S is a valid Nikodym complement.
       Condition: For every x in F_49^2, there is a line L through x such that L \ {x} is disjoint from S.
       - If x in S, L must be tangent to S at x. Since S subset U, and tangents to U are disjoint from U\{u}, 
         any tangent to U at x works if x is in S.
       - If x not in S, L must be disjoint from S.
         If x in U \ S, the tangent T_x is disjoint from S (since T_x \cap U = {x} and x not in S).
         If x not in U, we need a line L through x such that L \cap S = empty.
         This implies L \cap U subset U \ S = R.
    3. So we need to find R subset U (minimizing |R| to maximize |S|) such that 
       every x not in U lies on a line L with L \cap U subset R.
    4. Greedy Set Cover with costs:
       - Universe: X = F_49^2 \ U
       - Sets: Lines L. Cost(L) = |(L \cap U) \ R|. Benefit(L) = |L \cap Remaining|.
       - Iteratively pick best line, update R and Remaining.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    p = 7
    q = p * p
    k_const = 3  # alpha^2 = 3

    # Field operations for F_49 = F_7[alpha], alpha^2 = 3
    # Point (a, b) represents a + b*alpha
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k_const) % p, (a[0] * b[1] + a[1] * b[0]) % p)

    def trace(a):
        # Tr(a + b*alpha) = (a + b*alpha) + (a + b*alpha)^7 = 2a (since b*alpha^7 = b*alpha^3 = b*3*alpha, wait)
        # (a+b*alpha)^7 = a^7 + b^7*alpha^7 = a + b*alpha^3 = a + 3b*alpha.
        # Sum: 2a + 4b*alpha? No.
        # In F_q^2, Tr(x) = x + x^q.
        # (a + b*alpha)^7 = a + b*alpha^7.
        # alpha^7 = alpha^(2*3+1) = (alpha^2)^3 * alpha = 3^3 * alpha = 27 * alpha = 6 * alpha = -alpha.
        # So (a+b*alpha)^7 = a - b*alpha.
        # Tr = 2a.
        return (2 * a[0]) % p

    def norm(a):
        # N(a) = a * a^q = (a + b*alpha)(a - b*alpha) = a^2 - b^2 * alpha^2 = a^2 - 3b^2
        return (a[0] * a[0] - a[1] * a[1] * k_const) % p

    num_points = q * q
    
    # Precompute point indices
    # idx = ((xa * p + xb) * p + ya) * p + yb
    # x = (xa, xb), y = (ya, yb)
    
    # Identify Unital U: Tr(x) + Norm(y) = 0
    unital_indices = set()
    for xa in range(p):
        for xb in range(p):
            x = (xa, xb)
            tx = trace(x)
            for ya in range(p):
                for yb in range(p):
                    y = (ya, yb)
                    ny = norm(y)
                    if (tx + ny) % p == 0:
                        idx = ((xa * p + xb) * p + ya) * p + yb
                        unital_indices.add(idx)
                        
    # Generate all lines
    # Directions: (1, m) for m in F_49, and (0, 1).
    lines = []
    line_to_unital_intersection = [] # List of sets of indices in U
    
    seen_lines = set()
    
    # For (1, m): y = m*x + c
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_points = []
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = mul(m, x)
                            y = add(mx, c)
                            idx = ((xa * p + xb) * p + y[0]) * p + y[1]
                            line_points.append(idx)
                    
                    line_tuple = tuple(sorted(line_points))
                    if line_tuple not in seen_lines:
                        seen_lines.add(line_tuple)
                        line_set = set(line_points)
                        lines.append(line_set)
                        # Intersection with U
                        inter = line_set & unital_indices
                        line_to_unital_intersection.append(inter)
    
    # For (0, 1): x = c
    for ca in range(p):
        for cb in range(p):
            c = (ca, cb)
            line_points = []
            for ya in range(p):
                for yb in range(p):
                    idx = ((ca * p + cb) * p + ya) * p + yb
                    line_points.append(idx)
                    
            line_tuple = tuple(sorted(line_points))
            if line_tuple not in seen_lines:
                seen_lines.add(line_tuple)
                line_set = set(line_points)
                lines.append(line_set)
                inter = line_set & unital_indices
                line_to_unital_intersection.append(inter)

    # Universe to cover: Points NOT in U
    universe = [i for i in range(num_points) if i not in unital_indices]
    remaining = set(universe)
    
    R = set() # Points in U that are removed (so not in S)
    
    # Precompute for each line:
    # - The set of points in U on the line (for cost calculation)
    # - The list of points in the universe on the line (for benefit calculation)
    
    line_universe_points = []
    for i, L in enumerate(lines):
        pts_in_universe = [p for p in L if p in unital_indices is False]
        # Optimization: store as list for fast intersection with 'remaining'
        line_universe_points.append(pts_in_universe)
        
    # Greedy Loop
    # We want to minimize |R|.
    # At each step, pick a line L that covers points in 'remaining'.
    # Cost of L = |(L \cap U) \ R|.
    # Benefit of L = |L \cap remaining|.
    
    # To speed up, we can maintain counts or just recompute.
    # Number of lines ~ 2450. Remaining starts ~ 2058.
    # Steps ~ 2058 / 49 ~ 42.
    # 42 * 2450 * 49 is small.
    
    while remaining:
        if time.time() > deadline:
            break
            
        best_line_idx = -1
        best_ratio = -1
        best_benefit = 0
        
        for i, L in enumerate(lines):
            # Calculate benefit
            # Points in L that are in remaining
            # L is a set. remaining is a set.
            # len(L & remaining)
            # But L & remaining is expensive if done naively?
            # We have line_universe_points[i] which is a list.
            # We can check how many are in remaining.
            
            # Optimization: If line has 0 points in remaining, skip.
            # We can maintain a counter?
            # For now, simple loop.
            count_benefit = 0
            for p in line_universe_points[i]:
                if p in remaining:
                    count_benefit += 1
            
            if count_benefit == 0:
                continue
                
            # Calculate cost
            # Points in L \cap U that are NOT in R
            inter_U = line_to_unital_intersection[i]
            count_cost = 0
            for u in inter_U:
                if u not in R:
                    count_cost += 1
            
            # We want to maximize benefit / (cost + 1) or similar?
            # If cost is 0, it's free.
            # If cost > 0, we pay.
            # Heuristic: Maximize benefit, tie-break by min cost.
            # Or Maximize benefit / (1 + cost).
            
            # Let's try: Maximize benefit. Among those, Minimize cost.
            # This might be too aggressive on benefit.
            # Better: Score = benefit - lambda * cost?
            # Given small sizes, let's try Max Benefit, then Min Cost.
            
            if count_benefit > best_benefit:
                best_benefit = count_benefit
                best_line_idx = i
                best_ratio = -count_cost # Store negative cost to maximize (min cost)
            elif count_benefit == best_benefit:
                # Tie: pick lower cost
                current_cost = -best_ratio
                if count_cost < current_cost:
                    best_line_idx = i
                    best_ratio = -count_cost

        if best_line_idx == -1:
            # No line covers remaining? Should not happen.
            # Force pick any line? Or break?
            # If remaining is not empty, we must cover it.
            # This implies our assumption that U-based S works might fail or we ran out of lines?
            # All lines are available. If a point is remaining, it is on some line.
            # That line has some cost.
            # Why would best_line_idx be -1? Only if count_benefit was 0 for all.
            # Which means remaining points are not in any line_universe_points?
            # This is impossible.
            break
            
        # Commit
        L = lines[best_line_idx]
        inter_U = line_to_unital_intersection[best_line_idx]
        
        # Update R
        R.update(inter_U)
        
        # Update remaining
        remaining.difference_update(L)

    # Construct S = U \ R
    S = unital_indices - R
    
    # Construct N = F^2 \ S
    N = [i for i in range(num_points) if i not in S]
    
    # Convert indices to points
    points = []
    for idx in N:
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        points.append([[xa, xb], [ya, yb]])
        
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
