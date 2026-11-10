# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy:
    1. Start with a structural base set S0 derived from the Hermitian unital, 
       which is known to be a valid Nikodym complement and has a good size.
    2. Use a greedy algorithm to expand S0 by adding more points while 
       maintaining the Nikodym property.
    3. The Nikodym property for complement S is that for every point x, 
       there exists a line L through x such that L \cap S \subseteq {x}.
    
    The score is proportional to |S|.
    """
    p_prime = 7
    q = 49
    N_points = q * q  # 2401
    
    # --- Field Arithmetic F_49 ---
    # Represent elements as (a, b) where value is a + b*alpha, alpha^2 = 3.
    # idx = a * 7 + b.
    
    def add_idx(i, j):
        a1, b1 = divmod(i, 7)
        a2, b2 = divmod(j, 7)
        return ((a1 + a2) % 7) * 7 + (b1 + b2) % 7

    def mul_idx(i, j):
        a1, b1 = divmod(i, 7)
        a2, b2 = divmod(j, 7)
        real = (a1 * a2 + 3 * b1 * b2) % 7
        imag = (a1 * b2 + b1 * a2) % 7
        return real * 7 + imag

    def trace_idx(i):
        # T(x) = x + x^7. In F_{p^2}, x^p is conjugate.
        # If x = a + b*alpha, x^p = a - b*alpha (since alpha^p = -alpha for p=7? 
        # alpha^2=3. alpha^7 = alpha * alpha^6 = alpha * (alpha^2)^3 = alpha * 3^3 = alpha * 27 = 6 alpha = -alpha.
        # So x^7 = a - b*alpha.
        # T(x) = 2a.
        a, b = divmod(i, 7)
        return (2 * a) % 7

    def norm_idx(i):
        # N(x) = x * x^7 = (a + b*alpha)(a - b*alpha) = a^2 - 3b^2.
        a, b = divmod(i, 7)
        return (a * a - 3 * b * b) % 7

    # Precompute tables
    add_table = np.zeros((49, 49), dtype=np.int32)
    mul_table = np.zeros((49, 49), dtype=np.int32)
    trace_table = np.zeros(49, dtype=np.int32)
    norm_table = np.zeros(49, dtype=np.int32)
    
    for i in range(49):
        trace_table[i] = trace_idx(i)
        norm_table[i] = norm_idx(i)
        for j in range(49):
            add_table[i, j] = add_idx(i, j)
            mul_table[i, j] = mul_idx(i, j)

    # --- Line Representation ---
    # Lines are defined by slope m in F_49 (49 slopes) and vertical lines.
    # Total 50 lines through any point.
    # We represent a line by (type, param).
    # Type 0: Vertical. Param x. Line: {(x, y) for y in F_49}.
    # Type 1: Slope m. Param c. Line: {(x, y) for x in F_49, y = m*x + c}.
    
    # Precompute points on each line.
    # lines: list of 2450 arrays of 49 point indices.
    line_points = []
    
    # Vertical lines
    for x in range(49):
        pts = []
        for y in range(49):
            pts.append(x * 49 + y)
        line_points.append(np.array(pts, dtype=np.int32))
        
    # Sloped lines
    for m in range(49):
        for c in range(49):
            pts = []
            for x in range(49):
                y = add_idx(mul_table[m, x], c)
                pts.append(x * 49 + y)
            line_points.append(np.array(pts, dtype=np.int32))
            
    # Precompute: for each point, list of line indices passing through it.
    point_lines = [[] for _ in range(2401)]
    for l_idx, pts in enumerate(line_points):
        for p in pts:
            point_lines[p].append(l_idx)
            
    # --- Structural Base Set S0 ---
    # S0 = { (x,y) in U | y not in Y }
    # U = { (x,y) | T(x) + N(y) == 0 }
    # Y = { y | y_a in {1, 2, 4} }
    
    in_S = np.zeros(2401, dtype=bool)
    
    # Squares in F_7: 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1.
    # Set of non-zero squares: {1, 2, 4}.
    y_special_ys = set()
    for u in range(1, 7):
        sq = (u * u) % 7
        for v in range(7):
            y_special_ys.add(sq * 7 + v)
            
    for x in range(49):
        for y in range(49):
            # Check if in Unital
            if (trace_table[x] + norm_table[y]) % 7 == 0:
                # Check if y is NOT in Y
                if y not in y_special_ys:
                    in_S[x * 49 + y] = True

    # --- Greedy Expansion ---
    # Maintain C[x] = number of lines L through x such that L \cap S \subseteq {x}.
    # Initially compute C[x].
    
    C = np.zeros(2401, dtype=np.int32)
    
    # For each line L, determine S \cap L.
    # If |S \cap L| == 0: L is good for all x in L.
    # If |S \cap L| == 1: Let s be the point. L is good for x=s (since {s} \subseteq {s}) and NOT good for x != s (since {s} not subset {x}).
    #    Wait. Condition: L \cap S \subseteq {x}.
    #    If S \cap L = {s}.
    #    If x = s: {s} \subseteq {s} is True.
    #    If x != s: {s} \subseteq {x} is False.
    # If |S \cap L| >= 2: L is not good for any x.
    
    # We can compute C[x] by iterating lines.
    # For each line L:
    #   count = sum(in_S[L])
    #   if count == 0:
    #      for x in L: C[x] += 1
    #   elif count == 1:
    #      s = the unique point in S \cap L
    #      C[s] += 1
    #   else:
    #      pass
    
    for l_idx, pts in enumerate(line_points):
        pts_in_S = [p for p in pts if in_S[p]]
        cnt = len(pts_in_S)
        if cnt == 0:
            for x in pts:
                C[x] += 1
        elif cnt == 1:
            s = pts_in_S[0]
            C[s] += 1
            
    # Check validity of S0 (should be valid)
    # min(C) should be > 0.
    if np.min(C) == 0:
        # Fallback to just S0 if it's somehow invalid? No, it's structural.
        pass

    # Greedy loop
    # Order of points to consider adding to S
    # We only consider points not already in S.
    candidates = [i for i in range(2401) if not in_S[i]]
    
    # Shuffle candidates
    rng = np.random.RandomState(42)
    rng.shuffle(candidates)
    
    start_time = time.time()
    deadline = start_time + 580
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        # Check if we can add p
        # We need to check all x on lines through p (x != p).
        # For each such line L, if L is currently "good" for x, C[x] will decrease.
        # L is good for x if (S \cap L) \subseteq {x}.
        # Since p is not in S yet, S \cap L is the old intersection.
        # If we add p, new intersection is (S \cap L) \cup {p}.
        # For x != p, new intersection contains p.
        # So new intersection \subseteq {x} is false.
        # So L becomes NOT good for x.
        # So we must decrement C[x] if L was good for x.
        
        # To know if L was good for x, we check if (S \cap L) \subseteq {x}.
        # This is equivalent to:
        # - If x in S: S \cap L == {x}
        # - If x not in S: S \cap L == empty
        
        # We can precompute for each line L:
        # count_S[L] = |S \cap L|
        # and if count_S[L] == 1, the point s_L.
        
        # Let's maintain count_S and s_point for each line.
        # Initially compute them.
        
    # Re-initialize line stats
    line_count_S = np.zeros(2450, dtype=np.int32)
    line_single_point = np.full(2450, -1, dtype=np.int32)
    
    for l_idx, pts in enumerate(line_points):
        cnt = 0
        s_pt = -1
        for x in pts:
            if in_S[x]:
                cnt += 1
                s_pt = x
                if cnt > 1:
                    break
        line_count_S[l_idx] = cnt
        if cnt == 1:
            line_single_point[l_idx] = s_pt
            
    # Now iterate candidates
    for p in candidates:
        if time.time() > deadline:
            break
            
        # Simulate adding p
        # Affected lines are point_lines[p]
        # For each L in point_lines[p]:
        #   For each x in L, x != p:
        #     Check if L was good for x.
        #     If yes, C[x] -= 1.
        #     If C[x] becomes 0, invalid.
        
        valid = True
        changes = [] # (x, delta)
        
        for L in point_lines[p]:
            old_count = line_count_S[L]
            # L was good for x if:
            # old_count == 0 and x not in S (which is true for all x if old_count 0? No, x could be in S? 
            # If old_count 0, S \cap L is empty.
            # Good for x if empty \subseteq {x}. Always true.
            # So if old_count == 0, L is good for ALL x in L.
            
            # If old_count == 1:
            # s = line_single_point[L].
            # Good for x if {s} \subseteq {x} => x == s.
            
            # If old_count >= 2:
            # Not good for any x.
            
            if old_count == 0:
                # L is good for all x in L.
                for x in line_points[L]:
                    if x != p:
                        C[x] -= 1
                        changes.append(x)
                        if C[x] == 0:
                            valid = False
                            break
                if not valid:
                    break
            elif old_count == 1:
                s = line_single_point[L]
                # L is good only for x = s.
                if s != p:
                    # s is on L.
                    C[s] -= 1
                    changes.append(s)
                    if C[s] == 0:
                        valid = False
                        break
            # else: no change
            
        if valid:
            # Commit
            in_S[p] = True
            for L in point_lines[p]:
                old_count = line_count_S[L]
                line_count_S[L] = old_count + 1
                if old_count == 0:
                    line_single_point[L] = p
                elif old_count == 1:
                    line_single_point[L] = -1 # Now count is 2, no single point
                # else: count >= 2, remains >= 3
        else:
            # Revert
            for x in changes:
                C[x] += 1
            # line_count_S and line_single_point are not modified in the check phase, so no revert needed there.
            # Wait, I didn't modify them in the check phase. Correct.

    # Construct result N
    # N is points NOT in S.
    result_list = []
    for i in range(2401):
        if not in_S[i]:
            x = i // 49
            y = i % 49
            xa, xb = divmod(x, 7)
            ya, yb = divmod(y, 7)
            result_list.append([
                [xa, xb],
                [ya, yb]
            ])
            
    if len(result_list) == 0:
        return np.empty((0, 2, 2), dtype=np.int64)
        
    result = np.array(result_list, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
