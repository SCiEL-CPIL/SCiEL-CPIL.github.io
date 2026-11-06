import time
import random
import math

def construct():
    n = 128
    deadline = time.time() + 80.0  # Safety margin before 90s kill
    
    # Precompute all nonconstant 4-term APs (0-indexed positions)
    aps = []
    pos_to_aps = [[] for _ in range(n)]
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            ap = (i, i + d, i + 2 * d, i + 3 * d)
            idx = len(aps)
            aps.append(ap)
            for pos in ap:
                pos_to_aps[pos].append(idx)
    
    def is_ap_valid(colors, ap):
        c0 = colors[ap[0]]
        c1 = colors[ap[1]]
        c2 = colors[ap[2]]
        c3 = colors[ap[3]]
        
        # Count distinct colors logic
        # Valid if distinct >= 3
        # Invalid if distinct < 3
        # distinct < 3 means:
        # 1. All 4 same
        # 2. 3 same, 1 diff
        # 3. 2 same, 2 same (different pairs)
        
        # Check all same
        if c0 == c1 == c2 == c3:
            return False
        
        # Check 3 same
        if c0 == c1 and c0 == c2: return False
        if c0 == c1 and c1 == c3: return False
        if c0 == c2 and c2 == c3: return False
        if c1 == c2 and c2 == c3: return False
        
        # Check 2-2 patterns
        # Pairs (0,1) and (2,3)
        if c0 == c1 and c2 == c3 and c0 != c2:
            return False
        # Pairs (0,2) and (1,3)
        if c0 == c2 and c1 == c3 and c0 != c1:
            return False
        # Pairs (0,3) and (1,2)
        if c0 == c3 and c1 == c2 and c0 != c1:
            return False
            
        return True

    def count_violations(colors):
        v = 0
        for ap in aps:
            if not is_ap_valid(colors, ap):
                v += 1
        return v

    def solve_with_k(K, time_limit, initial_colors=None):
        if initial_colors:
            colors = initial_colors[:]
        else:
            colors = [random.randint(0, K - 1) for _ in range(n)]
            
        curr_violations = count_violations(colors)
        best_colors = colors[:]
        best_violations = curr_violations
        
        if best_violations == 0:
            return best_colors

        temp = 5.0
        cooling = 0.995
        min_temp = 0.01
        
        while time.time() < time_limit and temp > min_temp:
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            new_color = random.randint(0, K - 1)
            if new_color == old_color:
                continue
            
            # Calculate delta
            affected = pos_to_aps[pos]
            
            curr_invalid = 0
            for ap_idx in affected:
                ap = aps[ap_idx]
                if not is_ap_valid(colors, ap):
                    curr_invalid += 1
            
            colors[pos] = new_color
            
            new_invalid = 0
            for ap_idx in affected:
                ap = aps[ap_idx]
                if not is_ap_valid(colors, ap):
                    new_invalid += 1
            
            delta_v = new_invalid - curr_invalid
            
            if delta_v < 0:
                accept = True
            elif delta_v == 0:
                accept = True
            else:
                prob = math.exp(-delta_v / temp)
                accept = random.random() < prob
            
            if accept:
                curr_violations += delta_v
                if curr_violations < best_violations:
                    best_violations = curr_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors
            else:
                colors[pos] = old_color
                
            temp *= cooling
        
        return best_colors

    t_start = time.time()
    
    # Try K=3 first with a generous time
    limit_3 = t_start + 20.0
    res_3 = solve_with_k(3, limit_3)
    if count_violations(res_3) == 0:
        return res_3

    # Try K=4
    limit_4 = t_start + 40.0
    res_4 = solve_with_k(4, limit_4)
    if count_violations(res_4) == 0:
        # Try to compress to 3
        # Map one color to another
        # Try all pairs of colors to merge
        for c1 in range(4):
            for c2 in range(4):
                if c1 == c2: continue
                # Create a 3-coloring by mapping c2 to c1
                mapped_colors = [c if c != c2 else c1 for c in res_4]
                # Run SA with K=3 starting from this
                limit_comp = min(deadline - 5.0, t_start + 55.0)
                if time.time() < limit_comp:
                    res_comp = solve_with_k(3, limit_comp, initial_colors=mapped_colors)
                    if count_violations(res_comp) == 0:
                        return res_comp
        return res_4

    # Try K=5
    limit_5 = t_start + 55.0
    res_5 = solve_with_k(5, limit_5)
    if count_violations(res_5) == 0:
        return res_5

    # Try K=6
    limit_6 = t_start + 65.0
    res_6 = solve_with_k(6, limit_6)
    if count_violations(res_6) == 0:
        return res_6

    # Fallback
    remaining = deadline - time.time()
    if remaining > 2.0:
        res_7 = solve_with_k(7, deadline - 1.0)
        if count_violations(res_7) == 0:
            return res_7

    return list(range(n))
