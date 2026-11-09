import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    n = 256
    
    def can_add(S, x):
        """O(|S|) check if adding x to S maintains 3-AP-free property."""
        # Case 1: x is the middle term: a + c = 2x, so c = 2x - a
        for a in S:
            c = 2 * x - a
            if c != a and c in S:
                return False
        # Case 2: x is an end term: x + c = 2b or a + x = 2b
        # Equivalently: for some b in S, 2b - x is in S and != b and != x
        for b in S:
            a = 2 * b - x
            if a != b and a != x and a in S:
                return False
        return True
    
    # Greedy initialization: add smallest elements first (maximizes reciprocal sum)
    S = set()
    for x in range(1, n + 1):
        if can_add(S, x):
            S.add(x)
    
    best_S = set(S)
    best_val = sum(1.0 / x for x in S)
    
    # Precompute all excluded elements sorted by reciprocal (descending)
    all_excluded = sorted(set(range(1, n + 1)) - S, key=lambda v: -1.0 / v)
    
    # Local search with aggressive swaps
    while time.time() < deadline:
        improved = False
        excluded = sorted(set(range(1, n + 1)) - S, key=lambda v: -1.0 / v)
        
        # Phase 1: Try adding each excluded element (prioritize small x)
        for x in excluded:
            if time.time() > deadline:
                break
            if can_add(S, x):
                new_val = best_val + 1.0 / x
                if new_val > best_val:
                    S = S | {x}
                    best_S = set(S)
                    best_val = new_val
                    improved = True
        
        if time.time() > deadline:
            break
        
        # Phase 2: Try single swaps (remove u, add v where 1/v > 1/u)
        if not improved:
            S_list = sorted(S, reverse=True)  # Try removing largest first
            excluded = sorted(set(range(1, n + 1)) - S, key=lambda v: -1.0 / v)
            
            for u in S_list:
                if time.time() > deadline:
                    break
                S_temp = S - {u}
                temp_val = best_val - 1.0 / u
                
                for v in excluded:
                    if v == u:
                        continue
                    if 1.0 / v <= 1.0 / u:
                        break  # No improvement possible
                    if can_add(S_temp, v):
                        new_val = temp_val + 1.0 / v
                        if new_val > best_val + 1e-15:
                            S = S_temp | {v}
                            best_S = set(S)
                            best_val = new_val
                            improved = True
                            break
                
                if improved:
                    break
        
        if time.time() > deadline:
            break
        
        # Phase 3: Try pair swaps for more aggressive improvement
        if not improved and time.time() < deadline - 3.0:
            S_list = sorted(S, reverse=True)
            excluded = sorted(set(range(1, n + 1)) - S, key=lambda v: -1.0 / v)
            
            found_improvement = False
            # Try removing the two largest elements and adding two smaller ones
            if len(S_list) >= 2 and len(excluded) >= 2:
                u1, u2 = S_list[0], S_list[1]
                S_temp = S - {u1} - {u2}
                temp_val = best_val - 1.0 / u1 - 1.0 / u2
                
                # Try adding pairs of small excluded elements
                for i, v1 in enumerate(excluded[:30]):
                    if time.time() > deadline:
                        break
                    if v1 == u1 or v1 == u2:
                        continue
                    if can_add(S_temp, v1):
                        S_temp2 = S_temp | {v1}
                        for v2 in excluded[i + 1:30]:
                            if v2 == u1 or v2 == u2:
                                continue
                            if 1.0 / v1 + 1.0 / v2 <= 1.0 / u1 + 1.0 / u2:
                                break
                            if can_add(S_temp2, v2):
                                new_val = temp_val + 1.0 / v1 + 1.0 / v2
                                if new_val > best_val + 1e-15:
                                    S = S_temp2 | {v2}
                                    best_S = set(S)
                                    best_val = new_val
                                    found_improvement = True
                                    break
                    if found_improvement:
                        break
            
            if found_improvement:
                continue
            else:
                break
    
    return list(best_S)
