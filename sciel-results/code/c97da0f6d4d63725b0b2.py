"""Combined program: fast greedy initialization + efficient local search with deadline guard."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s limit
    
    n = 256
    
    # Fast AP-free check: for a set S, x cannot be added if there exist a, b in S
    # such that {a, b, x} forms a 3-term AP.
    # This happens if:
    #   2*x == a + b  (x is the midpoint)
    #   2*a == b + x  (a is the midpoint, so x = 2*a - b)
    #   2*b == a + x  (b is the midpoint, so x = 2*b - a)
    # Equivalently, for each a in S, check if 2*a - x is in S and != a
    # and for each b in S, check if 2*b - x is in S and != b
    # and check if (2*x - a) is in S for some a... 
    # Actually simpler: for each a in S, the forbidden values are 2*a - b for all b in S.
    # But even simpler: x is forbidden if there exists a in S such that 2*x - a is in S and 2*x - a != a.
    # Wait, that's just checking if x is the midpoint.
    # Let me think again. A 3-AP is {u, v, w} with u+w=2v.
    # If we add x to S, we need to ensure no new 3-AP is formed involving x.
    # Cases:
    # 1. x is the middle: u + w = 2x, where u, w in S. So for each u in S, if 2x-u in S and 2x-u != u, bad.
    # 2. x is the end: x + w = 2v, where v, w in S. So for each v in S, if 2v-x in S and 2v-x != v and 2v-x != x, bad.
    #    Equivalently, for each v in S, if x = 2v - w for some w in S, w != v.
    
    def can_add(S, x):
        """Check if adding x to S maintains AP-free property."""
        # Case 1: x is the middle term
        for a in S:
            b = 2 * x - a
            if b != a and b in S:
                return False
        # Case 2: x is an end term (x is first or last)
        # If x is first: x, b, c with x+c=2b => c = 2b-x, need b,c in S, b!=x, c!=x
        # If x is last: a, b, x with a+x=2b => a = 2b-x, need a,b in S, a!=x, b!=x
        # Both covered by: for each b in S, if 2b-x in S and 2b-x != b and 2b-x != x
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
    
    # Local search: try to improve by swapping or adding/removing
    # Strategy: 
    # 1. Try adding excluded elements (prefer small ones)
    # 2. Try swapping: remove a large element, add a small one
    # 3. Try pair swaps for more aggressive improvement
    
    while time.time() < deadline:
        improved = False
        S_list = sorted(S)
        excluded = set(range(1, n + 1)) - S
        
        # Phase 1: Try adding each excluded element (prioritize small x)
        for x in sorted(excluded, key=lambda v: -1.0 / v):
            if time.time() > deadline:
                break
            if can_add(S, x):
                new_S = S | {x}
                new_val = best_val + 1.0 / x
                if new_val > best_val:
                    S = new_S
                    best_S = set(S)
                    best_val = new_val
                    improved = True
                    # Recalculate excluded
                    excluded = set(range(1, n + 1)) - S
        
        if time.time() > deadline:
            break
        
        # Phase 2: Try single swaps (remove u, add v where v < u)
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
                    if v >= u:
                        break  # We want v < u for improvement
                    if can_add(S_temp, v):
                        new_val = temp_val + 1.0 / v
                        if new_val > best_val + 1e-15:
                            S = S_temp | {v}
                            best_S = set(S)
                            best_val = new_val
                            improved = True
                            break  # Restart swap search with new S
                
                if improved:
                    break
        
        if not improved:
            # Phase 3: Try pair swaps (remove u1, u2, add v1, v2)
            # Only do this if we have time
            if time.time() > deadline - 5.0:
                break
            
            S_list = sorted(S, reverse=True)
            excluded = sorted(set(range(1, n + 1)) - S, key=lambda v: -1.0 / v)
            
            found_improvement = False
            # Try removing the two largest elements
            if len(S_list) >= 2:
                u1, u2 = S_list[0], S_list[1]
                S_temp = S - {u1} - {u2}
                temp_val = best_val - 1.0 / u1 - 1.0 / u2
                
                # Try adding pairs of small excluded elements
                for i, v1 in enumerate(excluded[:20]):  # Only try top 20 candidates
                    if v1 == u1 or v1 == u2:
                        continue
                    for v2 in excluded[i+1:20]:
                        if v2 == u1 or v2 == u2:
                            continue
                        # Check if both can be added
                        if can_add(S_temp, v1):
                            S_temp2 = S_temp | {v1}
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
                continue  # Loop back to try more improvements
            else:
                break  # No improvement possible
    
    return list(best_S)
