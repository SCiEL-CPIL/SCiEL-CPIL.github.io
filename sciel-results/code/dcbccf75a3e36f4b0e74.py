import time
from fractions import Fraction

def construct():
    start_time = time.time()
    limit = 85  # seconds, safety margin before 90s kill
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute for speed: 
    # For each x, store potential conflicts? No, dynamic set is easier.
    # But we can optimize can_add using a set for O(1) lookups.
    
    def can_add(x, S):
        # x is not in S
        # Check 1: x is largest (a, b, x). 2b = a+x.
        # For each a in S where a < x:
        #   b = (a+x)/2. If b is integer and b in S, conflict.
        # Check 2: x is smallest (x, b, c). 2b = x+c.
        # For each c in S where c > x:
        #   b = (x+c)/2. If b is integer and b in S, conflict.
        #   Alternatively: for each b in S where b > x: c = 2b-x. If c in S, conflict.
        # Check 3: x is middle (a, x, c). 2x = a+c.
        # For each a in S where a < x:
        #   c = 2x - a. If c in S, conflict.
        
        # Optimization: Use set S for O(1) lookup.
        # We can combine some checks.
        
        # Case 1: x is the max element of a triplet (a, b, x)
        # a + x = 2b => b = (a+x)/2
        # Iterate a in S, a < x
        for a in S:
            if a >= x:
                continue
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in S:
                    return False
        
        # Case 2: x is the min element of a triplet (x, b, c)
        # x + c = 2b => c = 2b - x
        # Iterate b in S, b > x
        for b in S:
            if b <= x:
                continue
            c = 2 * b - x
            if c > 0 and c in S:
                return False
        
        # Case 3: x is the middle element of a triplet (a, x, c)
        # a + c = 2x => c = 2x - a
        # Iterate a in S, a < x
        for a in S:
            if a >= x:
                continue
            c = 2 * x - a
            if c > 0 and c in S:
                return False
        
        return True

    def get_fraction_sum(S):
        return sum(Fraction(1, x) for x in S)

    def get_float_sum(S):
        return sum(1.0 / x for x in S)

    # Strategy 1: Standard Greedy (add smallest available)
    def greedy_standard():
        S = set()
        for x in range(1, N + 1):
            if can_add(x, S):
                S.add(x)
        return S

    # Strategy 2: Greedy with "Lookahead" or different ordering?
    # The objective is sum(1/x). Smaller x are more valuable.
    # Standard greedy picks the smallest possible x. This is optimal for "maximize count" 
    # but for "maximize weight", is it optimal?
    # Usually, yes, because 1/x is convex and decreasing. Picking a small x early 
    # might block multiple larger x's, but the loss in value from skipping a small x 
    # is huge compared to the gain from keeping several large x's.
    # However, sometimes skipping a "bad" small x (that blocks many others) 
    # might allow adding several medium x's.
    
    # Let's try a Greedy that considers the "cost" of blocking.
    # But first, let's run the standard greedy and then do local search.
    
    best_set = greedy_standard()
    best_score = get_float_sum(best_set)
    
    # Local Search: 
    # 1. Try removing an element x and see if we can add several new elements.
    # 2. Since 1/x is decreasing, we prefer to remove large x (low value) 
    #    if it allows adding small y (high value).
    
    # Efficient Local Search:
    # For each x in S (sorted by value, i.e., descending x):
    #   Temporarily remove x.
    #   Run a "Greedy Fill" on the remaining numbers (1..256) that are not in S\{x} 
    #   but can be added to S\{x}. 
    #   Actually, just try to add any number y not in S\{x} that is valid.
    #   Prioritize small y.
    
    def local_search(S_init):
        S = set(S_init)
        current_score = get_float_sum(S)
        improved = True
        # We iterate multiple passes until no improvement or time limit.
        while improved and (time.time() - start_time) < limit:
            improved = False
            # Sort S by reciprocal value ascending (i.e., x descending) 
            # because we want to remove low-value elements first.
            # If removing a low-value element allows adding high-value elements, it's a win.
            sorted_S = sorted(S, reverse=True)
            
            for x in sorted_S:
                if (time.time() - start_time) > limit:
                    return S, current_score
                
                S.remove(x)
                # Try to add new elements to S
                # We want to add elements y such that 1/y > 1/x (i.e., y < x) ideally,
                # or just any y that increases the total sum.
                # Since we removed x, we lost 1/x. We need to gain > 1/x from additions.
                
                # Let's try adding elements in increasing order of y (highest value first).
                # But we must ensure they don't conflict with each other or existing S.
                
                added_any = False
                # We can try a greedy fill for the "holes"
                # To be safe and fast, just try adding any valid y from 1 to 256
                # that is not in S.
                # We prioritize small y.
                
                potential_adds = []
                for y in range(1, N + 1):
                    if y in S:
                        continue
                    if can_add(y, S):
                        potential_adds.append(y)
                
                # Sort potential adds by value (1/y) descending -> y ascending
                potential_adds.sort()
                
                new_S = set(S)
                new_score = current_score - (1.0 / x) # approximate, will recalculate
                
                # Greedily add from potential_adds
                for y in potential_adds:
                    if (time.time() - start_time) > limit:
                        break
                    if can_add(y, new_S):
                        new_S.add(y)
                
                new_score = get_float_sum(new_S)
                
                if new_score > current_score + 1e-12:
                    S = new_S
                    current_score = new_score
                    improved = True
                    # Restart the loop with the new S? 
                    # Or just continue? 
                    # Restarting is safer to catch cascading improvements.
                    break 
                else:
                    # Restore S (it was already modified by S.remove(x), but we didn't assign S=new_S)
                    # Wait, S is still S\{x} here. We need to put x back if we don't accept.
                    S.add(x)
            
        return S, current_score

    # Run local search
    best_set, best_score = local_search(best_set)
    
    # Try other starting points?
    # Base-3 set (Cantor set) is 3-free.
    def base3_set():
        S = set()
        for x in range(1, N + 1):
            t = x
            valid = True
            while t > 0:
                if t % 3 == 2:
                    valid = False
                    break
                t //= 3
            if valid:
                S.add(x)
        return S

    S_base3 = base3_set()
    # Greedy fill on base3?
    # The base3 set is maximal 3-free? No, not necessarily. 
    # Let's try to fill it.
    def greedy_fill(S_init):
        S = set(S_init)
        for x in range(1, N + 1):
            if x not in S and can_add(x, S):
                S.add(x)
        return S

    S_b3_filled = greedy_fill(S_base3)
    score_b3 = get_float_sum(S_b3_filled)
    if score_b3 > best_score:
        best_score = score_b3
        best_set = S_b3_filled
        # Run local search on this too
        best_set, best_score = local_search(best_set)

    # Try a "Randomized Greedy" with a few restarts if time permits
    import random
    for _ in range(5):
        if (time.time() - start_time) > limit - 5:
            break
        
        # Randomized order? 
        # Or random subset of forbidden?
        # Let's try perturbing the standard greedy.
        # Pick a random x in the greedy set, forbid it, re-greedy.
        if not best_set:
            break
        x_forbid = random.choice(list(best_set))
        
        S_new = set()
        for x in range(1, N + 1):
            if x == x_forbid:
                continue
            if can_add(x, S_new):
                S_new.add(x)
        
        score_new = get_float_sum(S_new)
        if score_new > best_score:
            best_score = score_new
            best_set = S_new
            # Quick local search
            best_set, best_score = local_search(best_set)

    return list(best_set)
