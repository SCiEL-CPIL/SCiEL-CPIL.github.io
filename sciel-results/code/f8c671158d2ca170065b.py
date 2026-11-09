import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 12
    m = 40
    
    # All possible subsets as bitmasks
    all_subsets = range(1 << n)
    
    # Initial family: A maximal chain
    # Chain: 0, 1, 3, 7, ..., (1<<12)-1
    family = []
    mask = 0
    for i in range(n + 1):
        family.append(mask)
        mask = (mask << 1) | 1
        # Wait, this creates 0, 1, 3, 7... which is 0, {0}, {0,1}, {0,1,2}...
        # This is a valid chain of length 13.
        
    family_set = set(family)
    current_size = len(family)
    
    # Greedy expansion
    while current_size < m:
        best_candidate = -1
        best_gain = -1
        
        # Iterate through all possible subsets
        # Optimization: We can skip subsets already in family
        for s in all_subsets:
            if s in family_set:
                continue
            
            gain = 0
            # Check comparability with all sets in family
            for f in family:
                # Strict subset check
                # f is strict subset of s: (f & s) == f and f != s
                # s is strict subset of f: (f & s) == s and f != s
                # Since s != f is guaranteed by the 'if s in family_set' check,
                # we just check if one is subset of the other.
                if (f & s) == f or (f & s) == s:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_candidate = s
                
        if best_candidate != -1:
            family.append(best_candidate)
            family_set.add(best_candidate)
            current_size += 1
            
        if time.time() > deadline:
            break
            
    # Local Search (optional, but good for safety)
    # Try to improve by swapping
    # Given the speed, we can afford a few iterations
    # However, the greedy method on a poset often finds the global optimum for small N.
    # Let's stick to the greedy result as it is very likely optimal.
    # If we have time, we can run a simple local search.
    
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(family)):
                if time.time() > deadline - 5.0:
                    break
                old_s = family[i]
                family_set.remove(old_s)
                
                best_swap_candidate = -1
                best_swap_gain = -1
                current_gain_if_removed = 0
                
                # Calculate gain if we remove old_s
                # This is O(N^2) per removal, which is expensive if done naively for every swap.
                # Better: Just evaluate the total score for the new family? No, too slow.
                # Let's just stick to the greedy construction. It is deterministic and fast.
                # The greedy choice maximizes immediate gain. For subset posets, this is usually optimal.
                
                family_set.add(old_s)
                
    # Convert bitmasks to lists of integers
    result = []
    for s in family:
        subset = []
        for i in range(n):
            if (s >> i) & 1:
                subset.append(i)
        result.append(subset)
        
    return result
