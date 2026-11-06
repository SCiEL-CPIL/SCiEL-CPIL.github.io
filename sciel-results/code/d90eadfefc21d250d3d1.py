"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import sys
    sys.setrecursionlimit(100000)

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_sum_free(S):
        """
        Find a maximum sum-free subset of S using reduction to core elements and B&B.
        """
        n = len(S)
        if n <= 1:
            return S[:]
        
        s_set = set(S)
        
        # 1. Identify all sum triples (a, b, c) with a+b=c, a,b,c in S
        # We need to know which elements are part of ANY such triple.
        core_elements = set()
        
        # Iterate through all pairs to find sums
        # Optimization: Only check if sum is in s_set
        for a in S:
            for b in S:
                c = a + b
                if c in s_set:
                    # Triple (a, b, c) exists
                    core_elements.add(a)
                    core_elements.add(b)
                    core_elements.add(c)
        
        # 2. Elements not in core can always be included
        iso_elements = [x for x in S if x not in core_elements]
        
        # 3. Solve for the core elements
        core_list = list(core_elements)
        m = len(core_list)
        
        if m == 0:
            return iso_elements
        
        # If core is small, use exhaustive search / B&B
        # If core is large, we might need a heuristic, but for random sets it should be small.
        # Let's assume m <= 30 for B&B to be safe with pruning, else use greedy+local search.
        
        best_subset = []
        
        def is_sum_free_subset(subset):
            sub_set = set(subset)
            for a in sub_set:
                for b in sub_set:
                    if a + b in sub_set:
                        return False
            return True
        
        if m <= 25:
            # Branch and Bound
            # Sort core elements to improve pruning (e.g., by value or degree)
            # We can just use the list as is
            
            def search(idx, current_subset, current_size):
                nonlocal best_subset
                
                # Pruning: Even if we add all remaining, we can't beat best
                if current_size + (m - idx) <= len(best_subset):
                    return
                
                if idx == m:
                    if current_size > len(best_subset):
                        best_subset = current_subset[:]
                    return
                
                x = core_list[idx]
                
                # Option 1: Exclude x
                search(idx + 1, current_subset, current_size)
                
                # Option 2: Include x
                # Check if adding x keeps it sum-free
                # We can check incrementally: 
                # Does x form a sum with any pair in current_subset?
                # Or is x the sum of two elements in current_subset?
                # Or does x + a = b for some a,b in current_subset?
                
                valid = True
                for a in current_subset:
                    # Check if x + a is in current_subset (then x+a = x+a, triple x, a, x+a)
                    # Wait, if x+a is in current_subset, then {x, a, x+a} is a triple.
                    if x + a in current_subset:
                        valid = False
                        break
                    # Check if a + x is in current_subset (same as above)
                    # Check if x is sum of two in current_subset: a + b = x
                    # We can check this by: for each b in current_subset, if x-b is in current_subset
                    # But we already looped a. Let's combine.
                    
                if valid:
                    # Also check if x is sum of two elements in current_subset
                    # i.e., exists a, b in current_subset such that a+b=x
                    # We can check this separately or during the loop above?
                    # In the loop above, we checked if x+a is in current_subset.
                    # We need to check if x-a is in current_subset? No, that's a+b=x => b=x-a.
                    # So if there exists a in current_subset such that x-a is in current_subset, then a+(x-a)=x.
                    # So we need to check:
                    for a in current_subset:
                        if x - a in current_subset:
                            valid = False
                            break
                    
                    if valid:
                        # Also check a+b=x where a,b are the same? a=2a? No, a+b=x.
                        # If a=b, then 2a=x. So if x/2 is in current_subset and x/2 is integer.
                        # The loop `if x-a in current_subset` covers a=b if x-a == a => x=2a.
                        # So if x is even and x/2 in current_subset, then x-x/2 = x/2, so it's detected.
                        
                        # One more case: a+b=c where c=x. Covered.
                        # What about a+b=c where a=x? x+b=c. Covered by `if x+b in current_subset`.
                        # What about a+b=c where b=x? a+x=c. Covered by `if x+a in current_subset`.
                        
                        current_subset.append(x)
                        search(idx + 1, current_subset, current_size + 1)
                        current_subset.pop()
            
            search(0, [], 0)
            
        else:
            # Fallback for large core (unlikely for random sets)
            # Use greedy + local search
            # Greedy: pick elements that don't conflict
            T = set()
            for x in core_list:
                # Check if x can be added to T
                ok = True
                for a in T:
                    if x + a in T or x - a in T or a - x in T:
                         # Note: a-x in T means a = x + (a-x), so x + (a-x) = a. Triple (x, a-x, a).
                         # x+a in T means x+a = c. Triple (x, a, c).
                         # x-a in T means x = a + (x-a). Triple (a, x-a, x).
                         ok = False
                         break
                if ok:
                    T.add(x)
            
            # Local search
            improved = True
            while improved:
                improved = False
                for x in core_list:
                    if x in T:
                        T.remove(x)
                        # Try to add others
                        added = False
                        for y in core_list:
                            if y not in T:
                                ok = True
                                for a in T:
                                    if y + a in T or y - a in T or a - y in T:
                                        ok = False
                                        break
                                if ok:
                                    T.add(y)
                                    added = True
                        if added:
                            improved = True
                            break
            
            best_subset = list(T)
        
        result = iso_elements + best_subset
        return result
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        subset = solve_sum_free(S)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
