def construct():
    from math import gcd
    import time
    
    N = 512
    n = 511
    start_time = time.time()
    time_limit = 115
    
    # Precompute conflicts
    # conflicts[a] is a set of b such that gcd(a,b) > 1
    conflicts = [set() for _ in range(n+1)]
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            if gcd(a, b) > 1:
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    # Weights
    weights = {a: 1.0 / (512 - a) for a in range(1, 512)}
    
    # Order by weight descending
    order = list(range(1, 512))
    order.sort(key=lambda x: weights[x], reverse=True)
    
    # Greedy initial solution
    greedy_selected = set()
    greedy_available = set(range(1, 512))
    for v in order:
        if v in greedy_available:
            greedy_selected.add(v)
            greedy_available -= conflicts[v]
    
    best_weight = sum(weights[v] for v in greedy_selected)
    best_set = set(greedy_selected)
    
    import sys
    sys.setrecursionlimit(10000)
    
    def clique_cover_bound(available):
        """
        Computes an upper bound on the MWIS of the subgraph induced by 'available'.
        Uses a greedy clique cover:
        1. Sort available by weight desc.
        2. Pick highest weight v.
        3. Form a clique C starting with v, adding any other available u that is adjacent to all in C.
        4. Add weight of v to bound.
        5. Remove C from available.
        """
        if not available:
            return 0.0
        
        # Convert to list sorted by weight descending for efficiency in selection
        # We can just iterate through the global order and filter for available
        bound = 0.0
        remaining = set(available)
        
        while remaining:
            # Find vertex in remaining with max weight
            # Since 'order' is sorted, we can find the first one in 'remaining'
            v = None
            for u in order:
                if u in remaining:
                    v = u
                    break
            
            if v is None:
                break
                
            bound += weights[v]
            
            # Form clique C
            C = {v}
            # Try to add other vertices to C
            # A vertex u can be added to C if it is adjacent to all vertices currently in C
            # We iterate through remaining \ {v}
            to_remove = set()
            for u in remaining:
                if u == v:
                    continue
                # Check if u is adjacent to all in C
                is_adj_to_all = True
                for c in C:
                    if u not in conflicts[c]:
                        is_adj_to_all = False
                        break
                if is_adj_to_all:
                    C.add(u)
                    to_remove.add(u)
            
            # Remove C from remaining
            remaining -= C
            
        return bound

    def search(idx, selected, available, current_weight):
        nonlocal best_weight, best_set
        
        if time.time() - start_time > time_limit:
            return
        
        if not available:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(selected)
            return
        
        # Upper bound using clique cover
        ub = current_weight + clique_cover_bound(available)
        if ub <= best_weight:
            return
        
        if idx >= len(order):
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(selected)
            return
        
        v = order[idx]
        
        if v not in available:
            search(idx + 1, selected, available, current_weight)
            return
        
        # Branch 1: Include v
        new_selected = selected | {v}
        new_available = available - {v} - conflicts[v]
        search(idx + 1, new_selected, new_available, current_weight + weights[v])
        
        if time.time() - start_time > time_limit:
            return
        
        # Branch 2: Exclude v
        new_available2 = available - {v}
        search(idx + 1, selected, new_available2, current_weight)
    
    initial_available = set(range(1, 512))
    search(0, set(), initial_available, 0.0)
    
    return list(best_set)
