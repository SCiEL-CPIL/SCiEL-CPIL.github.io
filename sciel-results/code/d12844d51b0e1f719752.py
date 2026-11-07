import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    # Precompute all potential edges
    all_edges = list(combinations(range(n), 2))
    
    # Initial values: 1..14
    # This is a known good starting point for collision potential? 
    # Actually, let's try to find values that allow for many collisions.
    # Small integers tend to have more multiplicative collisions.
    # Let's stick with 1..14 for now, but the clique search will pick the best 35 edges.
    values = list(range(1, n + 1))
    
    # Calculate sums and products for all potential edges
    edge_sums = [values[u] + values[v] for u, v in all_edges]
    edge_prods = [values[u] * values[v] for u, v in all_edges]
    
    # We want to select a subset of 35 edges such that:
    # max(len(set(selected_sums)), len(set(selected_prods))) is minimized.
    
    # Let's iterate K (the target max distinct count) from a lower bound upwards.
    # Lower bound: 
    # If we pick 35 edges, can we have fewer than 35 distinct sums?
    # Yes, if collisions exist.
    # With values 1..14, how many distinct sums are possible?
    # Min sum 1+2=3, Max 13+14=27. Range 25.
    # So distinct sums <= 25 is possible if we can cover all sums? No, we only pick 35 edges.
    # If we pick edges to cover sums 3..27, that's 25 distinct sums.
    # Distinct products: 1*2=2 ... 13*14=182.
    # Many collisions in products for small numbers.
    
    # Let's start K at 20 and go up.
    # If we find a clique of size 35 in the graph defined by K, we are done.
    
    # Conflict Graph Construction:
    # Vertex i represents edge all_edges[i].
    # Two vertices i, j are connected in the "compatibility" graph if:
    # Adding edge j to a set containing edge i does not force the distinct count > K?
    # This is not a simple pairwise property. The "distinct count" is a global property of the set.
    # However, we can use a heuristic:
    # If we assume we want to keep distinct sums <= K and distinct prods <= K.
    
    # A better approach for "Max Clique" framing:
    # We can't easily define pairwise compatibility for a global cardinality constraint.
    # But we can use a greedy/backtracking search that tracks the current sets of sums/prods.
    
    # Let's implement a backtracking search with pruning.
    # State: current_edges, current_sums_set, current_prods_set, index in all_edges.
    # Goal: find 35 edges.
    # Pruning:
    # 1. If len(current_sums_set) > K or len(current_prods_set) > K: return False.
    # 2. If remaining_edges < needed_edges: return False.
    # 3. If len(current_sums_set) + (needed_edges) < K is not a valid pruning for "distinct", 
    #    but if len(current_sums_set) is already K, we can only add edges whose sum is in current_sums_set.
    
    # Let's try K from 25 up to 35.
    # 25 is the number of possible sums for 1..14 (3..27).
    # So K=25 is a strong lower bound for sums.
    
    best_result = None
    
    for K in range(25, 36):
        if time.time() > deadline:
            break
            
        found = False
        # We will use a recursive backtracking search
        # To speed it up, we can sort edges by some heuristic?
        # Or just iterate.
        
        # Optimization: 
        # If K=25, we MUST use only sums that are in the range 3..27.
        # And we must hit exactly 25 distinct sums? No, <= 25.
        # Since there are only 25 possible sums for 1..14, if we pick 35 edges,
        # we are forced to have collisions.
        
        # Let's define the search function
        def search(edge_idx, current_edges, sums_set, prods_set):
            if time.time() > deadline:
                return None
            
            if len(current_edges) == m:
                # Check if valid
                if len(sums_set) <= K and len(prods_set) <= K:
                    return list(current_edges)
                return None
            
            remaining = m - len(current_edges)
            
            # Pruning: Not enough edges left
            if len(all_edges) - edge_idx < remaining:
                return None
            
            # Pruning: If we already exceeded K distinct sums/prods, stop
            if len(sums_set) > K or len(prods_set) > K:
                return None
            
            # Heuristic pruning:
            # If len(sums_set) == K, we can ONLY add edges whose sum is in sums_set.
            # If len(prods_set) == K, we can ONLY add edges whose prod is in prods_set.
            
            for i in range(edge_idx, len(all_edges)):
                e = all_edges[i]
                s = edge_sums[i]
                p = edge_prods[i]
                
                # Check if adding this edge violates K
                new_sums_size = len(sums_set) + (1 if s not in sums_set else 0)
                new_prods_size = len(prods_set) + (1 if p not in prods_set else 0)
                
                if new_sums_size > K or new_prods_size > K:
                    continue
                
                # Recursive step
                current_edges.append(e)
                sums_set.add(s)
                prods_set.add(p)
                
                res = search(i + 1, current_edges, sums_set, prods_set)
                
                if res is not None:
                    return res
                
                # Backtrack
                current_edges.pop()
                if s in sums_set and len(sums_set) > 0:
                    # Careful with set removal if multiple edges have same sum
                    # We can't just remove from set if another edge in current_edges has same sum.
                    # This makes state management tricky.
                    pass
                # Actually, standard backtracking with sets requires careful handling.
                # Better to pass copies or use a counter.
                # Given the small size (35 edges), passing copies might be slow but safe.
                # Or we can use a dict of counts.
                
                # Let's refactor to use counts for proper backtracking.
                # But for now, let's just return the first found.
                # If we find one, we return. We don't need to explore all.
                
            return None
        
        # The above search has a bug in backtracking sets.
        # Let's rewrite search to use dictionaries for counts.
        
        def search_v2(edge_idx, current_edges, sums_count, prods_count):
            if time.time() > deadline:
                return None
            
            if len(current_edges) == m:
                if len(sums_count) <= K and len(prods_count) <= K:
                    return list(current_edges)
                return None
            
            remaining = m - len(current_edges)
            if len(all_edges) - edge_idx < remaining:
                return None
            
            if len(sums_count) > K or len(prods_count) > K:
                return None
            
            for i in range(edge_idx, len(all_edges)):
                e = all_edges[i]
                s = edge_sums[i]
                p = edge_prods[i]
                
                new_sums_size = len(sums_count) + (1 if s not in sums_count else 0)
                new_prods_size = len(prods_count) + (1 if p not in prods_count else 0)
                
                if new_sums_size > K or new_prods_size > K:
                    continue
                
                current_edges.append(e)
                sums_count[s] = sums_count.get(s, 0) + 1
                prods_count[p] = prods_count.get(p, 0) + 1
                
                res = search_v2(i + 1, current_edges, sums_count, prods_count)
                
                if res is not None:
                    return res
                
                # Backtrack
                current_edges.pop()
                sums_count[s] -= 1
                if sums_count[s] == 0:
                    del sums_count[s]
                prods_count[p] -= 1
                if prods_count[p] == 0:
                    del prods_count[p]
            
            return None
        
        res = search_v2(0, [], {}, {})
        if res:
            # res is a list of edge indices
            # Convert to [u, v] pairs
            edges = [list(all_edges[i]) for i in res]
            # Calculate score
            distinct_sums = len(set(edge_sums[i] for i in res))
            distinct_prods = len(set(edge_prods[i] for i in res))
            score = max(distinct_sums, distinct_prods)
            
            # Check if this is better than any previous (though we iterate K up, so first found is best for this K)
            # Since we iterate K up, the first K that yields a solution is the minimum possible for this value set.
            return {
                'values': values,
                'edges': edges
            }
        
        # If no solution for this K, try larger K
        # Note: search_v2 might be slow. If it times out, we return whatever we have or fallback.
        
    # Fallback: Return the seed program's result or a random valid one if search failed/timed out
    # To ensure validity, we return a simple valid configuration
    return {
        'values': list(range(1, n + 1)),
        'edges': [list(e) for e in all_edges[:m]]
    }
