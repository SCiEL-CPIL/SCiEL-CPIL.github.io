import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    
    # Candidate value sets: 
    # 1. Powers of 2 (minimizes sum collisions, products are unique powers)
    # 2. Small consecutive integers (maximizes sum collisions, products vary)
    # 3. Small integers with some structure
    candidates_values = [
        [2**i for i in range(n)],
        list(range(1, n+1)),
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610],
        [1, 2, 4, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
    ]
    
    # Filter values > 10000
    valid_candidates = []
    for vals in candidates_values:
        if all(1 <= v <= 10000 for v in vals) and len(set(vals)) == n:
            valid_candidates.append(vals)
            
    if not valid_candidates:
        valid_candidates = [list(range(1, n+1))]
        
    best_score = float('inf')
    best_result = None
    
    # Greedy edge selection for each candidate value set
    for values in valid_candidates:
        if time.time() > deadline:
            break
            
        used_edges = set()
        current_edges = []
        sums = set()
        prods = set()
        
        # Precompute all possible edge sums and products
        edge_data = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_data.append((u, v, s, p))
        
        # Sort edges by some heuristic? 
        # A good heuristic: edges that reuse existing sums/products are "cheap".
        # Initially all are "expensive" (add 1 to both sets).
        # We want to minimize max(|S|, |P|).
        
        # Greedy: At each step, pick the edge that adds the fewest new elements to S and P.
        # Ties broken by... maybe sum value?
        
        remaining = list(range(len(edge_data)))
        
        for _ in range(m):
            if time.time() > deadline:
                break
                
            best_idx = -1
            best_cost = float('inf')
            best_new_s = None
            best_new_p = None
            
            # Optimization: Only check a subset if time is tight? 
            # For 91 edges, 35 steps, it's fast enough.
            
            for i in remaining:
                u, v, s, p = edge_data[i]
                new_s = 0 if s in sums else 1
                new_p = 0 if p in prods else 1
                # We want to minimize the resulting max size.
                # Current max: max(len(sums), len(prods))
                # New max: max(len(sums)+new_s, len(prods)+new_p)
                
                current_max = max(len(sums), len(prods))
                new_max = max(len(sums) + new_s, len(prods) + new_p)
                
                # Heuristic: Minimize new_max. If tie, minimize total additions.
                cost = (new_max, new_s + new_p, s, p)
                
                if cost < best_cost:
                    best_cost = cost
                    best_idx = i
                    best_new_s = s
                    best_new_p = p
            
            if best_idx == -1:
                break
                
            remaining.remove(best_idx)
            u, v, s, p = edge_data[best_idx]
            current_edges.append([u, v])
            sums.add(s)
            prods.add(p)
            
        if len(current_edges) < m:
            # If we didn't pick enough, fill with remaining
            for i in remaining:
                if len(current_edges) >= m:
                    break
                u, v, s, p = edge_data[i]
                current_edges.append([u, v])
                sums.add(s)
                prods.add(p)
                
        score = max(len(sums), len(prods))
        
        if score < best_score:
            best_score = score
            best_result = {'values': values, 'edges': current_edges}
            
            # If we hit the theoretical minimum or close, we can stop?
            # Min possible sums: m edges. If all sums distinct, |S|=m.
            # Min possible prods: m edges. If all prods distinct, |P|=m.
            # So lower bound is m=35.
            if best_score <= 35:
                break

    if best_result is None:
        # Fallback
        best_result = {'values': list(range(1, n+1)), 'edges': [list(e) for e in all_edges[:m]]}
        
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return best_result
