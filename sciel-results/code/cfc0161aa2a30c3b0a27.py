"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    # Helper to evaluate a candidate (values, edges)
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    # Helper to get all possible edges
    all_edges = list(combinations(range(n), 2))
    
    # Initial strategy: Geometric progression for values
    # This makes products distinct (mostly) and sums somewhat structured.
    # Base 2: 1, 2, 4, ..., 2^13 = 8192. All <= 10000.
    base_values = [2**i for i in range(n)]
    
    best_score = float('inf')
    best_values = base_values.copy()
    best_edges = []
    
    # Initial edge selection: 
    # For geometric progression, products are 2^(i+j). Sums are 2^i + 2^j.
    # We want to minimize max(distinct_sums, distinct_prods).
    # With 35 edges, if we pick edges such that sums collide, we reduce distinct sums.
    # But products of distinct powers of 2 are always distinct.
    # So distinct_prods will be 35.
    # To beat this, we need distinct_sums < 35 AND distinct_prods < 35? No, max.
    # If we can make distinct_sums <= 35 and distinct_prods <= 35, score is 35.
    # Can we get lower?
    # If we use values that are not powers of 2, products might collide.
    # Example: 1, 2, 3, 4... 
    # 1*4 = 2*2 (not allowed, distinct vertices? 1*4 uses idx 0,3; 2*2 not allowed as loop? 
    # Edges are between different vertices. So 1*4 and 2*2 is not a valid comparison for edges unless we have edge (0,3) and edge (1,1) which is loop.
    # So product collisions require distinct pairs (u,v) and (w,x) with u!=v, w!=x, {u,v}!={w,x} such that v_u*v_v == v_w*v_v.
    # Example: 1*6 = 2*3. Vertices 0,5 and 1,2.
    
    # Let's use a random restart local search.
    
    def random_values():
        # Generate random distinct positive integers <= 10000
        # Bias towards smaller numbers to increase collision chance?
        # Or bias towards specific structures?
        # Let's try a mix.
        return sorted(random.sample(range(1, max_val+1), n))
    
    def optimize_edges_for_values(values, max_time=None):
        # Greedy edge selection:
        # Start with no edges.
        # We want to pick 35 edges.
        # To minimize distinct sums/prods, we should pick edges that "reuse" existing sum/product values.
        # However, the first edge creates 1 sum and 1 prod.
        # Subsequent edges should ideally add 0 new sums and 0 new prods, or 1 new sum and 0 new prod, etc.
        
        # This is a set cover / packing problem.
        # Greedy:
        # 1. Compute all possible (sum, prod) for all pairs.
        # 2. We need to select 35 pairs.
        # 3. Minimize |Union of sums| and |Union of prods|.
        
        # Precompute features
        # pair_feat[i][j] = (sum, prod)
        
        # Heuristic: 
        # Sort all edges by (sum, prod) frequency? No, we don't know frequencies until we pick.
        # Let's try: 
        # 1. Find the most "popular" sums and products among all possible edges?
        # 2. Pick edges that share these.
        
        # Simpler Greedy:
        # Start with the edge that has the "smallest" sum/prod?
        # Or just random?
        
        # Let's try a greedy that picks edges minimizing the increase in unique counts.
        # State: current_set_sums, current_set_prods
        # For each step, evaluate all remaining edges.
        # Cost(edge) = (1 if sum not in current_set_sums else 0) + (1 if prod not in current_set_prods else 0)
        # Pick edge with min cost. Tie-break randomly.
        
        current_sums = set()
        current_prods = set()
        selected_edges = []
        remaining = set(range(len(all_edges)))
        
        # Precompute sum/prod for all edges for speed
        edge_sums = [values[u] + values[v] for u, v in all_edges]
        edge_prods = [values[u] * values[v] for u, v in all_edges]
        
        for _ in range(m):
            best_idx = -1
            best_cost = float('inf')
            best_tie_break = float('inf')
            
            # Optimization: If time is running out, just pick random from remaining
            if max_time and time.time() > max_time:
                idx = random.choice(list(remaining))
                selected_edges.append(all_edges[idx])
                current_sums.add(edge_sums[idx])
                current_prods.add(edge_prods[idx])
                remaining.remove(idx)
                continue
                
            # Sample a subset of remaining edges if too many?
            # 91 edges total. 35 to pick. 91 is small enough to check all each time.
            for idx in remaining:
                s = edge_sums[idx]
                p = edge_prods[idx]
                
                cost_s = 0 if s in current_sums else 1
                cost_p = 0 if p in current_prods else 1
                cost = cost_s + cost_p
                
                # Tie breaker: prefer edges that contribute to "dense" areas?
                # Or just random.
                tie_break = random.random()
                
                if cost < best_cost or (cost == best_cost and tie_break < best_tie_break):
                    best_cost = cost
                    best_idx = idx
                    best_tie_break = tie_break
            
            if best_idx == -1:
                # Fallback
                best_idx = random.choice(list(remaining))
                
            selected_edges.append(all_edges[best_idx])
            current_sums.add(edge_sums[best_idx])
            current_prods.add(edge_prods[best_idx])
            remaining.remove(best_idx)
            
        return selected_edges

    def optimize_values_for_edges(edges, max_time=None):
        # Local search on values.
        # Pick a random vertex, change its value, see if score improves.
        current_values = best_values.copy()
        current_score = evaluate(current_values, edges)
        
        improved = True
        while improved and (not max_time or time.time() < max_time):
            improved = False
            # Random order of vertices
            verts = list(range(n))
            random.shuffle(verts)
            for v in verts:
                # Try changing value[v]
                original_val = current_values[v]
                # Propose new value: random in [1, 10000]
                # Ensure distinctness
                new_val = random.randint(1, max_val)
                attempts = 0
                while new_val in current_values and attempts < 10:
                    new_val = random.randint(1, max_val)
                    attempts += 1
                if new_val in current_values:
                    continue
                    
                current_values[v] = new_val
                new_score = evaluate(current_values, edges)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current_values[v] = original_val
                    
        return current_values, current_score

    # Main Loop
    best_score = float('inf')
    best_result = None
    
    # Start with geometric progression
    current_values = base_values.copy()
    
    # Initial edge optimization
    current_edges = optimize_edges_for_values(current_values)
    current_score = evaluate(current_values, current_edges)
    
    if current_score < best_score:
        best_score = current_score
        best_result = {'values': current_values.copy(), 'edges': [list(e) for e in current_edges]}
        
    # Local Search Loop
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Alternate: optimize edges, then values
        # 1. Optimize edges for current values
        new_edges = optimize_edges_for_values(current_values, max_time=deadline - 5)
        new_score = evaluate(current_values, new_edges)
        
        if new_score < best_score:
            best_score = new_score
            best_result = {'values': current_values.copy(), 'edges': [list(e) for e in new_edges]}
            current_edges = new_edges
            current_score = new_score
        else:
            # If edge optimization didn't help, maybe values need change
            # But keep current edges for value optimization?
            # Let's try optimizing values for the best edges found so far
            pass
            
        # 2. Optimize values for current (or best) edges
        # Use the edges that gave the best score so far
        target_edges = best_result['edges'] if best_result else current_edges
        
        new_values, new_val_score = optimize_values_for_edges(target_edges, max_time=deadline - 5)
        
        if new_val_score < best_score:
            best_score = new_val_score
            best_result = {'values': new_values.copy(), 'edges': [list(e) for e in target_edges]}
            current_values = new_values
            current_score = new_val_score
        else:
            # If no improvement, perturb values to escape local minima
            # Random restart for values?
            if iterations % 5 == 0:
                # Try a different random value set
                # Maybe geometric with different base?
                # Or random?
                # Let's try random sampling
                candidate_values = random_values()
                # Quick edge opt
                cand_edges = optimize_edges_for_values(candidate_values, max_time=deadline - 5)
                cand_score = evaluate(candidate_values, cand_edges)
                
                if cand_score < best_score:
                    best_score = cand_score
                    best_result = {'values': candidate_values.copy(), 'edges': [list(e) for e in cand_edges]}
                    current_values = candidate_values
                    current_edges = cand_edges
                    current_score = cand_score

    # Final check
    if best_result is None:
        # Fallback to initial
        best_result = {'values': base_values, 'edges': [list(e) for e in current_edges]}
        
    # Ensure types are correct
    res = {
        'values': [int(v) for v in best_result['values']],
        'edges': [[int(u), int(v)] for u, v in best_result['edges']]
    }
    
    return res

# EVOLVE-BLOCK-END
