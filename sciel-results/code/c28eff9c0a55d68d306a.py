"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 14
    m = 35
    maximum = 10000

    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            v = values[i]
            w = values[j]
            sums.add(v + w)
            prods.add(v * w)
        return max(len(sums), len(prods))

    def greedy_edge_selection(values):
        """
        Selects edges one by one.
        At each step, choose the edge (i,j) not yet selected that minimizes the
        increase in max(distinct_sums, distinct_prods).
        Ties are broken by favoring edges that create more collisions (i.e., 
        edges where sum or prod is already in the set).
        """
        all_edges = list(combinations(range(n), 2))
        selected_edges = []
        current_sums = set()
        current_prods = set()
        
        # Pre-compute sums and prods for all edges to speed up selection
        edge_props = {}
        for i, j in all_edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_props[(i,j)] = (s, p)

        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            best_collision_count = -1
            
            # To speed this up, we could limit candidates, but 91 edges is small.
            # However, iterating 91 * 35 times is ~3000 ops, which is trivial.
            for e in all_edges:
                if e in selected_edges:
                    continue
                
                s, p = edge_props[e]
                
                # Calculate new score if we add this edge
                # Note: We don't actually add it to the set yet, just simulate
                new_sums_count = len(current_sums) + (0 if s in current_sums else 1)
                new_prods_count = len(current_prods) + (0 if p in current_prods else 1)
                
                new_score = max(new_sums_count, new_prods_count)
                
                # Secondary metric: prefer edges that collide more (higher chance of future collisions?)
                # Actually, just minimizing the immediate score is the primary driver.
                # If scores are equal, prefer the one with more collisions (s in sums OR p in prods)
                collisions = (1 if s in current_sums else 0) + (1 if p in current_prods else 0)
                
                if new_score < best_score or (new_score == best_score and collisions > best_collision_count):
                    best_score = new_score
                    best_collision_count = collisions
                    best_edge = e
            
            if best_edge is not None:
                selected_edges.append(best_edge)
                s, p = edge_props[best_edge]
                current_sums.add(s)
                current_prods.add(p)
                
        return selected_edges

    def local_search_values(values, edges):
        """
        Given a fixed set of edges, try to improve the value set.
        Uses a simple hill climbing / random restart on values.
        """
        current_values = list(values)
        current_score = _raw_value(current_values, edges)
        best_values = list(current_values)
        best_score = current_score
        
        # Pre-calculate edge indices for speed
        edge_list = list(edges)
        
        steps = 500
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            
            # Pick a random vertex to change
            idx = random.randint(0, n-1)
            old_val = current_values[idx]
            
            # Try a few random new values
            for _ in range(5):
                new_val = random.randint(1, maximum)
                if new_val == old_val:
                    continue
                # Ensure distinctness
                if new_val in current_values and current_values.index(new_val) != idx:
                    continue
                
                # Swap
                current_values[idx] = new_val
                
                # Calculate new score
                # Only need to recalculate edges incident to idx
                new_sums = set()
                new_prods = set()
                
                # To be safe and simple, recompute all. 
                # 35 edges * 2 ops = 70 ops. Very fast.
                for i, j in edge_list:
                    new_sums.add(current_values[i] + current_values[j])
                    new_prods.add(current_values[i] * current_values[j])
                
                new_score = max(len(new_sums), len(new_prods))
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_values = list(current_values)
                        break # Accept improvement, restart local search on this new base? Or continue?
                    # If it improved current but not best, keep it and continue
                else:
                    # Revert if no improvement (strict hill climbing)
                    current_values[idx] = old_val
                    break # If first try failed, revert and try next vertex? 
                    # Actually, let's try more values for this vertex if first fails?
                    # No, let's just revert and move on to next step to explore space.
            
            # If we found a global best, we could restart from there, but for now just keep climbing
        
        return best_values, best_score

    # Initial candidate: Small integers often have good collision properties for sums
    # e.g., 1..14
    initial_values = list(range(1, n+1))
    
    # 1. Get a good edge set for this initial value set
    edges = greedy_edge_selection(initial_values)
    
    # 2. Refine values for this edge set
    best_values, best_score = local_search_values(initial_values, edges)
    
    # 3. Try a few other structural starting points
    # Powers of 2 (bad for collisions, but good baseline)
    # Arithmetic Progressions (good for sum collisions)
    
    start_configs = [
        list(range(1, n+1)),
        [i*2 for i in range(1, n+1)], # AP step 2
        [i*3 for i in range(1, n+1)], # AP step 3
        # Random small values
        random.sample(range(1, 100), n),
        random.sample(range(1, 100), n),
    ]
    
    for config in start_configs:
        if time.time() > deadline - 10:
            break
        
        # Ensure distinctness
        if len(set(config)) != n:
            continue
            
        # 1. Greedy edges
        edges_c = greedy_edge_selection(config)
        
        # 2. Local search values
        vals_c, score_c = local_search_values(config, edges_c)
        
        if score_c < best_score:
            best_score = score_c
            best_values = vals_c
            # Keep the edges that produced this score
            # Note: local_search_values doesn't return edges, it assumes edges are fixed.
            # So we must use edges_c for this config.
            # We need to store the best edges corresponding to best_values.
            # Let's refactor slightly to track (values, edges, score)
            
            # Wait, the local search changes values but keeps edges fixed.
            # So the pair is (vals_c, edges_c).
            # I need to store the best edges too.
            
    # Let's redo the loop properly to track edges
    
    best_final_score = float('inf')
    best_final_values = None
    best_final_edges = None
    
    # Re-run the search with proper tracking
    start_configs = [
        list(range(1, n+1)),
        [i*2 for i in range(1, n+1)],
        [i*3 for i in range(1, n+1)],
        random.sample(range(1, 200), n),
        random.sample(range(1, 200), n),
        random.sample(range(1, 1000), n),
    ]
    
    for config in start_configs:
        if time.time() > deadline - 5:
            break
        
        if len(set(config)) != n:
            continue
        
        edges_c = greedy_edge_selection(config)
        vals_c, score_c = local_search_values(config, edges_c)
        
        # Verify score
        actual_score = _raw_value(vals_c, edges_c)
        
        if actual_score < best_final_score:
            best_final_score = actual_score
            best_final_values = vals_c
            best_final_edges = edges_c

    # If no result found (shouldn't happen), fallback
    if best_final_values is None:
        best_final_values = list(range(1, n+1))
        best_final_edges = greedy_edge_selection(best_final_values)

    return {
        'values': [int(v) for v in best_final_values],
        'edges': [[int(i), int(j)] for i, j in best_final_edges]
    }
# EVOLVE-BLOCK-END
