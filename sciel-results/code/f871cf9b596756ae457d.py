"""Strategy: Joint Greedy Construction with Local Search for Values and Edges."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 14
    m = 35
    
    all_pairs = list(itertools.combinations(range(n), 2))
    
    def evaluate(vals, edges):
        """Calculate the raw score for a given set of values and edges."""
        if not vals or not edges:
            return float('inf')
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(vals[u] + vals[v])
            prods.add(vals[u] * vals[v])
        return max(len(sums), len(prods))
    
    def greedy_edge_selection(vals):
        """
        Select 35 edges greedily to maximize collision with existing sums/products.
        We want to pick edges (u,v) such that vals[u]+vals[v] and vals[u]*vals[v]
        are already in the sets of sums/prods from previously picked edges.
        If not, we prefer values that create new sums/prods with the highest frequency potential.
        """
        # Pre-compute sums and prods for all pairs
        pair_sums = {}
        pair_prods = {}
        for u, v in all_pairs:
            pair_sums[(u, v)] = vals[u] + vals[v]
            pair_prods[(u, v)] = vals[u] * vals[v]
        
        # Count potential collisions if we were to pick all edges
        sum_counts = {}
        prod_counts = {}
        for s in pair_sums.values():
            sum_counts[s] = sum_counts.get(s, 0) + 1
        for p in pair_prods.values():
            prod_counts[p] = prod_counts.get(p, 0) + 1
            
        selected_edges = set()
        current_sums = set()
        current_prods = set()
        
        # We will pick 35 edges.
        # Heuristic: Pick edge that maximizes (1 if sum in current_sums else 0) + (1 if prod in current_prods else 0)
        # Tie-breaker: Pick edge with highest sum_counts[sum] + prod_counts[prod] to build towards dense clusters.
        
        for _ in range(m):
            best_edge = None
            best_score = -1
            best_tiebreak = -1
            
            # Optimization: Instead of checking all remaining pairs every time,
            # we can just iterate. 91 pairs is small.
            for e in all_pairs:
                if e in selected_edges:
                    continue
                
                s = pair_sums[e]
                p = pair_prods[e]
                
                score = 0
                if s in current_sums:
                    score += 1
                if p in current_prods:
                    score += 1
                
                # Tie-breaker: prefer edges that connect to high-frequency sums/prods overall
                tiebreak = sum_counts.get(s, 0) + prod_counts.get(p, 0)
                
                if score > best_score or (score == best_score and tiebreak > best_tiebreak):
                    best_score = score
                    best_tiebreak = tiebreak
                    best_edge = e
            
            if best_edge:
                selected_edges.add(best_edge)
                current_sums.add(pair_sums[best_edge])
                current_prods.add(pair_prods[best_edge])
                
        return list(selected_edges)

    def local_search(vals, edges, time_limit):
        """
        Improve the configuration by swapping values and swapping edges.
        """
        edges_set = set(tuple(e) for e in edges)
        current_raw = evaluate(vals, edges_set)
        best_raw = current_raw
        best_vals = list(vals)
        best_edges = list(edges_set)
        
        # Pre-compute pair data for current vals
        pair_sums = {}
        pair_prods = {}
        for u, v in all_pairs:
            pair_sums[(u, v)] = vals[u] + vals[v]
            pair_prods[(u, v)] = vals[u] * vals[v]
            
        # Count occurrences in current edges
        sum_counts = {}
        prod_counts = {}
        for u, v in edges_set:
            s = pair_sums[(u, v)]
            p = pair_prods[(u, v)]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
            
        def get_raw_from_counts():
            return max(len(sum_counts), len(prod_counts))
        
        # Local Search Loop
        while time.time() < time_limit:
            improved = False
            
            # 1. Try swapping values
            # Pick two random indices i, j and swap their values
            # Or pick i and change its value to a random candidate?
            # Swapping is safer to keep distinctness if we start with distinct values.
            # Let's try swapping values at random positions.
            for _ in range(50):
                if time.time() > time_limit:
                    break
                i, j = random.sample(range(n), 2)
                if vals[i] == vals[j]:
                    continue
                old_vi, old_vj = vals[i], vals[j]
                vals[i], vals[j] = vals[j], vals[i]
                
                # Recalculate affected edges
                # Edges incident to i or j are affected.
                # This is complex to update incrementally.
                # Given n=14, m=35, full re-evaluation is fast enough?
                # 35 edges * 2 ops = 70 ops. Very fast.
                new_raw = evaluate(vals, edges_set)
                
                if new_raw < best_raw:
                    best_raw = new_raw
                    best_vals = list(vals)
                    best_edges = list(edges_set)
                    improved = True
                else:
                    # Revert
                    vals[i], vals[j] = old_vi, old_vj
                    
            # 2. Try swapping edges
            # Remove one edge, add one edge.
            for _ in range(50):
                if time.time() > time_limit:
                    break
                e_rm = random.choice(list(edges_set))
                # Pick e_add not in edges_set
                e_add = random.choice(all_pairs)
                while e_add in edges_set:
                    e_add = random.choice(all_pairs)
                
                # Update counts
                s_rm = pair_sums[e_rm]
                p_rm = pair_prods[e_rm]
                s_ad = pair_sums[e_add]
                p_ad = pair_prods[e_add]
                
                # Remove e_rm
                sum_counts[s_rm] -= 1
                if sum_counts[s_rm] == 0:
                    del sum_counts[s_rm]
                prod_counts[p_rm] -= 1
                if prod_counts[p_rm] == 0:
                    del prod_counts[p_rm]
                
                edges_set.remove(e_rm)
                
                # Add e_add
                sum_counts[s_ad] = sum_counts.get(s_ad, 0) + 1
                prod_counts[p_ad] = prod_counts.get(p_ad, 0) + 1
                
                edges_set.add(e_add)
                
                new_raw = get_raw_from_counts()
                
                if new_raw < best_raw:
                    best_raw = new_raw
                    best_vals = list(vals)
                    best_edges = list(edges_set)
                    improved = True
                else:
                    # Revert
                    edges_set.remove(e_add)
                    sum_counts[s_ad] -= 1
                    if sum_counts[s_ad] == 0:
                        del sum_counts[s_ad]
                    prod_counts[p_ad] -= 1
                    if prod_counts[p_ad] == 0:
                        del prod_counts[p_ad]
                        
                    edges_set.add(e_rm)
                    sum_counts[s_rm] = sum_counts.get(s_rm, 0) + 1
                    prod_counts[p_rm] = prod_counts.get(p_rm, 0) + 1
                    
                # Update pair_sums/prods if values changed? 
                # In this loop, values are fixed (only edge swaps). 
                # If values were swapped in step 1, pair_sums/prods are stale for step 2.
                # We need to recompute pair_sums/prods if values changed.
                # To keep it simple and robust, if we do value swaps, we should recompute pair data.
                # But doing that every value swap is expensive.
                # Alternative: Only do edge swaps in local search. Value optimization is done by restarting with new random values.
                # This is a valid structural choice: "Restart Local Search".
                
            if not improved:
                # If no improvement in edge swaps, break to try a new value set
                break
                
        return best_vals, list(best_edges), best_raw

    best_overall_raw = float('inf')
    best_overall_vals = None
    best_overall_edges = None
    
    # Initial Candidates
    # We want value sets that naturally create collisions.
    # 1. Consecutive small integers
    # 2. Multiples of small numbers
    # 3. Random small integers
    
    candidate_generators = []
    
    # Gen 1: Consecutive
    for start in range(1, 5):
        candidate_generators.append(lambda s=start: [s + i for i in range(n)])
        
    # Gen 2: Multiples
    for step in range(2, 10):
        candidate_generators.append(lambda st=step: [i * st for i in range(1, n + 1)])
        
    # Gen 3: Random small
    for _ in range(10):
        candidate_generators.append(lambda: sorted(random.sample(range(1, 100), n)))
        
    # Gen 4: "Smooth" numbers (highly composite)
    # e.g. 1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 27
    candidate_generators.append(lambda: [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 27])
    candidate_generators.append(lambda: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    
    # Run the main loop
    for gen in candidate_generators:
        if time.time() > deadline - 5:
            break
            
        # Generate a batch of random variations if it's a random generator, or just use it directly
        # To ensure diversity, we run local search on multiple random permutations/perturbations of the base set?
        # No, let's just run local search on the generated set.
        
        vals = gen()
        if len(set(vals)) != n or any(v < 1 or v > 10000 for v in vals):
            continue
            
        # Greedy edge selection
        edges = greedy_edge_selection(vals)
        
        # Local search (Edge swaps only, as per design decision above to avoid stale pair data complexity)
        # But wait, if we don't change values, we are stuck in the value basin.
        # Let's add a simple "Value Perturbation" phase before edge local search.
        # Or better: The "Restart" strategy.
        # We will run Local Search on the current (vals, edges).
        # If we want to change values, we do it in the outer loop by generating new candidates.
        # However, the current program's weakness was fixing values too early.
        # Let's allow value swaps in local search, but recompute pair data only when a value swap is accepted.
        
        # Revised Local Search:
        # 1. Init pair data
        # 2. Loop:
        #    a. Try edge swaps (fast, uses current pair data)
        #    b. Try value swaps (slow, requires recompute of pair data for incident edges or full recompute)
        #       Since n=14, full recompute of pair_sums/prods is 91 pairs. Very fast.
        #       So we can afford to recompute pair_sums/prods after every value swap attempt?
        #       91 pairs * 2 ops = 182 ops.
        #       Evaluation of 35 edges = 70 ops.
        #       Total ~250 ops per value swap.
        #       If we do 100 value swaps, that's 25,000 ops. Negligible.
        
        # Let's implement a unified local search.
        
        current_vals = list(vals)
        current_edges_set = set(tuple(e) for e in edges)
        
        # Recompute pair data
        pair_sums = {}
        pair_prods = {}
        for u, v in all_pairs:
            pair_sums[(u, v)] = current_vals[u] + current_vals[v]
            pair_prods[(u, v)] = current_vals[u] * current_vals[v]
            
        sum_counts = {}
        prod_counts = {}
        for u, v in current_edges_set:
            s = pair_sums[(u, v)]
            p = pair_prods[(u, v)]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
            
        current_raw = max(len(sum_counts), len(prod_counts))
        best_raw = current_raw
        best_vals = list(current_vals)
        best_edges = list(current_edges_set)
        
        no_improve_count = 0
        
        while time.time() < deadline - 2:
            improved = False
            
            # Phase 1: Edge Swaps
            for _ in range(100):
                e_rm = random.choice(list(current_edges_set))
                e_add = random.choice(all_pairs)
                while e_add in current_edges_set:
                    e_add = random.choice(all_pairs)
                
                s_rm = pair_sums[e_rm]
                p_rm = pair_prods[e_rm]
                s_ad = pair_sums[e_add]
                p_ad = pair_prods[e_add]
                
                # Remove
                sum_counts[s_rm] -= 1
                if sum_counts[s_rm] == 0: del sum_counts[s_rm]
                prod_counts[p_rm] -= 1
                if prod_counts[p_rm] == 0: del prod_counts[p_rm]
                current_edges_set.remove(e_rm)
                
                # Add
                sum_counts[s_ad] = sum_counts.get(s_ad, 0) + 1
                prod_counts[p_ad] = prod_counts.get(p_ad, 0) + 1
                current_edges_set.add(e_add)
                
                new_raw = max(len(sum_counts), len(prod_counts))
                
                if new_raw < best_raw:
                    best_raw = new_raw
                    best_vals = list(current_vals)
                    best_edges = list(current_edges_set)
                    improved = True
                else:
                    # Revert
                    current_edges_set.remove(e_add)
                    sum_counts[s_ad] -= 1
                    if sum_counts[s_ad] == 0: del sum_counts[s_ad]
                    prod_counts[p_ad] -= 1
                    if prod_counts[p_ad] == 0: del prod_counts[p_ad]
                    
                    current_edges_set.add(e_rm)
                    sum_counts[s_rm] = sum_counts.get(s_rm, 0) + 1
                    prod_counts[p_rm] = prod_counts.get(p_rm, 0) + 1
                    
            # Phase 2: Value Swaps
            for _ in range(50):
                i, j = random.sample(range(n), 2)
                if current_vals[i] == current_vals[j]:
                    continue
                old_vi, old_vj = current_vals[i], current_vals[j]
                current_vals[i], current_vals[j] = current_vals[j], current_vals[i]
                
                # Recompute pair data for all pairs? Or just incident?
                # All pairs is simpler and fast enough (91 pairs).
                for u, v in all_pairs:
                    pair_sums[(u, v)] = current_vals[u] + current_vals[v]
                    pair_prods[(u, v)] = current_vals[u] * current_vals[v]
                    
                # Recompute counts from scratch? Or update?
                # Update is tricky because many edges change values.
                # Just recompute counts from scratch.
                new_sum_counts = {}
                new_prod_counts = {}
                for u, v in current_edges_set:
                    s = pair_sums[(u, v)]
                    p = pair_prods[(u, v)]
                    new_sum_counts[s] = new_sum_counts.get(s, 0) + 1
                    new_prod_counts[p] = new_prod_counts.get(p, 0) + 1
                    
                new_raw = max(len(new_sum_counts), len(new_prod_counts))
                
                if new_raw < best_raw:
                    best_raw = new_raw
                    best_vals = list(current_vals)
                    best_edges = list(current_edges_set)
                    improved = True
                    # Update state
                    sum_counts = new_sum_counts
                    prod_counts = new_prod_counts
                else:
                    # Revert
                    current_vals[i], current_vals[j] = old_vi, old_vj
                    # Pair data is now stale, but we will recompute it if we accept a future swap.
                    # For correctness of next edge swap, we need correct pair data.
                    # So we must revert pair data too?
                    # Easier: Just recompute pair data from current_vals (which is reverted)
                    for u, v in all_pairs:
                        pair_sums[(u, v)] = current_vals[u] + current_vals[v]
                        pair_prods[(u, v)] = current_vals[u] * current_vals[v]
                        
            if not improved:
                no_improve_count += 1
                if no_improve_count > 3:
                    break
            else:
                no_improve_count = 0
                
        if best_raw < best_overall_raw:
            best_overall_raw = best_raw
            best_overall_vals = best_vals
            best_overall_edges = best_edges
            
    # Final Result Construction
    if best_overall_vals is None:
        # Fallback
        vals = list(range(1, 15))
        edges = greedy_edge_selection(vals)
        best_overall_vals = vals
        best_overall_edges = edges
        
    # Ensure types are correct
    final_vals = [int(v) for v in best_overall_vals]
    final_edges = [[int(u), int(v)] for u, v in best_overall_edges]
    final_edges.sort()
    
    return {
        'values': final_vals,
        'edges': final_edges
    }
# EVOLVE-BLOCK-END
