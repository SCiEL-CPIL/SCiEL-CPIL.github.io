"""Strategy: Deterministic structured value pools and greedy maximum-weight clique approximation for edge selection."""
# EVOLVE-BLOCK-START
import itertools
import time
import heapq
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    
    # 1. Generate candidate value sets.
    # We use "smooth numbers" (numbers with only small prime factors) because 
    # products of smooth numbers are more likely to collide with other products,
    # and sums of smooth numbers in specific ranges can also be controlled.
    
    candidate_sets = []
    
    # Set A: Powers of 2 and 3 (classic multiplicative structure)
    set_a = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    candidate_sets.append(set_a)
    
    # Set B: Powers of 2, 3, and 5 (more collision potential)
    set_b = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    candidate_sets.append(set_b)
    
    # Set C: Arithmetic progression with small steps (good for sums)
    set_c = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    candidate_sets.append(set_c)
    
    # Set D: Mixed small integers
    set_d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    candidate_sets.append(set_d)
    
    # Set E: Geometric progression base 2, then perturb
    set_e = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    candidate_sets.append(set_e)
    
    # Set F: Randomized smooth numbers
    smooth_primes = [2, 3, 5, 7]
    random.seed(42) # Deterministic for reproducibility
    set_f = set()
    while len(set_f) < n:
        v = 1
        for _ in range(random.randint(1, 5)):
            v *= random.choice(smooth_primes)
        if v <= 10000:
            set_f.add(v)
    candidate_sets.append(list(set_f))
    
    def evaluate(values, edges):
        distinct_sums = set()
        distinct_prods = set()
        for u, v in edges:
            distinct_sums.add(values[u] + values[v])
            distinct_prods.add(values[u] * values[v])
        return max(len(distinct_sums), len(distinct_prods))

    def select_edges_greedy(values):
        """
        Select m edges using a greedy approach that prioritizes edges 
        that create the most collisions (i.e., reuse existing sum/product values).
        
        We model this as a priority queue where the "weight" of an edge is 
        the number of times its sum and product have already appeared in 
        the selected edge set.
        """
        all_edges = list(itertools.combinations(range(n), 2))
        
        # Precompute sums and products for all edges
        edge_sum = {}
        edge_prod = {}
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_sum[(u, v)] = s
            edge_prod[(u, v)] = p
            
        # Count frequency of each sum and product across ALL possible edges
        # This helps identify "dense" regions of the value space
        sum_freq = {}
        prod_freq = {}
        for u, v in all_edges:
            s = edge_sum[(u, v)]
            p = edge_prod[(u, v)]
            sum_freq[s] = sum_freq.get(s, 0) + 1
            prod_freq[p] = prod_freq.get(p, 0) + 1
            
        selected_edges = []
        used_vertices = [0] * n # Not strictly necessary for simple graph, but good for future extensions
        current_sums = set()
        current_prods = set()
        
        # We will greedily pick edges. 
        # Score for an edge (u,v) given current selection:
        # +1 if sum is in current_sums
        # +1 if prod is in current_prods
        # + small bonus if sum/prod are frequent in the global space (heuristic)
        
        # To speed up, we can just iterate through edges sorted by "potential" 
        # and pick the best available one at each step.
        
        # Initial potential: global frequency
        potentials = []
        for u, v in all_edges:
            s = edge_sum[(u, v)]
            p = edge_prod[(u, v)]
            # Higher global frequency means more likely to collide with other edges
            pot = sum_freq[s] + prod_freq[p]
            potentials.append((pot, u, v))
        
        # Sort by potential descending
        potentials.sort(key=lambda x: -x[0])
        
        # We will maintain a heap of candidates.
        # However, a simpler greedy: 
        # 1. Start with empty set.
        # 2. Find edge with max (current_collisions + global_potential).
        # 3. Add it, update current sets.
        # 4. Repeat m times.
        
        # Optimization: Instead of recalculating all scores every time (O(M^2 * N^2)),
        # we can just use the global potential as a proxy, and break ties by 
        # checking current collisions.
        
        # Let's try a slightly smarter greedy:
        # Group edges by sum. Within a group, group by product.
        # Pick the largest group first? No, that's just frequency.
        
        # Let's stick to a priority queue approach where we update scores incrementally.
        # But for 91 edges, O(91^2) is small.
        
        best_edges = []
        remaining_edges = set(all_edges)
        
        for _ in range(m):
            best_edge = None
            best_score = -1
            for e in remaining_edges:
                u, v = e
                s = edge_sum[e]
                p = edge_prod[e]
                score = 0
                if s in current_sums:
                    score += 10
                if p in current_prods:
                    score += 10
                # Tie-breaker: global frequency
                score += sum_freq[s] * 0.1 + prod_freq[p] * 0.1
                if score > best_score:
                    best_score = score
                    best_edge = e
                    
            if best_edge:
                best_edges.append(best_edge)
                remaining_edges.remove(best_edge)
                current_sums.add(edge_sum[best_edge])
                current_prods.add(edge_prod[best_edge])
            else:
                break
                
        return best_edges

    best_result = None
    best_score = float('inf')
    
    # Evaluate each candidate set with greedy edge selection
    for i, vals in enumerate(candidate_sets):
        if time.time() > deadline:
            break
            
        # Ensure distinctness and range
        if len(set(vals)) != n or any(v < 1 or v > 10000 for v in vals):
            continue
            
        edges = select_edges_greedy(vals)
        if len(edges) < m:
            continue
            
        score = evaluate(vals, edges)
        
        if score < best_score:
            best_score = score
            best_result = (list(vals), list(edges))
            
    # If we have time, do a local search on the best result
    if time.time() < deadline - 5.0 and best_result:
        current_vals, current_edges = best_result
        current_score = best_score
        best_vals = list(current_vals)
        best_edges = list(current_edges)
        
        # Simple local search: swap values or edges
        all_edges = list(itertools.combinations(range(n), 2))
        
        # Define a few restarts or just iterate
        for _ in range(1000): # Fixed small number of local steps
            if time.time() > deadline - 1.0:
                break
                
            # Try mutating an edge
            new_edges = list(current_edges)
            idx = random.randint(0, len(new_edges) - 1)
            old_edge = new_edges[idx]
            
            # Pick a random alternative edge
            new_edge = random.choice(all_edges)
            if new_edge not in new_edges:
                new_edges[idx] = new_edge
                
                # Check validity
                if len(set(new_edges)) == m:
                    new_score = evaluate(current_vals, new_edges)
                    if new_score < current_score:
                        current_edges = new_edges
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_vals = list(current_vals)
                            best_edges = list(current_edges)
                            
    if best_result is None:
        # Fallback
        vals = list(range(1, 15))
        edges = list(itertools.combinations(range(n), 2))[:m]
        return {'values': vals, 'edges': [[int(u), int(v)] for u, v in edges]}
        
    vals, edges = best_result
    return {'values': [int(v) for v in vals], 'edges': [[int(u), int(v)] for u, v in edges]}
# EVOLVE-BLOCK-END
