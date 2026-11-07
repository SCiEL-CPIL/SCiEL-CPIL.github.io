import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    
    def score_of(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def greedy_edge_selection(values, m, deadline):
        """
        Greedily select m edges that minimize the number of distinct sums and products.
        """
        edge_sum = {}
        edge_prod = {}
        for u, v in all_edges:
            edge_sum[(u, v)] = values[u] + values[v]
            edge_prod[(u, v)] = values[u] * values[v]
        
        selected_edges = []
        selected_sums = set()
        selected_prods = set()
        
        for _ in range(m):
            if time.time() > deadline - 5.0:
                break
            best_edge = None
            best_score = float('inf')
            
            # To speed up, we can limit candidates or just iterate all 91
            for e in all_edges:
                if e in selected_edges:
                    continue
                s = edge_sum[e]
                p = edge_prod[e]
                
                # Score: number of NEW distinct sums and products added
                new_sums = 0 if s in selected_sums else 1
                new_prods = 0 if p in selected_prods else 1
                # We want to minimize the max of total distinct sums and prods
                # Approximation: minimize (new_sums + new_prods) or similar
                # Better: look ahead? No, just greedy on immediate addition
                # If we add an edge, total sums becomes len(selected_sums) + new_sums
                # total prods becomes len(selected_prods) + new_prods
                # We want to minimize max(total_sums, total_prods)
                
                current_max = max(len(selected_sums), len(selected_prods))
                new_max = max(len(selected_sums) + new_sums, len(selected_prods) + new_prods)
                
                # Tie-breaking: prefer edges that add fewer total distincts
                tie_breaker = new_sums + new_prods
                
                if new_max < best_score or (new_max == best_score and tie_breaker < best_score):
                    best_score = new_max
                    best_edge = e
            
            if best_edge is None:
                break
                
            selected_edges.append(best_edge)
            selected_sums.add(edge_sum[best_edge])
            selected_prods.add(edge_prod[best_edge])
            
        return selected_edges

    def local_search_edges(values, edges, deadline):
        """
        Local search on edge selection.
        """
        cur_edges = list(edges)
        cur_score = score_of(values, cur_edges)
        best_score = cur_score
        best_edges = list(cur_edges)
        
        # Precompute sums/prods for all edges
        edge_sum = {}
        edge_prod = {}
        for u, v in all_edges:
            edge_sum[(u, v)] = values[u] + values[v]
            edge_prod[(u, v)] = values[u] * values[v]
            
        cur_edges_set = set(cur_edges)
        
        no_improve = 0
        it = 0
        while time.time() < deadline - 10.0:
            it += 1
            # Pick a random edge to remove
            idx = random.randint(0, m - 1)
            e_out = cur_edges[idx]
            
            # Pick a random edge to add (not currently in set)
            # To be efficient, pick from a random sample of all_edges
            e_in = random.choice(all_edges)
            if e_in in cur_edges_set:
                continue
                
            # Perform swap
            new_edges = list(cur_edges)
            new_edges[idx] = e_in
            new_score = score_of(values, new_edges)
            
            if new_score < cur_score:
                cur_edges = new_edges
                cur_edges_set.discard(e_out)
                cur_edges_set.add(e_in)
                cur_score = new_score
                no_improve = 0
                if cur_score < best_score:
                    best_score = cur_score
                    best_edges = list(cur_edges)
            else:
                no_improve += 1
                # Simulated annealing-like acceptance for equal or slightly worse to escape local minima
                if new_score == cur_score and random.random() < 0.1:
                    cur_edges = new_edges
                    cur_edges_set.discard(e_out)
                    cur_edges_set.add(e_in)
                    # cur_score stays same
                elif no_improve > 200 and random.random() < 0.05:
                    cur_edges = new_edges
                    cur_edges_set.discard(e_out)
                    cur_edges_set.add(e_in)
                    cur_score = new_score
                    no_improve = 0
            
            if it % 100 == 0:
                pass # Just a heartbeat
            
        return best_edges, best_score

    def optimize_values(values, edges, deadline):
        """
        Optimize values for a fixed set of edges.
        """
        cur_vals = list(values)
        cur_score = score_of(cur_vals, edges)
        best_score = cur_score
        best_vals = list(cur_vals)
        
        no_improve = 0
        it = 0
        while time.time() < deadline - 10.0:
            it += 1
            idx = random.randint(0, n - 1)
            old_val = cur_vals[idx]
            
            # Try to change value at idx
            # Strategy: try to make sum or product collide with another edge
            # For simplicity, try a few random values and some targeted ones
            
            candidates = set()
            
            # Targeted: try to match sum with another edge
            neighbors = [v for u, v in edges if u == idx or v == idx]
            if neighbors:
                # Pick a random neighbor
                nbr = random.choice(neighbors)
                # Pick a random other edge
                other_e = random.choice(edges)
                ou, ov = other_e
                if idx not in other_e:
                    target_sum = cur_vals[ou] + cur_vals[ov]
                    cand = target_sum - cur_vals[nbr]
                    if 1 <= cand <= 10000 and cand != old_val:
                        candidates.add(cand)
                    
                    target_prod = cur_vals[ou] * cur_vals[ov]
                    if cur_vals[nbr] != 0:
                        cand_p = target_prod // cur_vals[nbr]
                        if cand_p * cur_vals[nbr] == target_prod:
                            if 1 <= cand_p <= 10000 and cand_p != old_val:
                                candidates.add(cand_p)
            
            # Random small perturbations
            for _ in range(5):
                delta = random.randint(-10, 10)
                cand = old_val + delta
                if 1 <= cand <= 10000:
                    candidates.add(cand)
            
            # Random large jump
            cand = random.randint(1, 100)
            candidates.add(cand)
            
            # Filter candidates: must be distinct from other values
            valid_candidates = []
            for c in candidates:
                if c == old_val:
                    continue
                if any(v == c for j, v in enumerate(cur_vals) if j != idx):
                    continue
                valid_candidates.append(c)
                
            if not valid_candidates:
                continue
                
            new_val = random.choice(valid_candidates)
            new_vals = list(cur_vals)
            new_vals[idx] = new_val
            new_score = score_of(new_vals, edges)
            
            if new_score < cur_score:
                cur_vals = new_vals
                cur_score = new_score
                no_improve = 0
                if cur_score < best_score:
                    best_score = cur_score
                    best_vals = list(cur_vals)
            else:
                no_improve += 1
                if no_improve > 100 and random.random() < 0.05:
                    cur_vals = new_vals
                    cur_score = new_score
                    no_improve = 0
        
        return best_vals, best_score

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Initial value sets to try
    init_sets = []
    # 1. 1 to 14
    init_sets.append(list(range(1, 15)))
    # 2. 1 to 13, 16 (skip 14 to change parity/divisibility)
    init_sets.append(list(range(1, 14)) + [16])
    # 3. Highly composite numbers
    init_sets.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    # 4. Arithmetic progression with step 2
    init_sets.append([2*i for i in range(1, 15)])
    # 5. Random small numbers
    for _ in range(5):
        vals = random.sample(range(1, 30), n)
        init_sets.append(vals)
        
    random.shuffle(init_sets)
    
    for vals in init_sets:
        if time.time() > deadline - 20.0:
            break
            
        # 1. Greedy edge selection
        edges = greedy_edge_selection(vals, m, deadline)
        if len(edges) < m:
            # Fill with random edges if greedy failed
            existing = set(edges)
            while len(edges) < m:
                e = random.choice(all_edges)
                if e not in existing:
                    edges.append(e)
                    existing.add(e)
                    
        score = score_of(vals, edges)
        
        # 2. Local search on edges
        edges, score = local_search_edges(vals, edges, deadline)
        
        if score < best_score:
            best_score = score
            best_vals = list(vals)
            best_edges = list(edges)
            
        # 3. Optimize values for these edges
        if time.time() < deadline - 15.0:
            new_vals, new_score = optimize_values(vals, edges, deadline)
            if new_score < best_score:
                best_score = new_score
                best_vals = new_vals
                best_edges = list(edges)
                
                # Re-optimize edges for new values
                if time.time() < deadline - 10.0:
                    edges, score = local_search_edges(best_vals, best_edges, deadline)
                    if score < best_score:
                        best_score = score
                        best_edges = edges

    # Final fallback
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])
        
    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
