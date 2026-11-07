"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import Counter

P = {'n': 14, 'm': 35, 'maximum': 10000}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = P['n']
    m = P['m']
    max_val = P['maximum']
    
    all_pairs = list(itertools.combinations(range(n), 2))
    all_pairs_set = set(frozenset(p) for p in all_pairs)
    
    def get_score(vals, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        return max(len(sums), len(prods))
    
    def compute_pair_data(vals):
        """Compute sums and products for all pairs."""
        sums = []
        prods = []
        for i, j in all_pairs:
            sums.append(vals[i] + vals[j])
            prods.append(vals[i] * vals[j])
        return sums, prods
    
    def find_best_edges(vals):
        """Find the m edges that minimize max(distinct_sums, distinct_prods)."""
        sums, prods = compute_pair_data(vals)
        
        sum_counts = Counter(sums)
        prod_counts = Counter(prods)
        
        # Score each pair by collision potential
        scored = []
        for k, (i, j) in enumerate(all_pairs):
            s = sums[k]
            p = prods[k]
            # Higher score = more collisions in complete graph
            score = sum_counts[s] * 3 + prod_counts[p] * 2
            scored.append((score, s, p, i, j))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Greedy selection with incremental tracking
        selected = []
        selected_set = set()
        seen_sums = set()
        seen_prods = set()
        
        for score, s, p, i, j in scored:
            if len(selected) >= m:
                break
            key = frozenset((i, j))
            if key in selected_set:
                continue
            
            selected.append([i, j])
            selected_set.add(key)
            seen_sums.add(s)
            seen_prods.add(p)
        
        return selected
    
    def optimize_edges_for_values(vals):
        """Refine edges using local search after initial greedy selection."""
        current_edges = find_best_edges(vals)
        current_score = get_score(vals, current_edges)
        
        edge_set = set(frozenset(e) for e in current_edges)
        non_edges = [p for p in all_pairs if frozenset(p) not in edge_set]
        
        if not non_edges:
            return current_edges, current_score
        
        improved = True
        max_iters = 50
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if not improved:
                break
            improved = False
            
            for _ in range(10):
                e_idx = random.randint(0, m - 1)
                new_edge = random.choice(non_edges)
                
                temp_edges = current_edges[:]
                temp_edges[e_idx] = list(new_edge)
                
                temp_score = get_score(vals, temp_edges)
                
                if temp_score < current_score:
                    current_edges = temp_edges
                    current_score = temp_score
                    improved = True
                    
                    # Update edge sets
                    edge_set = set(frozenset(e) for e in current_edges)
                    non_edges = [p for p in all_pairs if frozenset(p) not in edge_set]
                    break
        
        return current_edges, current_score
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Generate diverse candidate pools
    small_pool = list(range(1, 31))
    medium_pool = list(range(1, 80))
    factor_rich = [v for v in range(1, 200) if sum(1 for i in range(1, int(v**0.5)+1) if v % i == 0) >= 4]
    if len(factor_rich) < 50:
        factor_rich = list(range(1, 100))
    
    pools = [small_pool, medium_pool, factor_rich, list(range(1, 150))]
    
    num_restarts = 25
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Choose a pool and initialize
        pool = pools[restart % len(pools)]
        pool_copy = pool[:]
        random.shuffle(pool_copy)
        
        if len(pool_copy) >= n:
            current_values = pool_copy[:n]
        else:
            # Fill in from a larger range
            remaining = [v for v in range(1, 200) if v not in pool_copy]
            random.shuffle(remaining)
            current_values = pool_copy + remaining[:n - len(pool_copy)]
        
        current_values = sorted(set(current_values))[:n]
        if len(current_values) < n:
            # Ensure we have exactly n distinct values
            for v in range(1, 200):
                if v not in current_values:
                    current_values.append(v)
                if len(current_values) == n:
                    break
            current_values = sorted(current_values)
        
        # Find best edges
        current_edges = find_best_edges(current_values)
        current_score = get_score(current_values, current_edges)
        
        if current_score < best_score:
            best_score = current_score
            best_values = current_values[:]
            best_edges = [e[:] for e in current_edges]
        
        # Local search on values
        max_val_iters = 200
        no_improve_count = 0
        for _ in range(max_val_iters):
            if time.time() > deadline:
                break
            if no_improve_count > 50:
                break
            
            # Try replacing one value
            idx = random.randint(0, n - 1)
            old_val = current_values[idx]
            
            # Try several replacements
            found_improvement = False
            for _ in range(20):
                # Sample from a biased distribution favoring small numbers
                if random.random() < 0.7:
                    new_val = random.randint(1, 50)
                else:
                    new_val = random.randint(1, 200)
                
                if new_val == old_val or new_val in current_values:
                    continue
                if new_val < 1 or new_val > max_val:
                    continue
                
                temp_vals = current_values[:]
                temp_vals[idx] = new_val
                temp_vals = sorted(set(temp_vals))
                
                if len(temp_vals) != n:
                    continue
                
                # Re-find best edges
                temp_edges = find_best_edges(temp_vals)
                temp_score = get_score(temp_vals, temp_edges)
                
                if temp_score < current_score:
                    current_values = temp_vals
                    current_edges = temp_edges
                    current_score = temp_score
                    found_improvement = True
                    no_improve_count = 0
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_values = current_values[:]
                        best_edges = [e[:] for e in current_edges]
                    break
            
            if not found_improvement:
                no_improve_count += 1
            
            # Occasionally try edge refinement
            if random.random() < 0.3:
                refined_edges, refined_score = optimize_edges_for_values(current_values)
                if refined_score < current_score:
                    current_edges = refined_edges
                    current_score = refined_score
                    if current_score < best_score:
                        best_score = current_score
                        best_values = current_values[:]
                        best_edges = [e[:] for e in current_edges]
    
    # Final polish: optimize edges for the best values found
    if best_values is not None:
        polished_edges, polished_score = optimize_edges_for_values(best_values)
        if polished_score < best_score:
            best_score = polished_score
            best_edges = [e[:] for e in polished_edges]
    
    # Final type safety
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = find_best_edges(best_values)
    
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
