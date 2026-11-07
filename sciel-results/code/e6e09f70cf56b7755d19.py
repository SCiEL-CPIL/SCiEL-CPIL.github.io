import time
import random
from itertools import combinations
from collections import Counter

def _raw_value(values, edges):
    """Compute max(distinct_sums, distinct_products) for given values and edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _optimize_edges_fast(values, m=35, deadline=None):
    """
    Optimizes edge selection using fast local search with Counter-based score updates.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)
    
    # Precompute sums and products for all edges
    edge_sums = [0] * num_all_edges
    edge_prods = [0] * num_all_edges
    for idx, (i, j) in enumerate(all_edges):
        edge_sums[idx] = values[i] + values[j]
        edge_prods[idx] = values[i] * values[j]
    
    best_score = float('inf')
    best_edges = None
    
    num_restarts = 50 # We can afford more restarts now
    
    for restart in range(num_restarts):
        if deadline and time.time() > deadline:
            break
            
        # Random initialization
        current_edges = set(random.sample(range(num_all_edges), m))
        
        # Initialize counters
        sum_counts = Counter()
        prod_counts = Counter()
        for idx in current_edges:
            sum_counts[edge_sums[idx]] += 1
            prod_counts[edge_prods[idx]] += 1
            
        current_distinct_sums = len(sum_counts)
        current_distinct_prods = len(prod_counts)
        current_score = max(current_distinct_sums, current_distinct_prods)
        
        # Local Search
        improved = True
        max_iterations = 100
        iteration = 0
        while improved and iteration < max_iterations and (not deadline or time.time() < deadline):
            improved = False
            iteration += 1
            
            # Shuffle edges to avoid bias
            current_list = list(current_edges)
            outside_indices = [idx for idx in range(num_all_edges) if idx not in current_edges]
            random.shuffle(current_list)
            random.shuffle(outside_indices)
            
            for e_in_idx in current_list:
                s_in = edge_sums[e_in_idx]
                p_in = edge_prods[e_in_idx]
                
                for e_out_idx in outside_indices:
                    s_out = edge_sums[e_out_idx]
                    p_out = edge_prods[e_out_idx]
                    
                    # Simulate swap
                    # Remove e_in
                    sum_counts[s_in] -= 1
                    prod_counts[p_in] -= 1
                    
                    new_distinct_sums = current_distinct_sums
                    new_distinct_prods = current_distinct_prods
                    
                    if sum_counts[s_in] == 0:
                        new_distinct_sums -= 1
                    if prod_counts[p_in] == 0:
                        new_distinct_prods -= 1
                        
                    # Add e_out
                    sum_counts[s_out] += 1
                    prod_counts[p_out] += 1
                    
                    if sum_counts[s_out] == 1:
                        new_distinct_sums += 1
                    if prod_counts[p_out] == 1:
                        new_distinct_prods += 1
                        
                    new_score = max(new_distinct_sums, new_distinct_prods)
                    
                    if new_score < current_score:
                        # Accept swap
                        current_edges.remove(e_in_idx)
                        current_edges.add(e_out_idx)
                        
                        current_distinct_sums = new_distinct_sums
                        current_distinct_prods = new_distinct_prods
                        current_score = new_score
                        improved = True
                        
                        if current_score < best_score:
                            best_score = current_score
                            best_edges = [all_edges[idx] for idx in current_edges]
                        
                        # Break inner loop to restart with new state
                        # We need to update outside_indices? 
                        # e_out is now in, e_in is now out.
                        # For simplicity and correctness in the next iteration, 
                        # we can just break and let the while loop re-evaluate.
                        # But we modified sum_counts/prod_counts, which is correct.
                        break
                if improved:
                    break
                    # Note: We broke the inner loop, but the outer loop (for e_in_idx) 
                    # should also break to restart the scan with the new state.
                    # The 'improved' flag handles the while loop, but we are still 
                    # in the for e_in_idx loop. We need to break that too.
                    # Actually, if we break the inner for, we are still in the outer for.
                    # We should break the outer for as well to restart the full scan.
                    # So we set a flag or just break.
                    # Let's restructure: if improved, break outer for.
            
            # If we broke out of the for e_in_idx loop due to improvement, 
            # the 'improved' flag is True, so the while loop continues.
            # If no improvement was found in the entire scan, improved remains False.
            
        if current_score < best_score:
            best_score = current_score
            best_edges = [all_edges[idx] for idx in current_edges]
            
    return best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    best_raw = float('inf')
    best_result = None
    
    # Candidate value sets
    # Small integers are best for collisions in both sums and products
    value_sets = [
        list(range(1, 15)),
        list(range(2, 16)),
        list(range(3, 17)),
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 36, 48], # Multiplicative structure
        [1, 2, 3, 5, 7, 10, 14, 15, 20, 21, 28, 30, 35, 42], # Divisors of 42/210
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], # Powers of 2 (bad for sums, good for prods)
    ]
    
    for vals in value_sets:
        if deadline and time.time() > deadline:
            break
        
        if len(set(vals)) != 14 or any(v <= 0 or v > 10000 for v in vals):
            continue
            
        edges, raw = _optimize_edges_fast(vals, 35, deadline)
        
        if edges and raw < best_raw:
            best_raw = raw
            best_result = {'values': vals, 'edges': edges}
            
    if best_result is None:
        # Fallback
        vals = list(range(1, 15))
        edges = list(combinations(range(14), 2))[:35]
        best_result = {'values': vals, 'edges': edges}
        
    # Format output
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
