import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    
    n = 14
    m = 35
    
    # Candidate value sets designed for product/sum collisions
    candidate_value_sets = [
        # Powers of 2: products are 2^(a+b), 25 distinct max
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        # Mix of small composites for more collisions
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        # All even numbers with shared factors
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32],
        # Multiples of 6
        [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84],
        # Small values with many factors
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72],
    ]
    
    best_result = None
    best_score = float('inf')
    
    for values in candidate_value_sets:
        # Verify constraints
        if len(values) != n:
            continue
        if len(set(values)) != n:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
        
        all_edges = list(combinations(range(n), 2))
        num_all_edges = len(all_edges)
        
        # Precompute sums and products
        edge_sums = [values[u] + values[v] for u, v in all_edges]
        edge_prods = [values[u] * values[v] for u, v in all_edges]
        
        # Get unique sums and products with mapping
        unique_sums = list(set(edge_sums))
        unique_prods = list(set(edge_prods))
        
        # Greedy edge selection to minimize max(distinct_sums, distinct_prods)
        # Start with empty set, add edges that cause least new distinct values
        selected = []
        sum_count = {}
        prod_count = {}
        distinct_sums = 0
        distinct_prods = 0
        
        for _ in range(m):
            best_idx = -1
            best_new_max = float('inf')
            best_new_ds = 0
            best_new_dp = 0
            
            for idx in range(num_all_edges):
                if idx in selected:
                    continue
                
                s = edge_sums[idx]
                p = edge_prods[idx]
                
                new_ds = distinct_sums + (1 if s not in sum_count else 0)
                new_dp = distinct_prods + (1 if p not in prod_count else 0)
                new_max = max(new_ds, new_dp)
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_idx = idx
                    best_new_ds = new_ds
                    best_new_dp = new_dp
            
            if best_idx == -1:
                break
            
            selected.append(best_idx)
            s = edge_sums[best_idx]
            p = edge_prods[best_idx]
            sum_count[s] = sum_count.get(s, 0) + 1
            prod_count[p] = prod_count.get(p, 0) + 1
            distinct_sums = best_new_ds
            distinct_prods = best_new_dp
        
        current_score = max(distinct_sums, distinct_prods)
        
        # Local search: try swapping one edge for another
        selected_set = set(selected)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_swap = None
            best_swap_score = current_score
            
            for i_out in range(len(selected)):
                idx_out = selected[i_out]
                s_out = edge_sums[idx_out]
                p_out = edge_prods[idx_out]
                
                # Temporarily remove
                sum_count[s_out] -= 1
                prod_count[p_out] -= 1
                ds_after_remove = distinct_sums - (1 if sum_count[s_out] == 0 else 0)
                dp_after_remove = distinct_prods - (1 if prod_count[p_out] == 0 else 0)
                
                for idx_in in range(num_all_edges):
                    if idx_in in selected_set:
                        continue
                    
                    s_in = edge_sums[idx_in]
                    p_in = edge_prods[idx_in]
                    
                    new_ds = ds_after_remove + (1 if sum_count.get(s_in, 0) == 0 else 0)
                    new_dp = dp_after_remove + (1 if prod_count.get(p_in, 0) == 0 else 0)
                    new_max = max(new_ds, new_dp)
                    
                    if new_max < best_swap_score:
                        best_swap_score = new_max
                        best_swap = (i_out, idx_in)
                
                # Restore
                sum_count[s_out] += 1
                prod_count[p_out] += 1
            
            if best_swap:
                i_out, idx_in = best_swap
                idx_out = selected[i_out]
                
                s_out = edge_sums[idx_out]
                p_out = edge_prods[idx_out]
                s_in = edge_sums[idx_in]
                p_in = edge_prods[idx_in]
                
                sum_count[s_out] -= 1
                prod_count[p_out] -= 1
                if sum_count[s_out] == 0:
                    distinct_sums -= 1
                if prod_count[p_out] == 0:
                    distinct_prods -= 1
                
                if s_in not in sum_count or sum_count[s_in] == 0:
                    distinct_sums += 1
                sum_count[s_in] = sum_count.get(s_in, 0) + 1
                
                if p_in not in prod_count or prod_count[p_in] == 0:
                    distinct_prods += 1
                prod_count[p_in] = prod_count.get(p_in, 0) + 1
                
                selected[i_out] = idx_in
                selected_set.remove(idx_out)
                selected_set.add(idx_in)
                
                current_score = max(distinct_sums, distinct_prods)
                improved = True
        
        if current_score < best_score:
            best_score = current_score
            best_result = {
                'values': list(values),
                'edges': [[all_edges[i][0], all_edges[i][1]] for i in selected]
            }
        
        if time.time() > deadline - 5:
            break
    
    if best_result is None:
        # Fallback
        values = list(range(1, n + 1))
        all_edges = list(combinations(range(n), 2))
        best_result = {
            'values': values,
            'edges': [list(e) for e in all_edges[:m]]
        }
    
    # Ensure correct types
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return best_result
