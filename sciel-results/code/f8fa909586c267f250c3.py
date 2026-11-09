import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    # Candidate value sets optimized for additive/multiplicative collisions
    # Powers of 2 give many product collisions; small consecutive give sum collisions
    v1 = list(range(1, 15))
    v2 = [2 * i for i in range(1, 15)]
    v3 = [3 * i for i in range(1, 15)]
    v4 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    v5 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    # Highly composite numbers - many factor pairs for product collisions
    v6 = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    # Mix of small numbers with shared factors
    v7 = [1, 2, 3, 4, 6, 9, 12, 16, 18, 24, 36, 48, 64, 72]
    # Geometric progression
    v8 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # Small primes and composites mix
    v9 = [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 30]
    # Another collision-rich set
    v10 = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192]
    # More collision-rich candidates
    v13 = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 30, 32, 40, 48]
    v14 = [1, 2, 3, 4, 6, 8, 10, 12, 16, 18, 20, 24, 30, 36]
    v15 = [1, 2, 4, 6, 8, 10, 12, 16, 18, 20, 24, 28, 30, 32]
    v16 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    v17 = [1, 2, 3, 4, 6, 9, 12, 16, 18, 20, 24, 27, 30, 36]
    v18 = [1, 2, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30]
    
    random.seed(42)
    v11 = sorted(random.sample(range(1, 50), 14))
    v12 = sorted(random.sample(range(1, 80), 14))
    random.seed(123)
    v19 = sorted(random.sample(range(1, 40), 14))
    random.seed(456)
    v20 = sorted(random.sample(range(1, 60), 14))
    
    candidates = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20]
    
    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for v in candidates:
        key = tuple(sorted(v))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(v)
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    all_edges = list(combinations(range(n), 2))
    
    for vals in unique_candidates:
        if time.time() > deadline:
            break
            
        # Validate values
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > max_val:
            continue
        
        # Precompute sums and products for all edges
        edge_data = {}
        for u, v in all_edges:
            s = vals[u] + vals[v]
            p = vals[u] * vals[v]
            edge_data[(u, v)] = (s, p)
        
        # Clustering-based edge initialization: group edges by (sum, product)
        # and prefer edges that share both sum and product with other edges
        sum_counts = {}
        prod_counts = {}
        sp_counts = {}
        for e in all_edges:
            s, p = edge_data[e]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
            sp = (s, p)
            sp_counts[sp] = sp_counts.get(sp, 0) + 1
            
        # Score each edge: prefer high frequency in sum, product, and (sum,product) pair
        def edge_priority(e):
            s, p = edge_data[e]
            sp = (s, p)
            return (sp_counts[sp], sum_counts[s], prod_counts[p])
            
        # Sort all edges by priority
        sorted_edges = sorted(all_edges, key=edge_priority, reverse=True)
        
        # Greedy selection with look-ahead
        selected_set = set()
        s_counts = {}
        p_counts = {}
        
        for e in sorted_edges:
            if len(selected_set) >= m:
                break
            s, p = edge_data[e]
            selected_set.add(e)
            s_counts[s] = s_counts.get(s, 0) + 1
            p_counts[p] = p_counts.get(p, 0) + 1
        
        # If we didn't get enough edges (shouldn't happen), fill randomly
        while len(selected_set) < m:
            e = random.choice(all_edges)
            if e not in selected_set:
                selected_set.add(e)
                s, p = edge_data[e]
                s_counts[s] = s_counts.get(s, 0) + 1
                p_counts[p] = p_counts.get(p, 0) + 1
        
        current_score = max(len(s_counts), len(p_counts))
        
        # Precompute non-selected edges for faster swap search
        non_selected_list = [e for e in all_edges if e not in selected_set]
        
        # Local search: try swaps to reduce max(distinct_sums, distinct_prods)
        improved = True
        pass_count = 0
        max_passes = 30
        while improved and time.time() < deadline and pass_count < max_passes:
            improved = False
            pass_count += 1
            
            edge_list = list(selected_set)
            random.shuffle(edge_list)
            
            # Rebuild non-selected list
            non_selected_list = [e for e in all_edges if e not in selected_set]
            random.shuffle(non_selected_list)
            
            for e_out in edge_list:
                if time.time() > deadline:
                    break
                    
                s_out, p_out = edge_data[e_out]
                
                # Temporarily remove e_out
                s_counts[s_out] -= 1
                if s_counts[s_out] == 0:
                    del s_counts[s_out]
                p_counts[p_out] -= 1
                if p_counts[p_out] == 0:
                    del p_counts[p_out]
                
                cur_distinct_s = len(s_counts)
                cur_distinct_p = len(p_counts)
                cur_score = max(cur_distinct_s, cur_distinct_p)
                
                # Find best e_in from non-selected edges
                best_e_in = None
                best_new_score = cur_score + 1
                
                for e_in in non_selected_list:
                    s_in, p_in = edge_data[e_in]
                    
                    new_distinct_s = cur_distinct_s + (1 if s_in not in s_counts else 0)
                    new_distinct_p = cur_distinct_p + (1 if p_in not in p_counts else 0)
                    new_score = max(new_distinct_s, new_distinct_p)
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_e_in = e_in
                    # Early termination: if we can't beat current score
                    if best_new_score == cur_score and cur_distinct_s <= cur_distinct_p:
                        break
                    
                if best_e_in is not None and best_new_score < current_score:
                    # Perform swap
                    selected_set.discard(e_out)
                    selected_set.add(best_e_in)
                    
                    # Update counts
                    s_counts[s_out] = s_counts.get(s_out, 0) + 1
                    p_counts[p_out] = p_counts.get(p_out, 0) + 1
                    
                    s_in, p_in = edge_data[best_e_in]
                    s_counts[s_in] = s_counts.get(s_in, 0) + 1
                    p_counts[p_in] = p_counts.get(p_in, 0) + 1
                    
                    current_score = best_new_score
                    improved = True
                    # Update non_selected_list
                    non_selected_list.remove(best_e_in)
                    non_selected_list.append(e_out)
                    random.shuffle(non_selected_list)
                    break 
                else:
                    # Restore counts
                    s_counts[s_out] = s_counts.get(s_out, 0) + 1
                    p_counts[p_out] = p_counts.get(p_out, 0) + 1
                    
        final_score = max(len(s_counts), len(p_counts))
        
        if final_score < best_score:
            best_score = final_score
            best_values = vals
            best_edges = list(selected_set)
            
    # Fallback
    if best_values is None:
        best_values = v1
        best_edges = all_edges[:m]
        
    # Ensure types
    values = [int(v) for v in best_values]
    edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': values, 'edges': edges}
