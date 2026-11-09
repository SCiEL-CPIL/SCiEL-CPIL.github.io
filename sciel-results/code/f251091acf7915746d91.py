import time
import random
from itertools import combinations
from collections import defaultdict

def _precompute_edge_data(vals, all_edges):
    """Precompute sums and products for all edges."""
    edge_data = {}
    for u, v in all_edges:
        s = vals[u] + vals[v]
        p = vals[u] * vals[v]
        edge_data[(u, v)] = (s, p)
    return edge_data

def _greedy_edge_selection(vals, all_edges, m, edge_data):
    """Greedy edge selection with frequency-based priority."""
    selected_edges = []
    sums = set()
    prods = set()
    remaining_edges = list(all_edges)
    
    # Count frequencies
    sum_counts = defaultdict(int)
    prod_counts = defaultdict(int)
    for e in all_edges:
        s, p = edge_data[e]
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    # Sort by frequency (higher first)
    remaining_edges.sort(key=lambda e: (sum_counts[edge_data[e][0]], prod_counts[edge_data[e][1]]), reverse=True)
    
    while len(selected_edges) < m and remaining_edges:
        best_edge = None
        best_score_candidate = float('inf')
        
        # Check top candidates
        for e in remaining_edges[:100]:
            s, p = edge_data[e]
            new_sums = len(sums) + (1 if s not in sums else 0)
            new_prods = len(prods) + (1 if p not in prods else 0)
            score = max(new_sums, new_prods)
            
            if score < best_score_candidate:
                best_score_candidate = score
                best_edge = e
                
        if best_edge is None:
            best_edge = remaining_edges[0]
        
        selected_edges.append(best_edge)
        remaining_edges.remove(best_edge)
        s, p = edge_data[best_edge]
        sums.add(s)
        prods.add(p)
    
    return selected_edges, sums, prods

def _local_search(selected_set, all_edges, edge_data, current_score, deadline, max_passes=30):
    """Enhanced local search with efficient swap candidates."""
    selected_list = list(selected_set)
    non_selected_list = [e for e in all_edges if e not in selected_set]
    
    # Pre-sort non-selected edges by their potential (lower is better for reducing distinct counts)
    # We'll use a dynamic scoring based on current state
    
    s_counts = defaultdict(int)
    p_counts = defaultdict(int)
    for e in selected_set:
        s, p = edge_data[e]
        s_counts[s] += 1
        p_counts[p] += 1
    
    distinct_s = len(s_counts)
    distinct_p = len(p_counts)
    current_score = max(distinct_s, distinct_p)
    
    improved = True
    pass_count = 0
    
    while improved and time.time() < deadline and pass_count < max_passes:
        improved = False
        pass_count += 1
        
        # Rebuild lists
        selected_list = list(selected_set)
        random.shuffle(selected_list)
        non_selected_list = [e for e in all_edges if e not in selected_set]
        
        # Rebuild counts
        s_counts = defaultdict(int)
        p_counts = defaultdict(int)
        for e in selected_set:
            s, p = edge_data[e]
            s_counts[s] += 1
            p_counts[p] += 1
        
        distinct_s = len(s_counts)
        distinct_p = len(p_counts)
        current_score = max(distinct_s, distinct_p)
        
        for e_out in selected_list:
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
            
            # Sort non-selected by potential: prefer edges whose sum/prod are already in counts
            # This is a heuristic - edges with existing sums/prods are more likely to not increase distinct counts
            def swap_potential(e_in):
                s_in, p_in = edge_data[e_in]
                potential = 0
                if s_in not in s_counts:
                    potential += 1
                if p_in not in p_counts:
                    potential += 1
                return potential
            
            # Check top 80 candidates sorted by potential
            candidate_pool = sorted(non_selected_list[:200], key=swap_potential)[:80]
            
            for e_in in candidate_pool:
                s_in, p_in = edge_data[e_in]
                
                new_distinct_s = cur_distinct_s + (1 if s_in not in s_counts else 0)
                new_distinct_p = cur_distinct_p + (1 if p_in not in p_counts else 0)
                new_score = max(new_distinct_s, new_distinct_p)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_e_in = e_in
                    
            # If we found a better edge
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
                break 
            else:
                # Restore counts
                s_counts[s_out] = s_counts.get(s_out, 0) + 1
                p_counts[p_out] = p_counts.get(p_out, 0) + 1
    
    final_score = max(len(s_counts), len(p_counts))
    return selected_set, final_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    # Candidate value sets optimized for additive/multiplicative collisions
    v1 = list(range(1, 15))
    v2 = [2 * i for i in range(1, 15)]
    v3 = [3 * i for i in range(1, 15)]
    v4 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    v5 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    v6 = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    v7 = [1, 2, 3, 4, 6, 9, 12, 16, 18, 24, 36, 48, 64, 72]
    v8 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    v9 = [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 30]
    v10 = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192]
    
    # Additional structured candidates
    v11 = [1, 2, 3, 4, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072]  # Powers of 2 times 3
    v12 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]  # Even numbers
    v13 = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]  # Odd numbers
    v14 = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]  # Mix
    
    random.seed(42)
    v15 = sorted(random.sample(range(1, 60), 14))
    v16 = sorted(random.sample(range(1, 100), 14))
    
    candidates = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16]
    
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
        
        # Precompute
        edge_data = _precompute_edge_data(vals, all_edges)
        
        # Greedy initialization
        selected_edges, sums, prods = _greedy_edge_selection(vals, all_edges, m, edge_data)
        
        current_score = max(len(sums), len(prods))
        
        # Local search
        selected_set = set(selected_edges)
        selected_set, final_score = _local_search(selected_set, all_edges, edge_data, current_score, deadline)
        
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
