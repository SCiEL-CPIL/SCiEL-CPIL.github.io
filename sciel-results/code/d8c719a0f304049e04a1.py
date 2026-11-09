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
    """Greedy edge selection with joint frequency-based priority."""
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
    
    # Sort by joint frequency (higher first)
    remaining_edges.sort(key=lambda e: (sum_counts[edge_data[e][0]] + prod_counts[edge_data[e][1]]), reverse=True)
    
    while len(selected_edges) < m and remaining_edges:
        best_edge = None
        best_score_candidate = float('inf')
        
        # Check top candidates
        for e in remaining_edges[:150]:
            s, p = edge_data[e]
            new_sums = len(sums) + (1 if s not in sums else 0)
            new_prods = len(prods) + (1 if p not in prods else 0)
            score = max(new_sums, new_prods)
            
            if score < best_score_candidate:
                best_score_candidate = score
                best_edge = e
                if best_score_candidate == len(sums) and best_score_candidate == len(prods):
                    break
                
        if best_edge is None:
            best_edge = remaining_edges[0]
        
        selected_edges.append(best_edge)
        remaining_edges.remove(best_edge)
        s, p = edge_data[best_edge]
        sums.add(s)
        prods.add(p)
    
    return selected_edges, sums, prods

def _local_search(selected_set, all_edges, edge_data, deadline, max_passes=50):
    """Enhanced local search with full swap evaluation and frequency-aware ranking."""
    # Build counts
    s_counts = defaultdict(int)
    p_counts = defaultdict(int)
    for e in selected_set:
        s, p = edge_data[e]
        s_counts[s] += 1
        p_counts[p] += 1
    
    distinct_s = len(s_counts)
    distinct_p = len(p_counts)
    current_score = max(distinct_s, distinct_p)
    
    # Pre-sort non-selected edges by their sum/prod frequency (higher = better candidate)
    non_selected_list = [e for e in all_edges if e not in selected_set]
    
    # Build global frequency for ranking
    global_s_freq = defaultdict(int)
    global_p_freq = defaultdict(int)
    for e in all_edges:
        s, p = edge_data[e]
        global_s_freq[s] += 1
        global_p_freq[p] += 1
    
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
            
            # Rank candidates by how well their sum/prod align with current counts
            def swap_potential(e_in):
                s_in, p_in = edge_data[e_in]
                potential = 0
                if s_in not in s_counts:
                    potential += 2  # Higher penalty for new sum
                if p_in not in p_counts:
                    potential += 2  # Higher penalty for new product
                # Bonus if sum/prod already frequent
                if s_in in s_counts:
                    potential -= s_counts[s_in]
                if p_in in p_counts:
                    potential -= p_counts[p_in]
                return potential
            
            # Sort and check top candidates
            candidate_pool = sorted(non_selected_list, key=swap_potential)[:120]
            
            for e_in in candidate_pool:
                s_in, p_in = edge_data[e_in]
                
                new_distinct_s = cur_distinct_s + (1 if s_in not in s_counts else 0)
                new_distinct_p = cur_distinct_p + (1 if p_in not in p_counts else 0)
                new_score = max(new_distinct_s, new_distinct_p)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_e_in = e_in
                    if best_new_score <= cur_score:
                        break
                    
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

def _polish_pass(selected_set, all_edges, edge_data, deadline):
    """Final high-precision polish: try all single-edge swaps with exact evaluation."""
    s_counts = defaultdict(int)
    p_counts = defaultdict(int)
    for e in selected_set:
        s, p = edge_data[e]
        s_counts[s] += 1
        p_counts[p] += 1
    
    current_score = max(len(s_counts), len(p_counts))
    non_selected_list = [e for e in all_edges if e not in selected_set]
    
    # Try multiple passes of exhaustive single-swap
    for _ in range(5):
        if time.time() > deadline:
            break
        
        improved = False
        selected_list = list(selected_set)
        random.shuffle(selected_list)
        
        for e_out in selected_list:
            if time.time() > deadline:
                break
            
            s_out, p_out = edge_data[e_out]
            
            s_counts[s_out] -= 1
            if s_counts[s_out] == 0:
                del s_counts[s_out]
            p_counts[p_out] -= 1
            if p_counts[p_out] == 0:
                del p_counts[p_out]
            
            cur_distinct_s = len(s_counts)
            cur_distinct_p = len(p_counts)
            cur_score = max(cur_distinct_s, cur_distinct_p)
            
            best_e_in = None
            best_new_score = current_score
            
            for e_in in non_selected_list:
                s_in, p_in = edge_data[e_in]
                new_distinct_s = cur_distinct_s + (1 if s_in not in s_counts else 0)
                new_distinct_p = cur_distinct_p + (1 if p_in not in p_counts else 0)
                new_score = max(new_distinct_s, new_distinct_p)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_e_in = e_in
            
            if best_e_in is not None and best_new_score < current_score:
                selected_set.discard(e_out)
                selected_set.add(best_e_in)
                
                s_counts[s_out] = s_counts.get(s_out, 0) + 1
                p_counts[p_out] = p_counts.get(p_out, 0) + 1
                
                s_in, p_in = edge_data[best_e_in]
                s_counts[s_in] = s_counts.get(s_in, 0) + 1
                p_counts[p_in] = p_counts.get(p_in, 0) + 1
                
                non_selected_list.remove(best_e_in)
                current_score = best_new_score
                improved = True
                break
            else:
                s_counts[s_out] = s_counts.get(s_out, 0) + 1
                p_counts[p_out] = p_counts.get(p_out, 0) + 1
        
        if not improved:
            break
    
    final_score = max(len(s_counts), len(p_counts))
    return selected_set, final_score

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin before 90s kill
    
    n = 14
    m = 35
    max_val = 10000
    
    # Value sets optimized for additive/multiplicative collisions
    # Focus on highly composite numbers and divisor structures
    candidates = [
        # Small consecutive - many sum collisions
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        # Powers and multiples - product collisions
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        # Mix of small numbers with shared factors
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        # Divisor-structured: divisors of 720
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24],
        # Highly composite focused
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        # Mix for both sums and products
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 30],
        # Even numbers - product structure
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        # Small with gaps
        [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
        # Multiplicative chain
        [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26],
        # Balanced small numbers
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18],
        # Another divisor set
        [1, 2, 3, 4, 6, 9, 12, 16, 18, 24, 36, 48, 64, 72],
        # Random small
    ]
    
    random.seed(42)
    for _ in range(10):
        vals = sorted(random.sample(range(1, 50), 14))
        candidates.append(vals)
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for v in candidates:
        if len(set(v)) != n or min(v) < 1 or max(v) > max_val:
            continue
        key = tuple(sorted(v))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(list(v))
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    best_selected_set = None
    
    all_edges = list(combinations(range(n), 2))
    
    # Time per candidate: distribute evenly
    time_per_candidate = (deadline - start_time) / (len(unique_candidates) + 1)
    
    for vals in unique_candidates:
        if time.time() > deadline:
            break
        
        # Precompute
        edge_data = _precompute_edge_data(vals, all_edges)
        
        # Greedy initialization
        selected_edges, sums, prods = _greedy_edge_selection(vals, all_edges, m, edge_data)
        current_score = max(len(sums), len(prods))
        
        # Local search
        selected_set = set(selected_edges)
        selected_set, final_score = _local_search(selected_set, all_edges, edge_data, deadline)
        
        if final_score < best_score:
            best_score = final_score
            best_values = vals
            best_selected_set = set(selected_set)
            best_edges = list(selected_set)
    
    # Final polish on best result
    if best_selected_set is not None and time.time() < deadline - 5:
        # Rebuild edge_data for best values
        edge_data = _precompute_edge_data(best_values, all_edges)
        best_selected_set, final_score = _polish_pass(best_selected_set, all_edges, edge_data, deadline)
        if final_score < best_score:
            best_score = final_score
            best_edges = list(best_selected_set)
    
    # Fallback
    if best_values is None:
        best_values = list(range(1, 15))
        best_edges = all_edges[:m]
    
    # Ensure types
    values = [int(v) for v in best_values]
    edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': values, 'edges': edges}
