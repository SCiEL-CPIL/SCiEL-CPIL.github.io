import time
import random
from itertools import combinations
from collections import defaultdict
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    maximum = 10000
    
    # Curated candidate value sets designed to have many repeated sums/products
    candidates = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 36, 48],
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20],
        [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28],
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 19, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 18, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 18, 20],
    ]
    
    # Filter valid candidates
    valid_candidates = []
    seen = set()
    for v in candidates:
        if len(set(v)) != n:
            continue
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            continue
        if not all(1 <= x <= maximum for x in v):
            continue
        key = tuple(sorted(v))
        if key in seen:
            continue
        seen.add(key)
        valid_candidates.append(v)
    
    if not valid_candidates:
        valid_candidates = [list(range(1, 15))]
    
    best_score = float('inf')
    best_edges = None
    best_vals = None
    
    all_possible_edges = [list(e) for e in combinations(range(n), 2)]
    num_edges_total = len(all_possible_edges)  # 91
    
    random.seed(42)
    
    for vals in valid_candidates:
        if time.time() > deadline - 20.0:
            break
            
        # Precompute sums and products for all edges
        edge_sums = [0] * num_edges_total
        edge_prods = [0] * num_edges_total
        for idx, (i, j) in enumerate(all_possible_edges):
            edge_sums[idx] = vals[i] + vals[j]
            edge_prods[idx] = vals[i] * vals[j]
        
        # Count frequencies
        sum_freq = defaultdict(int)
        prod_freq = defaultdict(int)
        for s in edge_sums:
            sum_freq[s] += 1
        for p in edge_prods:
            prod_freq[p] += 1
        
        # Sort sum values by frequency (most frequent first)
        sorted_sum_vals = sorted(sum_freq.keys(), key=lambda x: -sum_freq[x])
        sorted_prod_vals = sorted(prod_freq.keys(), key=lambda x: -prod_freq[x])
        
        # Try different K values from small to large
        found_solution = False
        for K in range(5, 30):
            if time.time() > deadline - 15.0:
                break
            
            if K > best_score:
                break  # Can't improve
            
            # Get top K most frequent sums and products
            top_sums = set(sorted_sum_vals[:K])
            top_prods = set(sorted_prod_vals[:K])
            
            # Find all edges whose sum is in top_sums AND product is in top_prods
            compatible_edges = []
            for idx in range(num_edges_total):
                if edge_sums[idx] in top_sums and edge_prods[idx] in top_prods:
                    compatible_edges.append(idx)
            
            if len(compatible_edges) < m:
                continue
            
            # We need to select m edges from compatible_edges
            # Use greedy: pick edges that maximize coverage of sum/product values
            # Actually, since all edges are already in top K, we just need any m of them
            # But we want to minimize the ACTUAL distinct sums and products used
            
            # Greedy: build set of selected edges, tracking which sums/prods are used
            # Start with edges that have high frequency sums and products
            edge_priority = []
            for idx in compatible_edges:
                s_score = sum_freq[edge_sums[idx]]
                p_score = prod_freq[edge_prods[idx]]
                edge_priority.append((s_score * p_score, s_score, p_score, idx))
            
            edge_priority.sort(reverse=True)
            
            selected = []
            sum_counts = defaultdict(int)
            prod_counts = defaultdict(int)
            
            for _, _, _, idx in edge_priority:
                if len(selected) >= m:
                    break
                selected.append(idx)
                sum_counts[edge_sums[idx]] += 1
                prod_counts[edge_prods[idx]] += 1
            
            if len(selected) < m:
                continue
            
            current_score = max(len(sum_counts), len(prod_counts))
            
            if current_score >= best_score:
                continue
            
            # Local search to improve
            improved = True
            max_iters = 100
            iter_count = 0
            
            while improved and iter_count < max_iters and time.time() < deadline - 15.0:
                improved = False
                iter_count += 1
                
                # Try random swaps
                for _ in range(50):
                    if time.time() > deadline - 15.0:
                        break
                    
                    out_pos = random.randint(0, len(selected) - 1)
                    idx_out = selected[out_pos]
                    
                    selected_set = set(selected)
                    
                    s_out = edge_sums[idx_out]
                    p_out = edge_prods[idx_out]
                    
                    # Find candidate edges that don't add new sums/prods
                    candidates_in = []
                    for idx in range(num_edges_total):
                        if idx in selected_set:
                            continue
                        s_in = edge_sums[idx]
                        p_in = edge_prods[idx]
                        adds_sum = 1 if s_in not in sum_counts else 0
                        adds_prod = 1 if p_in not in prod_counts else 0
                        frees_sum = 1 if sum_counts[s_out] == 1 else 0
                        frees_prod = 1 if prod_counts[p_out] == 1 else 0
                        net_change = (adds_sum - frees_sum) + (adds_prod - frees_prod)
                        if net_change <= 0:
                            candidates_in.append((net_change, sum_freq[s_in], prod_freq[p_in], idx))
                    
                    if not candidates_in:
                        continue
                    
                    candidates_in.sort()
                    _, _, _, idx_in = candidates_in[0]
                    
                    s_in = edge_sums[idx_in]
                    p_in = edge_prods[idx_in]
                    
                    # Update counts
                    sum_counts[s_out] -= 1
                    if sum_counts[s_out] == 0:
                        del sum_counts[s_out]
                    prod_counts[p_out] -= 1
                    if prod_counts[p_out] == 0:
                        del prod_counts[p_out]
                    
                    sum_counts[s_in] += 1
                    prod_counts[p_in] += 1
                    
                    new_ds = len(sum_counts)
                    new_dp = len(prod_counts)
                    new_score = max(new_ds, new_dp)
                    
                    if new_score < current_score:
                        selected[out_pos] = idx_in
                        current_score = new_score
                        improved = True
                        break
                    else:
                        # Revert
                        sum_counts[s_in] -= 1
                        if sum_counts[s_in] == 0:
                            del sum_counts[s_in]
                        prod_counts[p_in] -= 1
                        if prod_counts[p_in] == 0:
                            del prod_counts[p_in]
                        
                        sum_counts[s_out] += 1
                        prod_counts[p_out] += 1
            
            if current_score < best_score:
                best_score = current_score
                best_edges = [all_possible_edges[i] for i in selected]
                best_vals = vals
                found_solution = True
                
                # If we achieved a very good score, we can stop early
                if best_score <= 7:
                    break
        
        # Also try pure greedy without K constraint
        if time.time() < deadline - 15.0:
            edge_scores = []
            for idx in range(num_edges_total):
                s_score = sum_freq[edge_sums[idx]]
                p_score = prod_freq[edge_prods[idx]]
                combined = min(s_score, p_score)
                edge_scores.append((combined, s_score, p_score, idx))
            
            edge_scores.sort(reverse=True)
            
            selected = []
            sum_counts = defaultdict(int)
            prod_counts = defaultdict(int)
            
            for _, _, _, idx in edge_scores:
                if len(selected) >= m:
                    break
                selected.append(idx)
                sum_counts[edge_sums[idx]] += 1
                prod_counts[edge_prods[idx]] += 1
            
            current_score = max(len(sum_counts), len(prod_counts))
            
            # Local search
            improved = True
            max_iters = 100
            iter_count = 0
            
            while improved and iter_count < max_iters and time.time() < deadline - 15.0:
                improved = False
                iter_count += 1
                
                for _ in range(50):
                    if time.time() > deadline - 15.0:
                        break
                    
                    out_pos = random.randint(0, len(selected) - 1)
                    idx_out = selected[out_pos]
                    
                    selected_set = set(selected)
                    
                    s_out = edge_sums[idx_out]
                    p_out = edge_prods[idx_out]
                    
                    candidates_in = []
                    for idx in range(num_edges_total):
                        if idx in selected_set:
                            continue
                        s_in = edge_sums[idx]
                        p_in = edge_prods[idx]
                        adds_sum = 1 if s_in not in sum_counts else 0
                        adds_prod = 1 if p_in not in prod_counts else 0
                        frees_sum = 1 if sum_counts[s_out] == 1 else 0
                        frees_prod = 1 if prod_counts[p_out] == 1 else 0
                        net_change = (adds_sum - frees_sum) + (adds_prod - frees_prod)
                        if net_change <= 0:
                            candidates_in.append((net_change, sum_freq[s_in], prod_freq[p_in], idx))
                    
                    if not candidates_in:
                        continue
                    
                    candidates_in.sort()
                    _, _, _, idx_in = candidates_in[0]
                    
                    s_in = edge_sums[idx_in]
                    p_in = edge_prods[idx_in]
                    
                    sum_counts[s_out] -= 1
                    if sum_counts[s_out] == 0:
                        del sum_counts[s_out]
                    prod_counts[p_out] -= 1
                    if prod_counts[p_out] == 0:
                        del prod_counts[p_out]
                    
                    sum_counts[s_in] += 1
                    prod_counts[p_in] += 1
                    
                    new_ds = len(sum_counts)
                    new_dp = len(prod_counts)
                    new_score = max(new_ds, new_dp)
                    
                    if new_score < current_score:
                        selected[out_pos] = idx_in
                        current_score = new_score
                        improved = True
                        break
                    else:
                        sum_counts[s_in] -= 1
                        if sum_counts[s_in] == 0:
                            del sum_counts[s_in]
                        prod_counts[p_in] -= 1
                        if prod_counts[p_in] == 0:
                            del prod_counts[p_in]
                        
                        sum_counts[s_out] += 1
                        prod_counts[p_out] += 1
            
            if current_score < best_score:
                best_score = current_score
                best_edges = [all_possible_edges[i] for i in selected]
                best_vals = vals
    
    # Fallback
    if best_edges is None:
        best_vals = valid_candidates[0]
        best_edges = all_possible_edges[:m]
    
    # Final validation
    assert len(best_vals) == n
    assert len(set(best_vals)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in best_vals)
    assert all(1 <= v <= maximum for v in best_vals)
    assert len(best_edges) == m
    assert len(set(tuple(sorted(e)) for e in best_edges)) == m
    
    return {'values': list(best_vals), 'edges': [list(e) for e in best_edges]}
