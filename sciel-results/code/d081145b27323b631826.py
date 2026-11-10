import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    n = 14
    m = 35
    maximum = 10000

    random.seed(42)

    # Precompute all possible edges
    all_possible_edges = [list(e) for e in combinations(range(n), 2)]
    num_edges_total = len(all_possible_edges)  # 91

    # Helper to compute raw value
    def compute_raw(vals, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        return max(len(sums), len(prods))

    # Helper to perform local search with incremental updates
    def local_search(vals, initial_edges):
        # Precompute sums and products for all possible edges
        edge_sums = [0] * num_edges_total
        edge_prods = [0] * num_edges_total
        for idx, (i, j) in enumerate(all_possible_edges):
            edge_sums[idx] = vals[i] + vals[j]
            edge_prods[idx] = vals[i] * vals[j]

        # Map edge index to (i, j) and vice versa? 
        # We work with indices of all_possible_edges.
        
        # Initial set of edges (indices)
        # Convert initial_edges (list of [i,j]) to indices
        edge_to_idx = {tuple(e): idx for idx, e in enumerate(all_possible_edges)}
        
        current_indices = set()
        for e in initial_edges:
            current_indices.add(edge_to_idx[tuple(e)])
            
        # Initialize counts
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        for idx in current_indices:
            sum_counts[edge_sums[idx]] += 1
            prod_counts[edge_prods[idx]] += 1
            
        distinct_sums = len(sum_counts)
        distinct_prods = len(prod_counts)
        current_score = max(distinct_sums, distinct_prods)
        
        improved = True
        iter_count = 0
        max_iters = 5000
        
        while improved and iter_count < max_iters and time.time() < deadline - 2.0:
            improved = False
            iter_count += 1
            
            # Pick a random edge to remove
            idx_out = random.choice(list(current_indices))
            
            # Pick a random edge to add
            # Try a few random candidates
            found_swap = False
            # To make it faster, we can try to find an edge that shares sum or prod
            # But random is okay for now with high iter count
            for _ in range(30): 
                idx_in = random.randint(0, num_edges_total - 1)
                if idx_in in current_indices:
                    continue
                
                # Simulate swap
                s_out = edge_sums[idx_out]
                p_out = edge_prods[idx_out]
                s_in = edge_sums[idx_in]
                p_in = edge_prods[idx_in]
                
                # Calculate delta distinct sums
                delta_ds = 0
                # Remove s_out
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0:
                    delta_ds -= 1
                    del sum_counts[s_out]
                
                # Add s_in
                # Note: if s_in == s_out, we just removed and re-added? 
                # If s_in == s_out, count went down by 1 then up by 1. 
                # If it became 0, we deleted it. Then adding it back creates it again.
                # Let's handle this carefully.
                
                # Actually, it's safer to calculate new counts in a temporary way or rollback.
                # Let's rollback and do it properly.
                
                # Rollback s_out
                if s_out not in sum_counts:
                    sum_counts[s_out] = 1
                else:
                    sum_counts[s_out] += 1
                
                # Let's just compute the change directly without modifying state until accepted.
                c_s_out = sum_counts.get(s_out, 0)
                c_s_in = sum_counts.get(s_in, 0)
                
                # If we remove s_out:
                # If c_s_out == 1, distinct_sums decreases by 1.
                # If c_s_out > 1, distinct_sums stays same.
                # If we add s_in:
                # If c_s_in == 0, distinct_sums increases by 1.
                # If c_s_in > 0, distinct_sums stays same.
                
                # BUT, if s_out == s_in, the net effect is 0.
                # Let's handle the general case:
                # New count of s_out = c_s_out - 1
                # New count of s_in = c_s_in + 1
                
                new_c_s_out = c_s_out - 1
                new_c_s_in = c_s_in + 1
                
                if s_out == s_in:
                    delta_ds = 0
                else:
                    if new_c_s_out == 0:
                        delta_ds -= 1
                    if new_c_s_in == 1:
                        delta_ds += 1
                        
                new_distinct_sums = distinct_sums + delta_ds
                
                # Same for products
                c_p_out = prod_counts.get(p_out, 0)
                c_p_in = prod_counts.get(p_in, 0)
                
                new_c_p_out = c_p_out - 1
                new_c_p_in = c_p_in + 1
                
                if p_out == p_in:
                    delta_dp = 0
                else:
                    if new_c_p_out == 0:
                        delta_dp -= 1
                    if new_c_p_in == 1:
                        delta_dp += 1
                        
                new_distinct_prods = distinct_prods + delta_dp
                
                new_score = max(new_distinct_sums, new_distinct_prods)
                
                if new_score < current_score:
                    # Accept swap
                    current_indices.remove(idx_out)
                    current_indices.add(idx_in)
                    
                    # Update counts
                    sum_counts[s_out] -= 1
                    if sum_counts[s_out] == 0:
                        del sum_counts[s_out]
                    sum_counts[s_in] += 1
                    
                    prod_counts[p_out] -= 1
                    if prod_counts[p_out] == 0:
                        del prod_counts[p_out]
                    prod_counts[p_in] += 1
                    
                    current_score = new_score
                    distinct_sums = new_distinct_sums
                    distinct_prods = new_distinct_prods
                    improved = True
                    found_swap = True
                    break
            
            if not found_swap:
                # Try a different strategy? Or just break if stuck?
                # If we can't find an improvement in 30 tries for a random edge, 
                # maybe we should try another random edge out.
                # The loop continues, so it will pick a new idx_out.
                pass
                
        # Convert indices back to edges
        final_edges = [all_possible_edges[i] for i in current_indices]
        return current_score, final_edges

    # Helper to perform greedy edge selection
    def greedy_edge_selection(vals):
        # Precompute sums and products
        edge_sums = [0] * num_edges_total
        edge_prods = [0] * num_edges_total
        for idx, (i, j) in enumerate(all_possible_edges):
            edge_sums[idx] = vals[i] + vals[j]
            edge_prods[idx] = vals[i] * vals[j]
            
        used_sums = set()
        used_prods = set()
        selected_indices = []
        remaining_indices = list(range(num_edges_total))
        
        # Sort remaining by some heuristic? 
        # Maybe prefer edges with smaller sums/products? 
        # Or just random shuffle.
        random.shuffle(remaining_indices)
        
        for _ in range(m):
            best_idx = -1
            best_score = float('inf')
            for k, idx in enumerate(remaining_indices):
                s = edge_sums[idx]
                p = edge_prods[idx]
                
                # Calculate new distinct counts
                # If s is new, distinct_sums increases
                # If p is new, distinct_prods increases
                new_sums = len(used_sums) + (1 if s not in used_sums else 0)
                new_prods = len(used_prods) + (1 if p not in used_prods else 0)
                score = max(new_sums, new_prods)
                
                if score < best_score:
                    best_score = score
                    best_idx = k
            
            if best_idx >= 0:
                chosen_idx = remaining_indices[best_idx]
                selected_indices.append(chosen_idx)
                used_sums.add(edge_sums[chosen_idx])
                used_prods.add(edge_prods[chosen_idx])
                remaining_indices.pop(best_idx)
                
        final_edges = [all_possible_edges[i] for i in selected_indices]
        raw = compute_raw(vals, final_edges)
        return raw, final_edges

    best_raw = float('inf')
    best_values = None
    best_edges = None

    # Candidate value sets
    # 1. Small integers 1-14
    # 2. Multiplicative rich set
    # 3. Random small
    # 4. Random medium
    # 5. Structured (AP)
    
    candidate_generators = []
    
    def gen_small():
        return list(range(1, 15))
        
    def gen_rich():
        return [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72]
        
    def gen_random_small():
        return random.sample(range(1, 50), n)
        
    def gen_random_medium():
        return random.sample(range(1, 200), n)
        
    def gen_ap():
        a = random.randint(1, 10)
        d = random.randint(1, 5)
        vals = [a + i*d for i in range(n)]
        if max(vals) > maximum or len(set(vals)) != n:
            return gen_random_small()
        return vals

    candidate_generators.append(gen_small)
    candidate_generators.append(gen_rich)
    candidate_generators.append(gen_random_small)
    candidate_generators.append(gen_random_medium)
    candidate_generators.append(gen_ap)

    attempts = 0
    max_attempts = 50
    
    while time.time() < deadline - 5.0 and attempts < max_attempts:
        attempts += 1
        gen = candidate_generators[attempts % len(candidate_generators)]
        vals = gen()
        
        # Validate
        if len(set(vals)) != n or not all(1 <= v <= maximum for v in vals):
            continue
            
        # Greedy selection
        raw_greedy, edges_greedy = greedy_edge_selection(vals)
        
        # Local search
        raw_ls, edges_ls = local_search(vals, edges_greedy)
        
        if raw_ls < best_raw:
            best_raw = raw_ls
            best_values = vals
            best_edges = edges_ls
            
            # If we reach a very low value, we can stop early
            # Theoretical lower bound? 
            # With 35 edges, we have at least 35 sums (if all distinct) or fewer.
            # If we can get distinct sums <= 20 and distinct prods <= 20, that's good.
            if best_raw <= 25:
                break

    # Fallback
    if best_values is None:
        best_values = list(range(1, 15))
        best_edges = all_possible_edges[:m]
        best_raw = compute_raw(best_values, best_edges)

    # Final validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in best_values)
    assert all(1 <= v <= maximum for v in best_values)
    assert len(best_edges) == m
    assert len(set(tuple(sorted(e)) for e in best_edges)) == m
    
    return {'values': list(best_values), 'edges': [list(e) for e in best_edges]}
