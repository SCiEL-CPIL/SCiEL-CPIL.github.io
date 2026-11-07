"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

N = 14
M = 35
MAXV = 10000

def _raw_value(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        a = values[i]
        b = values[j]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _greedy_edges_safe(values, m):
    """Pick m edges to maximize sum/product collisions. Returns list of (i,j) tuples."""
    n = len(values)
    all_pairs = list(combinations(range(n), 2))
    chosen = []
    chosen_set = set()
    sums = set()
    prods = set()
    
    # Precompute pair properties
    pair_info = []
    for i, j in all_pairs:
        s = values[i] + values[j]
        p = values[i] * values[j]
        pair_info.append((i, j, s, p))
        
    # Sort candidates initially? No, greedy selection based on current state is better.
    # We will iterate m times.
    for _ in range(m):
        best_candidate = None
        best_key = None
        for i, j, s, p in pair_info:
            pair_key = (i, j)
            if pair_key in chosen_set:
                continue
            # Cost: number of new distinct sums and products
            new_s = 0 if s in sums else 1
            new_p = 0 if p in prods else 1
            # We want to minimize new distinct values.
            # Tie-breaker: prefer edges that are "central" or have lower values?
            # Actually, just minimizing (new_s + new_p) is the primary goal.
            key = (new_s + new_p, s, p)
            if best_key is None or key < best_key:
                best_key = key
                best_candidate = (i, j, s, p)
        
        if best_candidate is None:
            break
            
        i, j, s, p = best_candidate
        chosen.append((i, j))
        chosen_set.add((i, j))
        sums.add(s)
        prods.add(p)
        
    return chosen

def _local_search_safe(values, initial_edges, time_budget):
    """Local search to improve edge selection. Ensures no duplicates."""
    n = len(values)
    all_pairs = list(combinations(range(n), 2))
    edges_set = set(tuple(e) for e in initial_edges)
    cur_edges = [list(e) for e in initial_edges]
    cur_val = _raw_value(values, cur_edges)
    best_edges = [list(e) for e in cur_edges]
    best_val = cur_val
    
    rng = random.Random(42)
    deadline = time.time() + time_budget
    t = 1.0
    cool = 0.999
    
    while time.time() < deadline and t > 0.01:
        # Randomly pick an edge to remove
        if not cur_edges:
            break
        idx_rm = rng.randrange(len(cur_edges))
        e_rm = tuple(cur_edges[idx_rm])
        
        # Randomly pick a non-edge to add
        # To speed up, we can just pick from all_pairs and check
        cand = None
        for _ in range(50):
            c = tuple(rng.choice(all_pairs))
            if c not in edges_set:
                cand = c
                break
        
        if cand is None:
            t *= cool
            continue
            
        # Construct new state
        # Remove e_rm from cur_edges
        new_edges_list = [e for i, e in enumerate(cur_edges) if tuple(e) != e_rm]
        new_edges_list.append(list(cand))
        
        # Verify uniqueness (should be true by construction, but safe)
        if len(set(tuple(e) for e in new_edges_list)) != len(new_edges_list):
            t *= cool
            continue
            
        new_val = _raw_value(values, new_edges_list)
        delta = new_val - cur_val
        
        if delta <= 0 or rng.random() < math.exp(-delta / t):
            cur_edges = new_edges_list
            edges_set.discard(e_rm)
            edges_set.add(cand)
            cur_val = new_val
            
            if cur_val < best_val:
                best_val = cur_val
                best_edges = [list(e) for e in cur_edges]
                
        t *= cool
        
    return best_edges, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    best_result = None
    best_score = float('inf')
    
    # Candidate value sets
    # 1. Small integers 1..14 (high collision probability)
    vals_1 = list(range(1, N + 1))
    # 2. Multiples of 2: 2, 4, ..., 28
    vals_2 = [i * 2 for i in range(1, N + 1)]
    # 3. Powers of 2 (low collision, good for products distinctness? No, we want collisions)
    # 4. Mixed small numbers: 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610 (Fibonacci)
    # Fibonacci has additive properties but maybe not ideal for products.
    # Let's stick to small integers which are known to have many sum/product collisions.
    
    candidate_sets = [vals_1, vals_2]
    
    for vals in candidate_sets:
        if time.time() > deadline - 10:
            break
            
        # Ensure values are valid
        if not (all(isinstance(v, int) and 1 <= v <= MAXV for v in vals) and len(set(vals)) == N):
            continue
            
        # Phase 1: Greedy
        edges_greedy = _greedy_edges_safe(vals, M)
        if len(edges_greedy) < M:
            continue # Should not happen
            
        score_greedy = _raw_value(vals, edges_greedy)
        if score_greedy < best_score:
            best_score = score_greedy
            best_result = {
                'values': [int(v) for v in vals],
                'edges': [[int(i), int(j)] for i, j in edges_greedy]
            }
            
        # Phase 2: Local Search
        if time.time() < deadline - 5:
            budget = min(30.0, max(5.0, deadline - time.time() - 5))
            edges_ls, score_ls = _local_search_safe(vals, edges_greedy, budget)
            if score_ls < best_score:
                best_score = score_ls
                best_result = {
                    'values': [int(v) for v in vals],
                    'edges': [[int(i), int(j)] for i, j in edges_ls]
                }
                
    # Final Fallback if no result found (should not happen)
    if best_result is None:
        vals_fallback = list(range(1, N + 1))
        edges_fallback = _greedy_edges_safe(vals_fallback, M)
        best_result = {
            'values': [int(v) for v in vals_fallback],
            'edges': [[int(i), int(j)] for i, j in edges_fallback]
        }
        
    # Final Validation and Cleanup
    # 1. Ensure exactly M edges
    # 2. Ensure no duplicates
    # 3. Ensure indices are valid
    
    vals_out = best_result['values']
    edges_out = best_result['edges']
    
    # Convert to tuples for set operations
    edge_tuples = [tuple(e) for e in edges_out]
    
    # Remove duplicates
    unique_edges = []
    seen = set()
    for e in edge_tuples:
        # Normalize order (i, j) with i < j
        i, j = e
        if i > j:
            i, j = j, i
        key = (i, j)
        if key not in seen:
            seen.add(key)
            unique_edges.append([i, j])
            
    # If we lost edges due to duplicates, try to add valid ones from all_pairs
    if len(unique_edges) < M:
        all_pairs = list(combinations(range(N), 2))
        for p in all_pairs:
            if len(unique_edges) >= M:
                break
            if p not in seen:
                unique_edges.append(list(p))
                seen.add(p)
                
    # If we still don't have M, it's a critical error, but we take what we have.
    # However, the contract says exactly 35 edges.
    # We truncate to M if we somehow have more (should not happen)
    if len(unique_edges) > M:
        unique_edges = unique_edges[:M]
        
    # Final check: if we don't have M edges, the result is invalid.
    # But given N=14, total pairs = 91. We need 35. It's very unlikely to fail to find 35 unique ones.
    
    # Ensure types are Python int
    final_values = [int(v) for v in vals_out]
    final_edges = [[int(i), int(j)] for i, j in unique_edges]
    
    return {'values': final_values, 'edges': final_edges}

# EVOLVE-BLOCK-END
