import time
import random
from itertools import combinations

_DEADLINE = time.time() + 85

def _score(values, edges):
    """Compute raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _optimize_edges(values, m, n, max_iter=50000):
    """
    Greedy initialization followed by local search to find best m-edge subset.
    Uses incremental updates for speed.
    """
    all_edges = list(combinations(range(n), 2))
    # Precompute sums and products for all edges
    edge_data = {}
    for u, v in all_edges:
        s = values[u] + values[v]
        p = values[u] * values[v]
        edge_data[(u, v)] = (s, p)
    
    # State:
    # current_edges: set of tuples (u, v)
    # sum_counts: dict sum_value -> count
    # prod_counts: dict prod_value -> count
    # distinct_sums: int
    # distinct_prods: int
    
    # Greedy initialization
    current_edges = set()
    sum_counts = {}
    prod_counts = {}
    distinct_sums = 0
    distinct_prods = 0
    
    # Sort edges by some heuristic? Random order with greedy addition is good.
    # To make it deterministic per seed, we shuffle.
    shuffled_edges = all_edges[:]
    random.shuffle(shuffled_edges)
    
    for e in shuffled_edges:
        if len(current_edges) >= m:
            break
        u, v = e
        s, p = edge_data[e]
        
        # Check if adding this edge increases distinct counts
        new_distinct_sums = distinct_sums
        if sum_counts.get(s, 0) == 0:
            new_distinct_sums += 1
            
        new_distinct_prods = distinct_prods
        if prod_counts.get(p, 0) == 0:
            new_distinct_prods += 1
            
        # We just add it in greedy phase to fill up to m
        # A smarter greedy would pick the one with minimal increase, 
        # but that requires checking all remaining edges. 
        # For now, simple fill, then local search.
        current_edges.add(e)
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        distinct_sums = new_distinct_sums
        distinct_prods = new_distinct_prods
        
    current_score = max(distinct_sums, distinct_prods)
    best_score = current_score
    best_edges = set(current_edges)
    
    # Local Search
    # We will try to swap an edge in the set with an edge not in the set.
    # To speed up, we can pre-calculate potential gains.
    # But simple random swaps with O(1) update is fast enough.
    
    in_edges = list(current_edges)
    out_edges = [e for e in all_edges if e not in current_edges]
    
    if not out_edges:
        return best_score, [list(e) for e in best_edges]
        
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        if current_score <= 2:
            break
            
        # Pick a random edge to remove
        idx_in = random.randint(0, len(in_edges) - 1)
        e_in = in_edges[idx_in]
        s_in, p_in = edge_data[e_in]
        
        # Pick a random edge to add
        idx_out = random.randint(0, len(out_edges) - 1)
        e_out = out_edges[idx_out]
        s_out, p_out = edge_data[e_out]
        
        # Calculate new distinct counts
        # Remove e_in
        ns_sums = distinct_sums
        if sum_counts[s_in] == 1:
            ns_sums -= 1
            
        ns_prods = distinct_prods
        if prod_counts[p_in] == 1:
            ns_prods -= 1
            
        # Add e_out
        if sum_counts.get(s_out, 0) == 0:
            ns_sums += 1
        if prod_counts.get(p_out, 0) == 0:
            ns_prods += 1
            
        new_score = max(ns_sums, ns_prods)
        
        if new_score <= current_score:
            # Perform swap
            # Update counts
            sum_counts[s_in] -= 1
            if sum_counts[s_in] == 0:
                del sum_counts[s_in]
            prod_counts[p_in] -= 1
            if prod_counts[p_in] == 0:
                del prod_counts[p_in]
                
            sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
            prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
            
            distinct_sums = ns_sums
            distinct_prods = ns_prods
            current_score = new_score
            
            # Update sets
            current_edges.discard(e_in)
            current_edges.add(e_out)
            
            # Update in/out lists
            in_edges[idx_in] = e_out
            out_edges[idx_out] = e_in
            
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
                
    return best_score, [list(e) for e in best_edges]

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m
    best_result = None
    
    # Candidate value sets
    candidates = []
    
    # 1. Consecutive integers
    candidates.append(list(range(1, n + 1)))
    
    # 2. Small even numbers
    candidates.append(list(range(2, 2 * n + 1, 2)))
    
    # 3. Highly composite-like small numbers
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    
    # 4. Multiples of small numbers
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48])
    
    # 5. Random small sets
    random.seed(42)
    for _ in range(30):
        vals = sorted(random.sample(range(1, 30), n))
        candidates.append(vals)
        
    # 6. Arithmetic progressions
    for start in range(1, 5):
        for step in range(1, 3):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val:
                candidates.append(vals)
                
    for vals in candidates:
        if time.time() > _DEADLINE:
            break
            
        # Validate
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > max_val:
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
            
        # Run optimization
        # We can run multiple restarts for each value set if time permits
        # For now, one run with high max_iter
        score, edges = _optimize_edges(vals, m, n, max_iter=20000)
        
        if score < best_score:
            best_score = score
            best_result = (vals, edges)
            if best_score <= 2:
                break
                
    if best_result is None:
        # Fallback
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
        
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}
