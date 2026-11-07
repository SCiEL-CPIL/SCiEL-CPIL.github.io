"""Search-based program: minimize max(distinct_sums, distinct_products) over 35 edges of 14 vertices."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_score(values, edges):
    """Compute the raw score: max(#distinct sums, #distinct products) over edges."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _all_edges(n):
    """Generate all C(n,2) possible edges."""
    return list(itertools.combinations(range(n), 2))

def _greedy_edge_selection(values, target_m, all_edges):
    """
    Greedy: pick edges that maximize reuse of existing sum/product values.
    At each step, choose the edge whose (sum, product) pair is already seen most often.
    """
    n = len(values)
    edges_chosen = []
    sum_counts = {}
    prod_counts = {}
    
    # Precompute all edge attributes
    edge_info = []
    for u, v in all_edges:
        a, b = values[u], values[v]
        s = a + b
        p = a * b
        edge_info.append((u, v, s, p))
    
    # Sort by a heuristic: prefer edges with "smaller" sums and products initially,
    # but the real greedy is dynamic. Let's do a proper greedy.
    
    for _ in range(target_m):
        best_edge = None
        best_key = None
        for (u, v, s, p) in edge_info:
            # Skip if already chosen
            if (u, v) in [(e[0], e[1]) for e in edges_chosen]:
                continue
            if (v, u) in [(e[0], e[1]) for e in edges_chosen]:
                continue
            # Score: how many times this sum and product have been seen
            s_count = sum_counts.get(s, 0)
            p_count = prod_counts.get(p, 0)
            # Prefer edges that reuse existing sums/products
            key = (s_count + p_count, -s, -p)
            if best_key is None or key > best_key:
                best_key = key
                best_edge = (u, v, s, p)
        
        if best_edge is None:
            break
        
        u, v, s, p = best_edge
        edges_chosen.append((u, v))
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    return [list(e[:2]) for e in edges_chosen]

def _make_powers_values():
    """
    Create 14 distinct values using powers of 2 and 3.
    Products of powers of 2 and 3 have unique exponent representations,
    but sums of same exponents collide.
    """
    # Use a mix: some powers of 2, some powers of 3
    # 2^0=1, 2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=32, 2^6=64, 2^7=128
    # 3^0=1, 3^1=3, 3^2=9, 3^3=27, 3^4=81, 3^5=243, 3^6=729
    
    # Strategy: use values that are products of small powers
    # to create many collisions in products
    
    # Let's try: 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36
    # These are all of form 2^a * 3^b with small exponents
    candidates = []
    for a in range(0, 6):  # 2^0 to 2^5
        for b in range(0, 6):  # 3^0 to 3^5
            val = (2 ** a) * (3 ** b)
            if val > 0 and val <= 10000 and val not in candidates:
                candidates.append(val)
    
    # Sort and take 14
    candidates.sort()
    values = candidates[:14]
    
    # Ensure distinctness
    assert len(set(values)) == 14, "Values must be distinct"
    assert all(isinstance(v, int) and 0 < v <= 10000 for v in values)
    
    return values

def _try_value_set(values, all_edges, deadline):
    """Try a value set with greedy edge selection, return best score."""
    if time.time() > deadline:
        return None
    
    edges = _greedy_edge_selection(values, P['m'], all_edges)
    if len(edges) != P['m']:
        return None
    
    score = _raw_score(values, edges)
    return score, edges

def _local_search(values, edges, all_edges, deadline):
    """
    Local search: try swapping edges to improve score.
    """
    current_score = _raw_score(values, edges)
    edge_set = set(tuple(e) for e in edges)
    
    for _ in range(200):  # limited iterations
        if time.time() > deadline:
            break
        
        improved = False
        # Try removing one edge and adding another
        for i, e1 in enumerate(edges):
            e1_tuple = tuple(e1)
            for e2 in all_edges:
                e2_tuple = tuple(e2)
                if e2_tuple == e1_tuple:
                    continue
                if e2_tuple in edge_set:
                    continue
                
                # Try the swap
                new_edges = edges[:]
                new_edges[i] = list(e2)
                new_score = _raw_score(values, new_edges)
                
                if new_score < current_score:
                    edges = new_edges
                    current_score = new_score
                    edge_set = set(tuple(e) for e in edges)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return current_score, edges

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline, 10s margin
    
    all_edges = _all_edges(P['n'])
    
    best_score = float('inf')
    best_edges = None
    best_values = None
    
    # Strategy 1: Powers of 2 and 3
    values1 = _make_powers_values()
    result = _try_value_set(values1, all_edges, deadline)
    if result is not None:
        score, edges = result
        if score < best_score:
            best_score = score
            best_edges = edges
            best_values = values1[:]
    
    # Local search on best
    if best_values is not None and time.time() < deadline:
        score, edges = _local_search(best_values, best_edges, all_edges, deadline)
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Strategy 2: Try a few more value sets
    # Powers of 2 only: 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192
    values2 = [2**i for i in range(14)]
    if time.time() < deadline:
        result = _try_value_set(values2, all_edges, deadline)
        if result is not None:
            score, edges = result
            if score < best_score:
                best_score = score
                best_edges = edges
                best_values = values2[:]
                score, edges = _local_search(best_values, best_edges, all_edges, deadline)
                if score < best_score:
                    best_score = score
                    best_edges = edges
    
    # Strategy 3: Mixed small numbers
    values3 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    if time.time() < deadline:
        result = _try_value_set(values3, all_edges, deadline)
        if result is not None:
            score, edges = result
            if score < best_score:
                best_score = score
                best_edges = edges
                best_values = values3[:]
                score, edges = _local_search(best_values, best_edges, all_edges, deadline)
                if score < best_score:
                    best_score = score
                    best_edges = edges
    
    # Fallback: if no valid result, use the first one
    if best_edges is None:
        values1 = _make_powers_values()
        edges = _greedy_edge_selection(values1, P['m'], all_edges)
        best_values = values1
        best_edges = edges
    
    # Ensure types are correct
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
