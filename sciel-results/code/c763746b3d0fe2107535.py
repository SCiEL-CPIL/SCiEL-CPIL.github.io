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
    Uses a set for O(1) lookup of chosen edges.
    """
    n = len(values)
    edges_chosen = []
    chosen_set = set()
    sum_counts = {}
    prod_counts = {}
    
    # Precompute all edge attributes
    edge_info = []
    for u, v in all_edges:
        a, b = values[u], values[v]
        s = a + b
        p = a * b
        edge_info.append((u, v, s, p))
    
    for _ in range(target_m):
        best_edge = None
        best_key = None
        for (u, v, s, p) in edge_info:
            if (u, v) in chosen_set:
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
        chosen_set.add((u, v))
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    return [list(e[:2]) for e in edges_chosen]

def _make_powers_values():
    """
    Create 14 distinct values using powers of 2 and 3.
    Products of powers of 2 and 3 have unique exponent representations,
    but sums of same exponents collide.
    """
    candidates = []
    for a in range(0, 6):
        for b in range(0, 6):
            val = (2 ** a) * (3 ** b)
            if val > 0 and val <= 10000 and val not in candidates:
                candidates.append(val)
    
    candidates.sort()
    values = candidates[:14]
    
    assert len(set(values)) == 14, "Values must be distinct"
    assert all(isinstance(v, int) and 0 < v <= 10000 for v in values)
    
    return values

def _make_high_collision_values():
    """
    Values designed for high product collision: many pairs share the same product.
    Use values where v_i * v_j = v_k * v_l for many quadruples.
    E.g., include 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36
    """
    return [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]

def _make_geometric_values():
    """Geometric sequence: high sum diversity but structured products."""
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

def _make_arithmetic_values():
    """Arithmetic sequence: high sum collision potential."""
    return [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

def _make_mixed_values():
    """Mixed small numbers with good structure."""
    return [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]

def _make_prime_based_values():
    """Values based on small primes for controlled product structure."""
    # 2,3,5,7,11,13,17,19,23,29,31,37,41,43 - distinct primes
    # Products are unique (fundamental theorem) but sums may collide
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]

def _make_composite_values():
    """Composite numbers with shared factors for product collisions."""
    # 4,6,8,9,10,12,14,15,16,18,20,21,22,24
    return [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24]

def _try_value_set(values, all_edges, deadline):
    """Try a value set with greedy edge selection, return best score."""
    if time.time() > deadline:
        return None
    
    edges = _greedy_edge_selection(values, P['m'], all_edges)
    if len(edges) != P['m']:
        return None
    
    score = _raw_score(values, edges)
    return score, edges

def _fast_local_search(values, edges, all_edges, deadline, max_iters=500):
    """
    Optimized local search: try swapping edges to improve score.
    Uses incremental score tracking for speed.
    """
    n = len(values)
    m = len(edges)
    
    # Precompute edge attributes
    edge_attrs = {}
    for u, v in all_edges:
        a, b = values[u], values[v]
        edge_attrs[(u, v)] = (a + b, a * b)
    
    current_edges = [list(e) for e in edges]
    edge_set = set(tuple(e) for e in current_edges)
    
    # Current score state
    sum_counts = {}
    prod_counts = {}
    for u, v in current_edges:
        s, p = edge_attrs[(u, v)]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    def _current_score():
        n_sums = sum(1 for c in sum_counts.values() if c > 0)
        n_prods = sum(1 for c in prod_counts.values() if c > 0)
        return max(n_sums, n_prods)
    
    current_score = _current_score()
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(m):
            u1, v1 = current_edges[i]
            e1 = (u1, v1)
            s1, p1 = edge_attrs[e1]
            
            for e2 in all_edges:
                if e2 == e1 or e2 in edge_set:
                    continue
                
                s2, p2 = edge_attrs[e2]
                
                # Simulate swap: remove e1, add e2
                old_s_count = sum_counts[s1]
                old_p_count = prod_counts[p1]
                new_s_count = sum_counts.get(s2, 0)
                new_p_count = prod_counts.get(p2, 0)
                
                # After removal
                sum_counts[s1] = old_s_count - 1
                prod_counts[p1] = old_p_count - 1
                if sum_counts[s1] == 0:
                    del sum_counts[s1]
                if prod_counts[p1] == 0:
                    del prod_counts[p1]
                
                # After addition
                sum_counts[s2] = new_s_count + 1
                prod_counts[p2] = new_p_count + 1
                
                new_score = _current_score()
                
                if new_score < current_score:
                    # Accept the swap
                    current_edges[i] = list(e2)
                    edge_set.remove(e1)
                    edge_set.add(e2)
                    current_score = new_score
                    improved = True
                    break
                else:
                    # Revert
                    sum_counts[s1] = old_s_count
                    prod_counts[p1] = old_p_count
                    sum_counts[s2] = new_s_count
                    if new_s_count == 0:
                        del sum_counts[s2]
                    prod_counts[p2] = new_p_count
                    if new_p_count == 0:
                        del prod_counts[p2]
            
            if improved:
                break
        
        if not improved:
            break
    
    return current_score, current_edges

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75 second deadline, 15s safety margin
    
    all_edges = _all_edges(P['n'])
    
    best_score = float('inf')
    best_edges = None
    best_values = None
    
    # Try multiple value set strategies
    value_strategies = [
        _make_powers_values,
        _make_high_collision_values,
        _make_geometric_values,
        _make_arithmetic_values,
        _make_mixed_values,
        _make_prime_based_values,
        _make_composite_values,
    ]
    
    for make_values in value_strategies:
        if time.time() > deadline - 10:  # Leave time for local search
            break
        
        values = make_values()
        if len(set(values)) != 14:
            continue
        if not all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            continue
        
        result = _try_value_set(values, all_edges, deadline)
        if result is not None:
            score, edges = result
            if score < best_score:
                best_score = score
                best_edges = edges
                best_values = values[:]
    
    # Local search on best
    if best_values is not None and best_edges is not None and time.time() < deadline - 5:
        score, edges = _fast_local_search(best_values, best_edges, all_edges, deadline)
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
