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
    Greedy: pick edges that minimize the increase in distinct sums and products.
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
            
            s_count = sum_counts.get(s, 0)
            p_count = prod_counts.get(p, 0)
            
            # We want to minimize the number of NEW distinct sums and products.
            # new_s = 0 if s in sum_counts else 1
            # new_p = 0 if p in prod_counts else 1
            new_s = 1 if s_count == 0 else 0
            new_p = 1 if p_count == 0 else 0
            
            # Primary goal: minimize new distinct values.
            # Secondary: prefer edges with higher existing counts (more "popular").
            key = (new_s + new_p, -s_count, -p_count)
            
            if best_key is None or key < best_key:
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

def _make_linear_values():
    return list(range(1, 15))

def _make_quadratic_values():
    return [i*i for i in range(1, 15)]

def _make_powers_values():
    return [2**i for i in range(14)]

def _fast_local_search(values, edges, all_edges, deadline, max_iters=1000):
    """
    Local search to minimize max(distinct_sums, distinct_products).
    """
    n = len(values)
    m = len(edges)
    
    edge_attrs = {}
    for u, v in all_edges:
        a, b = values[u], values[v]
        edge_attrs[(u, v)] = (a + b, a * b)
    
    current_edges = [list(e) for e in edges]
    edge_set = set(tuple(e) for e in current_edges)
    
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
        # Try swapping each edge with a random or systematic set of others
        # To be faster, iterate through all possible swaps but break early if improved
        for i in range(m):
            u1, v1 = current_edges[i]
            e1 = (u1, v1)
            s1, p1 = edge_attrs[e1]
            
            for e2 in all_edges:
                if e2 == e1 or e2 in edge_set:
                    continue
                
                s2, p2 = edge_attrs[e2]
                
                old_s_count = sum_counts[s1]
                old_p_count = prod_counts[p1]
                new_s_count = sum_counts.get(s2, 0)
                new_p_count = prod_counts.get(p2, 0)
                
                # Simulate removal
                sum_counts[s1] = old_s_count - 1
                prod_counts[p1] = old_p_count - 1
                if sum_counts[s1] == 0:
                    del sum_counts[s1]
                if prod_counts[p1] == 0:
                    del prod_counts[p1]
                
                # Simulate addition
                sum_counts[s2] = new_s_count + 1
                prod_counts[p2] = new_p_count + 1
                
                new_score = _current_score()
                
                if new_score < current_score:
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
    deadline = start_time + 80  # 10s safety margin
    
    all_edges = _all_edges(P['n'])
    
    best_score = float('inf')
    best_edges = None
    best_values = None
    
    value_strategies = [
        _make_linear_values,
        _make_quadratic_values,
        _make_powers_values,
    ]
    
    for make_values in value_strategies:
        if time.time() > deadline - 15:
            break
        
        values = make_values()
        if len(set(values)) != 14:
            continue
        if not all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            continue
        
        edges = _greedy_edge_selection(values, P['m'], all_edges)
        if len(edges) != P['m']:
            continue
            
        score, edges = _fast_local_search(values, edges, all_edges, deadline - 5)
        
        if score < best_score:
            best_score = score
            best_edges = edges
            best_values = values[:]
            
    if best_edges is None:
        values = _make_linear_values()
        edges = _greedy_edge_selection(values, P['m'], all_edges)
        best_values = values
        best_edges = edges
        
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
