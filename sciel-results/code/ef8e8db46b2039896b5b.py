import time
import itertools
import numpy as np

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

def _make_linear_values():
    return list(range(1, 15))

def _make_quadratic_values():
    return [i * i for i in range(1, 15)]

def _make_powers_values():
    return [2 ** i for i in range(14)]

def _make_arithmetic_values():
    # Arithmetic progression with larger gaps to create sum collisions
    return [1 + 7 * i for i in range(14)]

def _make_geometric_values():
    # Geometric-like values for product collisions
    vals = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 9999]
    return vals

def _make_high_collision_values():
    # Values designed to maximize sum/product collisions
    # Use pairs that share sums: e.g., 1+14=2+13=3+12=...
    # And products: 1*14=2*7, 2*14=4*7, etc.
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

def _spectral_edge_selection(values, target_m, all_edges, deadline):
    """
    Spectral approach: Build similarity matrix based on sum/product collisions,
    then use eigenvector analysis to select edges that maximize intra-cluster connectivity.
    """
    n = len(values)
    edge_list = all_edges  # all C(14,2) = 91 edges
    
    # Precompute sum and product for each edge
    edge_sums = {}
    edge_prods = {}
    for u, v in edge_list:
        a, b = values[u], values[v]
        edge_sums[(u, v)] = a + b
        edge_prods[(u, v)] = a * b
    
    # Build collision-based similarity matrix
    # W[i,j] = 1 if edge (i,j) shares a sum OR product with edge (k,l)
    # Actually, we want: for each pair of edges, how "similar" are they?
    # Two edges are similar if they share a sum value or a product value
    
    m_total = len(edge_list)
    edge_idx = {e: i for i, e in enumerate(edge_list)}
    
    # Build sum collision matrix and product collision matrix
    sum_groups = {}
    prod_groups = {}
    for e in edge_list:
        s = edge_sums[e]
        p = edge_prods[e]
        if s not in sum_groups:
            sum_groups[s] = []
        sum_groups[s].append(e)
        if p not in prod_groups:
            prod_groups[p] = []
        prod_groups[p].append(e)
    
    # Similarity: edges that share a sum or product
    # Build weighted adjacency: weight = 2 if share both, 1 if share one, 0 if neither
    W = np.zeros((m_total, m_total), dtype=np.float64)
    
    for group in sum_groups.values():
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    e1, e2 = group[i], group[j]
                    i1, i2 = edge_idx[e1], edge_idx[e2]
                    W[i1, i2] += 1.0
                    W[i2, i1] += 1.0
    
    for group in prod_groups.values():
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    e1, e2 = group[i], group[j]
                    i1, i2 = edge_idx[e1], edge_idx[e2]
                    W[i1, i2] += 1.0
                    W[i2, i1] += 1.0
    
    # Build Laplacian L = D - W
    D = np.diag(W.sum(axis=1))
    L = D - W
    
    # Compute eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        # Fallback to greedy
        return _greedy_edge_selection(values, target_m, all_edges)
    
    # The Fiedler vector (2nd smallest eigenvalue) gives the best bipartition
    # For dense subgraph selection, we want edges that are "central" in the collision graph
    # Use the principal eigenvector (largest eigenvalue) or a combination
    
    # Strategy: rank edges by their "centrality" in the collision graph
    # An edge is central if it has high degree in the collision graph
    # AND its endpoints are well-connected to other high-degree edges
    
    # Compute edge centrality score:
    # 1. Degree in collision graph
    # 2. Weighted sum of neighbor degrees (like a 2-hop centrality)
    
    degree = W.sum(axis=1)
    
    # Also use eigenvector information
    # The largest eigenvector gives a global "importance" signal
    # Use the component corresponding to the largest eigenvalue
    idx_max = np.argmax(eigvals)
    principal_vec = eigvecs[:, idx_max]
    
    # Combine: centrality = degree + alpha * |principal_vec|
    alpha = 0.5
    centrality = degree + alpha * np.abs(principal_vec)
    
    # But we also need to ensure the selected edges form a valid simple graph
    # (no two edges can share a vertex if we want a matching, but we don't need that)
    # Actually, we just need 35 distinct edges from the 91 available
    
    # Greedy selection guided by centrality:
    # Pick the highest-centrality edge, then among remaining, pick next highest
    # that doesn't create conflicts (but there are no conflicts in this problem - 
    # we just need distinct edges)
    
    # Actually, the constraint is just: 35 distinct edges from 91 available.
    # So we can just pick the top 35 by centrality!
    
    # But wait - we need to verify this gives a good score. Let me think...
    # The collision graph tells us which edges share sum/product values.
    # If we pick edges that are highly connected in the collision graph,
    # they tend to share sum/product values with each other, reducing distinct counts.
    
    sorted_edges = sorted(range(m_total), key=lambda i: -centrality[i])
    
    # But we need to be careful: picking all top-35 might not be optimal
    # because we need the actual graph structure to matter.
    # Let's do a greedy selection that considers both centrality and actual collision benefit.
    
    selected = []
    selected_set = set()
    sum_counts = {}
    prod_counts = {}
    
    for idx in sorted_edges:
        if len(selected) >= target_m:
            break
        if time.time() > deadline:
            break
        
        e = edge_list[idx]
        if e in selected_set:
            continue
        
        u, v = e
        s = edge_sums[e]
        p = edge_prods[e]
        
        # Check if adding this edge is beneficial
        # (It always is in terms of not increasing distinct counts if it reuses)
        selected.append(e)
        selected_set.add(e)
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    if len(selected) < target_m:
        # Fill remaining with any unused edges
        for e in edge_list:
            if len(selected) >= target_m:
                break
            if e not in selected_set:
                selected.append(e)
                selected_set.add(e)
    
    return [list(e) for e in selected[:target_m]]

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
            
            new_s = 1 if s_count == 0 else 0
            new_p = 1 if p_count == 0 else 0
            
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

def _local_search(values, edges, all_edges, deadline, max_iters=200):
    """
    Local search: try edge swaps to reduce max(distinct_sums, distinct_products).
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
        n_sums = len(sum_counts)
        n_prods = len(prod_counts)
        return max(n_sums, n_prods)
    
    current_score = _current_score()
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        # Try each edge swap systematically
        for i in range(m):
            if time.time() > deadline:
                break
            
            u1, v1 = current_edges[i]
            e1 = (u1, v1)
            s1, p1 = edge_attrs[e1]
            
            for e2 in all_edges:
                if e2 == e1 or e2 in edge_set:
                    continue
                
                s2, p2 = edge_attrs[e2]
                
                # Simulate swap
                old_s1 = sum_counts[s1]
                old_p1 = prod_counts[p1]
                new_s2 = sum_counts.get(s2, 0)
                new_p2 = prod_counts.get(p2, 0)
                
                sum_counts[s1] = old_s1 - 1
                prod_counts[p1] = old_p1 - 1
                if sum_counts[s1] == 0:
                    del sum_counts[s1]
                if prod_counts[p1] == 0:
                    del prod_counts[p1]
                
                sum_counts[s2] = new_s2 + 1
                prod_counts[p2] = new_p2 + 1
                
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
                    sum_counts[s1] = old_s1
                    prod_counts[p1] = old_p1
                    sum_counts[s2] = new_s2
                    if new_s2 == 0:
                        del sum_counts[s2]
                    prod_counts[p2] = new_p2
                    if new_p2 == 0:
                        del prod_counts[p2]
            
            if improved:
                break
        
        if not improved:
            break
    
    return current_score, current_edges

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s limit
    
    all_edges = _all_edges(P['n'])
    
    best_score = float('inf')
    best_edges = None
    best_values = None
    
    value_strategies = [
        _make_linear_values,
        _make_quadratic_values,
        _make_powers_values,
        _make_arithmetic_values,
        _make_geometric_values,
        _make_high_collision_values,
    ]
    
    for make_values in value_strategies:
        if time.time() > deadline - 20:
            break
        
        values = make_values()
        if len(set(values)) != 14:
            continue
        if not all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            continue
        
        # Try spectral selection
        edges_spec = _spectral_edge_selection(values, P['m'], all_edges, deadline - 10)
        if len(edges_spec) != P['m']:
            continue
        
        score_spec, edges_spec = _local_search(values, edges_spec, all_edges, deadline - 5)
        
        if score_spec < best_score:
            best_score = score_spec
            best_edges = edges_spec
            best_values = values[:]
        
        # Also try greedy as comparison
        if time.time() > deadline - 15:
            continue
        
        edges_greedy = _greedy_edge_selection(values, P['m'], all_edges)
        if len(edges_greedy) != P['m']:
            continue
        
        score_greedy, edges_greedy = _local_search(values, edges_greedy, all_edges, deadline - 5)
        
        if score_greedy < best_score:
            best_score = score_greedy
            best_edges = edges_greedy
            best_values = values[:]
    
    if best_edges is None:
        values = _make_linear_values()
        edges = _greedy_edge_selection(values, P['m'], all_edges)
        best_values = values
        best_edges = edges
    
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
