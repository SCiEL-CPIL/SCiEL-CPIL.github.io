import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)
    
    def score_of(values, edge_list):
        sums = set()
        prods = set()
        for u, v in edge_list:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def greedy_edge_select(values, budget_time=5.0):
        """
        Greedily select m edges maximizing collision count.
        At each step, pick the edge that adds the fewest NEW distinct sums/products.
        Use a priority: prefer edges where both sum and product are already seen.
        """
        end_time = start_time + budget_time
        edges = []
        edge_set = set()
        sum_counts = {}
        prod_counts = {}
        
        # Precompute all edge sums and products
        edge_data = {}
        for e in all_edges:
            s = values[e[0]] + values[e[1]]
            p = values[e[0]] * values[e[1]]
            edge_data[e] = (s, p)
        
        # Group edges by (sum, product) - edges with same sum+product are interchangeable
        # Group edges by sum, and by product
        sum_groups = {}
        prod_groups = {}
        for e, (s, p) in edge_data.items():
            sum_groups.setdefault(s, []).append(e)
            prod_groups.setdefault(p, []).append(e)
        
        # Greedy: at each step, pick edge that minimizes new distinct values
        # Score an edge: -1 if sum already seen, -1 if product already seen
        # Prefer edges with both collisions
        
        for step in range(m):
            if time.time() > end_time:
                break
            
            best_edge = None
            best_new_distinct = float('inf')
            best_score = float('inf')
            
            for e in all_edges:
                if e in edge_set:
                    continue
                s, p = edge_data[e]
                new_distinct = 0
                if s not in sum_counts:
                    new_distinct += 1
                if p not in prod_counts:
                    new_distinct += 1
                
                # Tie-break: prefer edges that are in dense groups
                group_size = len(sum_groups[s]) + len(prod_groups[p])
                
                if new_distinct < best_new_distinct or (new_distinct == best_new_distinct and group_size > best_score):
                    best_new_distinct = new_distinct
                    best_score = group_size
                    best_edge = e
            
            if best_edge is None:
                # Fallback: pick any unused edge
                for e in all_edges:
                    if e not in edge_set:
                        best_edge = e
                        break
            
            if best_edge is None:
                break
            
            edges.append(best_edge)
            edge_set.add(best_edge)
            s, p = edge_data[best_edge]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        return edges
    
    def local_search_edges(values, edges, budget_iters=50000):
        """Local search to improve edge selection."""
        edges = list(edges)
        edges_set = set(edges)
        cur_score, sum_counts, prod_counts = _score_with_counts(values, edges)
        
        # Precompute
        edge_data = {}
        for e in all_edges:
            edge_data[e] = (values[e[0]] + values[e[1]], values[e[0]] * values[e[1]])
        
        no_improve = 0
        it = 0
        while it < budget_iters and time.time() < deadline - 8.0:
            it += 1
            
            # Single edge swap
            i = random.randint(0, m - 1)
            e_out = edges[i]
            s_out, p_out = edge_data[e_out]
            
            # Find best incoming edge
            best_e_in = None
            best_new_score = cur_score + 1  # threshold
            
            # Sample candidates
            candidates = [e for e in all_edges if e not in edges_set]
            random.shuffle(candidates)
            candidates = candidates[:min(200, len(candidates))]
            
            for e_in in candidates:
                s_in, p_in = edge_data[e_in]
                
                # Simulate
                sc = dict(sum_counts)
                pc = dict(prod_counts)
                
                sc[s_out] -= 1
                if sc[s_out] == 0:
                    del sc[s_out]
                pc[p_out] -= 1
                if pc[p_out] == 0:
                    del pc[p_out]
                
                sc[s_in] = sc.get(s_in, 0) + 1
                pc[p_in] = pc.get(p_in, 0) + 1
                
                ns = max(len(sc), len(pc))
                if ns < best_new_score:
                    best_new_score = ns
                    best_e_in = e_in
                
                if best_new_score < cur_score:
                    break
            
            if best_e_in is not None and best_new_score <= cur_score:
                edges[i] = best_e_in
                edges_set.discard(e_out)
                edges_set.add(best_e_in)
                
                s_in, p_in = edge_data[best_e_in]
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0:
                    del sum_counts[s_out]
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0:
                    del prod_counts[p_out]
                sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
                prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
                cur_score = best_new_score
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > 5000:
                break
        
        return edges, cur_score
    
    def _score_with_counts(values, edge_list):
        sum_counts = {}
        prod_counts = {}
        for u, v in edge_list:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        return max(len(sum_counts), len(prod_counts)), sum_counts, prod_counts
    
    def local_search_values(values, edges, budget_iters=50000):
        """Local search to improve values for fixed edges."""
        vals = list(values)
        cur_score, sum_counts, prod_counts = _score_with_counts(vals, edges)
        
        vertex_edges = [[] for _ in range(n)]
        for u, v in edges:
            vertex_edges[u].append(v)
            vertex_edges[v].append(u)
        
        no_improve = 0
        it = 0
        while it < budget_iters and time.time() < deadline - 8.0:
            it += 1
            idx = random.randint(0, n - 1)
            old_val = vals[idx]
            
            # Try to target existing sums or products
            r = random.random()
            if r < 0.3 and vertex_edges[idx]:
                nbr = random.choice(vertex_edges[idx])
                # Target an existing sum
                if sum_counts:
                    target = random.choice(list(sum_counts.keys()))
                    new_val = target - vals[nbr]
                    if new_val <= 0 or new_val > 10000:
                        new_val = old_val + random.randint(-10, 10)
                else:
                    new_val = old_val + random.randint(-10, 10)
            elif r < 0.5 and vertex_edges[idx]:
                nbr = random.choice(vertex_edges[idx])
                # Target an existing product
                if prod_counts:
                    target = random.choice(list(prod_counts.keys()))
                    if vals[nbr] != 0 and target % vals[nbr] == 0:
                        new_val = target // vals[nbr]
                        if new_val <= 0 or new_val > 10000:
                            new_val = old_val + random.randint(-10, 10)
                    else:
                        new_val = old_val + random.randint(-10, 10)
                else:
                    new_val = old_val + random.randint(-10, 10)
            elif r < 0.7:
                new_val = old_val + random.randint(-5, 5)
            elif r < 0.85:
                new_val = random.randint(1, 100)
            else:
                new_val = random.randint(1, 500)
            
            if new_val <= 0 or new_val > 10000:
                continue
            if any(v == new_val for j, v in enumerate(vals) if j != idx):
                continue
            
            # Incremental update
            for nbr in vertex_edges[idx]:
                s_old = old_val + vals[nbr]
                p_old = old_val * vals[nbr]
                s_new = new_val + vals[nbr]
                p_new = new_val * vals[nbr]
                
                sum_counts[s_old] -= 1
                if sum_counts[s_old] == 0:
                    del sum_counts[s_old]
                prod_counts[p_old] -= 1
                if prod_counts[p_old] == 0:
                    del prod_counts[p_old]
                
                sum_counts[s_new] = sum_counts.get(s_new, 0) + 1
                prod_counts[p_new] = prod_counts.get(p_new, 0) + 1
            
            new_score = max(len(sum_counts), len(prod_counts))
            
            if new_score <= cur_score:
                vals[idx] = new_val
                cur_score = new_score
                no_improve = 0
            else:
                for nbr in vertex_edges[idx]:
                    s_old = old_val + vals[nbr]
                    p_old = old_val * vals[nbr]
                    s_new = new_val + vals[nbr]
                    p_new = new_val * vals[nbr]
                    
                    sum_counts[s_new] -= 1
                    if sum_counts[s_new] == 0:
                        del sum_counts[s_new]
                    prod_counts[p_new] -= 1
                    if prod_counts[p_new] == 0:
                        del prod_counts[p_new]
                    
                    sum_counts[s_old] = sum_counts.get(s_old, 0) + 1
                    prod_counts[p_old] = prod_counts.get(p_old, 0) + 1
                no_improve += 1
            
            if no_improve > 3000:
                break
        
        return vals, cur_score
    
    # Generate structured candidate value sets
    candidates = []
    
    # 1. Divisors of 60: {1,2,3,4,5,6,10,12,15,20,30,60} + extras
    candidates.append([1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60, 7, 14])
    
    # 2. Divisors of 120: {1,2,3,4,5,6,8,10,12,15,20,24,30,40,60,120}
    candidates.append([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40])
    
    # 3. Divisors of 240: many factors
    candidates.append([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30])
    
    # 4. Small structured: pairs that sum to 10 and 20
    candidates.append([1, 9, 2, 8, 3, 7, 4, 6, 5, 11, 12, 13, 14, 15])
    
    # 5. Multiplicative structure: {1,2,3,4,6,12} × {1,5,7} style
    candidates.append([1, 2, 3, 4, 5, 6, 7, 10, 12, 14, 15, 20, 21, 28])
    
    # 6. Arithmetic progression (sum collisions)
    candidates.append(list(range(1, 15)))
    
    # 7. Powers of 2 (product structure)
    candidates.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    
    # 8. Highly composite: divisors of 360
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24])
    
    # 9. Mixed small
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # 10. Values with many common sum targets
    # (1,19),(2,18),(3,17),(4,16),(5,15),(6,14),(7,13),(8,12),(9,11) all sum to 20
    candidates.append([1, 19, 2, 18, 3, 17, 4, 16, 5, 15, 6, 14, 7, 13])
    
    # 11. Values with many common product targets
    # 1×24=2×12=3×8=4×6=24; 1×30=2×15=3×10=5×6=30
    candidates.append([1, 24, 2, 12, 3, 8, 4, 6, 5, 15, 10, 7, 9, 11])
    
    # 12. Compact with structure
    candidates.append([2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 40])
    
    # Evaluate each candidate with greedy edge selection
    best_score = float('inf')
    best_vals = None
    best_edges = None
    
    for cand in candidates:
        if time.time() > deadline - 30.0:
            break
        
        # Validate candidate
        if len(set(cand)) != n:
            continue
        if any(v < 1 or v > 10000 for v in cand):
            continue
        
        # Greedy edge selection
        edges = greedy_edge_select(cand, budget_time=3.0)
        if len(edges) < m:
            # Fill with random unused edges
            used = set(edges)
            for e in all_edges:
                if len(edges) >= m:
                    break
                if e not in used:
                    edges.append(e)
                    used.add(e)
        
        sc = score_of(cand, edges)
        if sc < best_score:
            best_score = sc
            best_vals = list(cand)
            best_edges = list(edges)
    
    # Local search on edges for best candidate
    if best_vals is not None and time.time() < deadline - 15.0:
        edges_ls, sc_ls = local_search_edges(best_vals, best_edges, budget_iters=30000)
        if sc_ls < best_score:
            best_score = sc_ls
            best_edges = edges_ls
    
    # Local search on values
    if best_vals is not None and time.time() < deadline - 10.0:
        vals_ls, sc_ls = local_search_values(best_vals, best_edges, budget_iters=30000)
        if sc_ls < best_score:
            best_score = sc_ls
            best_vals = vals_ls
    
    # Another round of edge optimization with improved values
    if best_vals is not None and time.time() < deadline - 5.0:
        edges_ls, sc_ls = local_search_edges(best_vals, best_edges, budget_iters=20000)
        if sc_ls < best_score:
            best_score = sc_ls
            best_edges = edges_ls
    
    # Final polish
    if best_vals is not None and time.time() < deadline - 2.0:
        vals_ls, sc_ls = local_search_values(best_vals, best_edges, budget_iters=10000)
        if sc_ls < best_score:
            best_score = sc_ls
            best_vals = vals_ls
    
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])
    
    # Validate and convert
    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    assert len(final_values) == 14
    assert len(set(final_values)) == 14
    assert all(isinstance(v, int) and 1 <= v <= 10000 for v in final_values)
    assert len(final_edges) == 35
    edge_set_check = set()
    for e in final_edges:
        assert isinstance(e, list) and len(e) == 2
        assert isinstance(e[0], int) and isinstance(e[1], int)
        assert 0 <= e[0] < 14 and 0 <= e[1] < 14
        assert e[0] != e[1]
        key = tuple(sorted(e))
        assert key not in edge_set_check
        edge_set_check.add(key)
    
    return {'values': final_values, 'edges': final_edges}
