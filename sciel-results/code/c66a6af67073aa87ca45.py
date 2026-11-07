"""Direct constructive approach for small sums and products along selected graph edges."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 14
    m = 35
    max_val = 10000
    all_edges = list(combinations(range(n), 2))
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        ns = len(sums)
        np_ = len(prods)
        return ns, np_, max(ns, np_)
    
    def find_edges_greedy_pruning(values, rng, time_limit, max_distinct):
        """Greedy edge selection with pruning: skip edges that would exceed max_distinct."""
        sums_seen = set()
        prods_seen = set()
        selected = []
        remaining = list(all_edges)
        rng.shuffle(remaining)
        
        for _ in range(m):
            if time.time() > time_limit:
                break
            best_edge = None
            best_score = float('inf')
            
            for edge in remaining:
                u, v = edge
                s = values[u] + values[v]
                p = values[u] * values[v]
                
                sum_new = s not in sums_seen
                prod_new = p not in prods_seen
                
                if sum_new and len(sums_seen) + 1 > max_distinct:
                    continue
                if prod_new and len(prods_seen) + 1 > max_distinct:
                    continue
                
                score = 0
                if sum_new:
                    score += 2
                if prod_new:
                    score += 2
                score += (s + p) / 1000000.0
                
                if score < best_score:
                    best_score = score
                    best_edge = edge
            
            if best_edge is None:
                # Relax: pick best remaining edge
                for edge in remaining:
                    u, v = edge
                    s = values[u] + values[v]
                    p = values[u] * values[v]
                    score = (s + p) / 1000000.0
                    if s not in sums_seen:
                        score += 2
                    if p not in prods_seen:
                        score += 2
                    if score < best_score:
                        best_score = score
                        best_edge = edge
            
            if best_edge is None:
                break
            
            selected.append(best_edge)
            remaining.remove(best_edge)
            u, v = best_edge
            sums_seen.add(values[u] + values[v])
            prods_seen.add(values[u] * values[v])
        
        # Fill if needed
        if len(selected) < m:
            used = set(tuple(e) for e in selected)
            for e in all_edges:
                if len(selected) >= m:
                    break
                if tuple(e) not in used:
                    selected.append(e)
                    used.add(tuple(e))
        
        return selected
    
    def local_search_edges(values, edges, rng, time_limit):
        """Local search on edges: swap one edge for another to reduce distinct sums/prods."""
        edges_set = set(tuple(e) for e in edges)
        _, _, current_score = evaluate(values, edges)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(300):
                if (time.time() - start_time) > time_limit:
                    break
                i = rng.randint(0, len(edges)-1)
                new_edge = rng.choice(all_edges)
                if tuple(new_edge) in edges_set:
                    continue
                
                old_edge = edges[i]
                edges[i] = new_edge
                edges_set.discard(tuple(old_edge))
                edges_set.add(tuple(new_edge))
                
                ns, np_, new_score = evaluate(values, edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    edges[i] = old_edge
                    edges_set.discard(tuple(new_edge))
                    edges_set.add(tuple(old_edge))
        
        return edges, current_score
    
    def local_search_values(values, edges, rng, time_limit):
        """Local search on values: perturb one value at a time to reduce distinct sums/prods."""
        _, _, current_score = evaluate(values, edges)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(60):
                if (time.time() - start_time) > time_limit:
                    break
                idx = rng.randint(0, n-1)
                old_val = values[idx]
                
                candidates = set()
                for delta in range(-15, 16):
                    v = old_val + delta
                    if 1 <= v <= max_val:
                        candidates.add(v)
                for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36, 40, 48, 60]:
                    for mult in range(1, min(60, max_val // base + 1)):
                        candidates.add(base * mult)
                for base in [2, 3, 4, 5]:
                    v = 1
                    while v <= max_val:
                        candidates.add(v)
                        v *= base
                for _ in range(20):
                    candidates.add(rng.randint(1, max_val))
                
                best_new_val = None
                best_new_score = current_score
                for new_val in sorted(candidates):
                    if new_val < 1 or new_val > max_val:
                        continue
                    if new_val in values:
                        continue
                    values[idx] = new_val
                    ns, np_, new_score = evaluate(values, edges)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                    else:
                        values[idx] = old_val
                        break
                    values[idx] = old_val
                
                if best_new_val is not None:
                    values[idx] = best_new_val
                    current_score = best_new_score
                    improved = True
        
        return values, current_score
    
    def local_search_both(values, edges, rng, time_limit):
        """Combined local search: alternate between value and edge optimization."""
        _, _, current_score = evaluate(values, edges)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # Try value moves
            for _ in range(40):
                if (time.time() - start_time) > time_limit:
                    break
                idx = rng.randint(0, n-1)
                old_val = values[idx]
                
                candidates = set()
                for delta in range(-10, 11):
                    v = old_val + delta
                    if 1 <= v <= max_val:
                        candidates.add(v)
                for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 24, 30, 36, 48, 60]:
                    for mult in range(1, min(50, max_val // base + 1)):
                        candidates.add(base * mult)
                for _ in range(15):
                    candidates.add(rng.randint(1, max_val))
                
                best_new_val = None
                best_new_score = current_score
                for new_val in sorted(candidates):
                    if new_val < 1 or new_val > max_val:
                        continue
                    if new_val in values:
                        continue
                    values[idx] = new_val
                    ns, np_, new_score = evaluate(values, edges)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                    else:
                        values[idx] = old_val
                        break
                    values[idx] = old_val
                
                if best_new_val is not None:
                    values[idx] = best_new_val
                    current_score = best_new_score
                    improved = True
            
            # Try edge moves
            edges_set = set(tuple(e) for e in edges)
            for _ in range(200):
                if (time.time() - start_time) > time_limit:
                    break
                i = rng.randint(0, len(edges)-1)
                new_edge = rng.choice(all_edges)
                if tuple(new_edge) in edges_set:
                    continue
                
                old_edge = edges[i]
                edges[i] = new_edge
                edges_set.discard(tuple(old_edge))
                edges_set.add(tuple(new_edge))
                
                ns, np_, new_score = evaluate(values, edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    edges[i] = old_edge
                    edges_set.discard(tuple(new_edge))
                    edges_set.add(tuple(old_edge))
        
        return values, edges, current_score
    
    best_overall_score = float('inf')
    best_result = None
    rng = random.Random(12345)
    
    value_strategies = [
        [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84],
        [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168],
        [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 30],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
        [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    ]
    
    # Phase 1: Try each value strategy with greedy edge selection + pruning
    for init_values in value_strategies:
        if (time.time() - start_time) > deadline - 15.0:
            break
        
        # Try multiple max_distinct targets
        for max_d in [18, 20, 22, 25, 30]:
            if (time.time() - start_time) > deadline - 12.0:
                break
            
            edges = find_edges_greedy_pruning(init_values, rng, deadline - 12.0, max_d)
            _, _, score = evaluate(init_values, edges)
            if score < best_overall_score:
                best_overall_score = score
                best_result = {'values': list(init_values), 'edges': [list(e) for e in edges]}
        
        # Local search on edges
        if best_result is not None and (time.time() - start_time) < deadline - 10.0:
            edges_ls = [list(e) for e in best_result['edges']]
            edges_ls, score_e = local_search_edges(init_values, edges_ls, rng, deadline - 10.0)
            if score_e < best_overall_score:
                best_overall_score = score_e
                best_result = {'values': list(init_values), 'edges': [list(e) for e in edges_ls]}
        
        # Local search on values
        if best_result is not None and (time.time() - start_time) < deadline - 8.0:
            values_ls = list(init_values)
            edges_ls = [list(e) for e in best_result['edges']]
            values_ls, score_v = local_search_values(values_ls, edges_ls, rng, deadline - 8.0)
            if score_v < best_overall_score:
                best_overall_score = score_v
                best_result = {'values': list(values_ls), 'edges': [list(e) for e in edges_ls]}
        
        # Combined local search
        if best_result is not None and (time.time() - start_time) < deadline - 5.0:
            values_ls = list(best_result['values'])
            edges_ls = [list(e) for e in best_result['edges']]
            values_ls, edges_ls, score_c = local_search_both(values_ls, edges_ls, rng, deadline - 5.0)
            if score_c < best_overall_score:
                best_overall_score = score_c
                best_result = {'values': list(values_ls), 'edges': [list(e) for e in edges_ls]}
    
    # Phase 2: More aggressive refinement on best result
    if best_result is not None:
        values = list(best_result['values'])
        edges = [list(e) for e in best_result['edges']]
        
        for iteration in range(30):
            if (time.time() - start_time) > deadline - 2.0:
                break
            
            values, edges, s = local_search_both(values, edges, rng, deadline - 2.0)
            if s < best_overall_score:
                best_overall_score = s
                best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
    
    # Phase 3: Final polish with tighter search
    if best_result is not None and (time.time() - start_time) < deadline - 1.0:
        values = list(best_result['values'])
        edges = [list(e) for e in best_result['edges']]
        
        # Try more diverse value perturbations
        for _ in range(20):
            if (time.time() - start_time) > deadline - 1.0:
                break
            idx = rng.randint(0, n-1)
            old_val = values[idx]
            candidates = set()
            for delta in range(-20, 21):
                v = old_val + delta
                if 1 <= v <= max_val:
                    candidates.add(v)
            for _ in range(30):
                candidates.add(rng.randint(1, max_val))
            
            best_new_val = None
            best_new_score = evaluate(values, edges)[2]
            for new_val in sorted(candidates):
                if new_val < 1 or new_val > max_val or new_val in values:
                    continue
                values[idx] = new_val
                ns, np_, new_score = evaluate(values, edges)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_val = new_val
                else:
                    values[idx] = old_val
                    break
                values[idx] = old_val
            
            if best_new_val is not None:
                values[idx] = best_new_val
                if best_new_score < best_overall_score:
                    best_overall_score = best_new_score
                    best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
    
    if best_result is None:
        best_result = {
            'values': list(range(1, n+1)),
            'edges': [list(e) for e in list(all_edges)[:m]]
        }
    
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
