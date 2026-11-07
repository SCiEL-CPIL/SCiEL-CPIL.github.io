import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_raw = float('inf')
    best_result = None
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    all_edges = list(combinations(range(14), 2))
    
    # Generate diverse value candidates
    value_candidates = []
    
    # Consecutive
    for offset in range(1, 50):
        vals = list(range(offset, offset + 14))
        if max(vals) <= 10000:
            value_candidates.append(vals)
    
    # Arithmetic progressions
    for step in range(1, 15):
        for offset in range(1, 30):
            vals = [offset + i * step for i in range(14)]
            if max(vals) <= 10000:
                value_candidates.append(vals)
    
    # Small structured sets
    base_sets = [
        list(range(1, 15)),
        list(range(2, 16)),
        [2,4,6,8,10,12,14,16,18,20,22,24,26,28],
        [3,6,9,12,15,18,21,24,27,30,33,36,39,42],
        [5,10,15,20,25,30,35,40,45,50,55,60,65,70],
        [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
        [1,3,5,7,9,11,13,15,17,19,21,23,25,27],
    ]
    value_candidates.extend(base_sets)
    
    # Remove duplicates
    seen = set()
    unique_candidates = []
    for v in value_candidates:
        t = tuple(v)
        if t not in seen:
            seen.add(t)
            unique_candidates.append(v)
    value_candidates = unique_candidates
    
    def greedy_pick_edges(values, all_edges):
        selected = []
        sums = set()
        prods = set()
        remaining = list(all_edges)
        
        for _ in range(35):
            if time.time() > deadline:
                break
            best_edge = None
            best_score = float('inf')
            for e in remaining:
                u, v = e
                s = values[u] + values[v]
                p = values[u] * values[v]
                new_sums = len(sums) + (0 if s in sums else 1)
                new_prods = len(prods) + (0 if p in prods else 1)
                score = max(new_sums, new_prods)
                if score < best_score:
                    best_score = score
                    best_edge = e
            if best_edge is None:
                break
            selected.append(best_edge)
            remaining.remove(best_edge)
            u, v = best_edge
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        
        return selected
    
    # Try greedy on all candidates
    for values in value_candidates:
        if time.time() > deadline - 10.0:
            break
        edges = greedy_pick_edges(values, all_edges)
        if len(edges) == 35:
            raw = evaluate(values, edges)
            if raw < best_raw:
                best_raw = raw
                best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
    
    # Local search on best result
    if best_result and time.time() < deadline - 5.0:
        values = best_result['values']
        current_edges = set(tuple(e) for e in best_result['edges'])
        current_raw = best_raw
        
        edge_sums = {}
        edge_prods = {}
        for u, v in all_edges:
            edge_sums[(u,v)] = values[u] + values[v]
            edge_prods[(u,v)] = values[u] * values[v]
        
        random.seed(42)
        local_deadline = deadline - 5.0
        
        while time.time() < local_deadline:
            e_in = random.choice(list(current_edges))
            candidates_out = [e for e in all_edges if e not in current_edges]
            e_out = random.choice(candidates_out)
            
            new_edges = (current_edges - {e_in}) | {e_out}
            
            sums = set()
            prods = set()
            for u, v in new_edges:
                sums.add(edge_sums[(u,v)])
                prods.add(edge_prods[(u,v)])
            new_raw = max(len(sums), len(prods))
            
            if new_raw < current_raw:
                current_raw = new_raw
                current_edges = new_edges
                if current_raw < best_raw:
                    best_raw = current_raw
                    best_result = {'values': list(values), 'edges': [list(e) for e in current_edges]}
    
    if best_result:
        vals = [int(v) for v in best_result['values']]
        eds = [[int(u), int(v)] for u, v in best_result['edges']]
        return {'values': vals, 'edges': eds}
    else:
        return {'values': list(range(1, 15)), 'edges': [list(e) for e in list(combinations(range(14), 2))[:35]]}
