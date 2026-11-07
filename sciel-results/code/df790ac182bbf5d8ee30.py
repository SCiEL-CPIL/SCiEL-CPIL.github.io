"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import Counter

def _raw_value(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _find_best_edges(values, m=35, n=14, time_limit=2.0):
    """Find m edges minimizing max(distinct_sums, distinct_products) using greedy + local search."""
    start_time = time.time()
    pairs = list(combinations(range(n), 2))
    pair_sums = {}
    pair_prods = {}
    for i, j in pairs:
        pair_sums[(i, j)] = values[i] + values[j]
        pair_prods[(i, j)] = values[i] * values[j]
    
    best_edges = None
    best_score = float('inf')
    
    def greedy_construct():
        selected = set()
        current_sums = set()
        current_prods = set()
        
        for _ in range(5):
            selected = set()
            current_sums = set()
            current_prods = set()
            
            shuffled_pairs = pairs[:]
            random.shuffle(shuffled_pairs)
            
            for _ in range(m):
                best_pair = None
                min_growth = float('inf')
                
                for p in shuffled_pairs:
                    if p in selected:
                        continue
                    
                    s = pair_sums[p]
                    pr = pair_prods[p]
                    
                    new_s = 1 if s not in current_sums else 0
                    new_p = 1 if pr not in current_prods else 0
                    growth = new_s + new_p
                    
                    if growth < min_growth:
                        min_growth = growth
                        best_pair = p
                
                if best_pair is None:
                    break
                    
                selected.add(best_pair)
                current_sums.add(pair_sums[best_pair])
                current_prods.add(pair_prods[best_pair])
            
            if len(selected) == m:
                edges = list(selected)
                score = _raw_value(values, edges)
                return edges, score
        
        return None, float('inf')

    edges, score = greedy_construct()
    if edges:
        best_edges = edges
        best_score = score
        
    if edges and time.time() < start_time + time_limit:
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            sum_counts = Counter()
            prod_counts = Counter()
            for e in edges:
                sum_counts[pair_sums[e]] += 1
                prod_counts[pair_prods[e]] += 1
            
            edge_set = set(edges)
            
            for idx, e in enumerate(edges):
                old_s = pair_sums[e]
                old_p = pair_prods[e]
                
                for new_e in pairs:
                    if new_e in edge_set:
                        continue
                    
                    new_s = pair_sums[new_e]
                    new_p = pair_prods[new_e]
                    
                    sum_counts[old_s] -= 1
                    if sum_counts[old_s] == 0:
                        del sum_counts[old_s]
                    prod_counts[old_p] -= 1
                    if prod_counts[old_p] == 0:
                        del prod_counts[old_p]
                    
                    sum_counts[new_s] += 1
                    prod_counts[new_p] += 1
                    
                    new_score = max(len(sum_counts), len(prod_counts))
                    
                    if new_score < best_score:
                        edges[idx] = new_e
                        edge_set.remove(e)
                        edge_set.add(new_e)
                        best_score = new_score
                        improved = True
                    else:
                        sum_counts[new_s] -= 1
                        if sum_counts[new_s] == 0:
                            del sum_counts[new_s]
                        prod_counts[new_p] -= 1
                        if prod_counts[new_p] == 0:
                            del prod_counts[new_p]
                        sum_counts[old_s] += 1
                        prod_counts[old_p] += 1
                        break
                
                if improved:
                    break
    
    return best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 14
    m = 35
    max_val = 10000
    
    best_result = None
    best_score = float('inf')
    
    # Candidate value sets
    candidates = []
    candidates.append(list(range(1, n+1)))
    candidates.append(list(range(2, n+2)))
    candidates.append([2*i for i in range(1, n+1)])
    candidates.append([3*i for i in range(1, n+1)])
    candidates.append([2**i for i in range(n)])
    
    for _ in range(10):
        vals = random.sample(range(1, 50), n)
        candidates.append(vals)
    for _ in range(10):
        vals = [random.randint(1, 20) for _ in range(n)]
        if len(set(vals)) == n:
            candidates.append(vals)
    
    seen = set()
    unique_candidates = []
    for vals in candidates:
        key = tuple(sorted(vals))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(vals)
    
    for vals in unique_candidates:
        if time.time() > deadline:
            break
        
        if len(set(vals)) != n:
            continue
        if any(v < 1 or v > max_val for v in vals):
            continue
            
        edges, score = _find_best_edges(vals, m, n, time_limit=1.0)
        if edges is not None and score < best_score:
            best_score = score
            best_result = {'values': list(vals), 'edges': [list(e) for e in edges]}
    
    if best_result is not None and time.time() < deadline - 5:
        current_vals = best_result['values']
        current_edges = best_result['edges']
        current_score = best_score
        
        for _ in range(20):
            if time.time() > deadline - 2:
                break
                
            new_vals = list(current_vals)
            idx = random.randint(0, n-1)
            new_val = max(1, min(max_val, new_vals[idx] + random.randint(-10, 10)))
            
            if new_val == new_vals[idx]:
                continue
            if new_val in [v for i, v in enumerate(new_vals) if i != idx]:
                continue
                
            new_vals[idx] = new_val
            
            new_edges, new_score = _find_best_edges(new_vals, m, n, time_limit=0.5)
            if new_edges is not None and new_score < current_score:
                current_vals = new_vals
                current_edges = new_edges
                current_score = new_score
                best_score = new_score
                best_result = {'values': list(current_vals), 'edges': [list(e) for e in current_edges]}
                
    if best_result is None:
        best_result = {'values': list(range(1, n+1)), 'edges': [list(e) for e in list(combinations(range(n), 2))[:m]]}
        
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
# EVOLVE-BLOCK-END
