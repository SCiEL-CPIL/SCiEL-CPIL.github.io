import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill
    
    n = 14
    m = 35
    maximum = 10000
    
    def raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            v, w = values[i], values[j]
            sums.add(v + w)
            prods.add(v * w)
        return max(len(sums), len(prods))
    
    all_edges = list(combinations(range(n), 2))
    
    best_result = None
    best_score = float('inf')
    
    def try_config(vals, edges):
        nonlocal best_result, best_score
        if len(set(vals)) != n or any(v <= 0 or v > maximum for v in vals):
            return
        if len(edges) != m:
            return
        # Check no duplicates/loops
        edge_set = set()
        for i, j in edges:
            if i == j:
                return
            key = (min(i, j), max(i, j))
            if key in edge_set:
                return
            edge_set.add(key)
        
        score = raw_value(vals, edges)
        if score < best_score:
            best_score = score
            best_result = (list(vals), [list(e) for e in edges])
    
    # Generate candidate value sets
    value_candidates = []
    
    # Consecutive small values
    for base in range(1, 100):
        vals = [base + i for i in range(n)]
        if max(vals) <= maximum:
            value_candidates.append(vals)
    
    # Arithmetic progressions
    for base in range(1, 50):
        for diff in range(1, 15):
            vals = [base + i * diff for i in range(n)]
            if max(vals) <= maximum and len(set(vals)) == n:
                value_candidates.append(vals)
    
    # Random samples
    for _ in range(500):
        vals = sorted(random.sample(range(1, maximum + 1), n))
        value_candidates.append(vals)
    
    # For each value set, use greedy + local search for edges
    for vals in value_candidates:
        if time.time() > deadline - 10.0:
            break
        if len(set(vals)) != n or any(v <= 0 or v > maximum for v in vals):
            continue
        
        # Precompute
        edge_sums = [vals[i] + vals[j] for i, j in all_edges]
        edge_prods = [vals[i] * vals[j] for i, j in all_edges]
        
        # Greedy initialization: pick edges with most common sums/prods
        sum_freq = {}
        prod_freq = {}
        for s in edge_sums:
            sum_freq[s] = sum_freq.get(s, 0) + 1
        for p in edge_prods:
            prod_freq[p] = prod_freq.get(p, 0) + 1
        
        edge_priority = []
        for idx in range(len(all_edges)):
            s = edge_sums[idx]
            p = edge_prods[idx]
            priority = sum_freq[s] + prod_freq[p]
            edge_priority.append((priority, idx))
        edge_priority.sort(reverse=True)
        
        selected = []
        used = set()
        for priority, idx in edge_priority:
            if len(selected) >= m:
                break
            if idx not in used:
                selected.append(idx)
                used.add(idx)
        
        if len(selected) < m:
            continue
        
        edges = [all_edges[idx] for idx in selected]
        try_config(vals, edges)
        
        # Local search: try swapping edges
        if time.time() > deadline - 5.0:
            continue
        
        current_score = raw_value(vals, edges)
        current_sum_set = set()
        current_prod_set = set()
        for i, j in edges:
            current_sum_set.add(vals[i] + vals[j])
            current_prod_set.add(vals[i] * vals[j])
        
        # Try replacing one edge with another
        for _ in range(200):  # limited swaps
            if time.time() > deadline - 3.0:
                break
            
            # Pick a random edge in selected
            pos = random.randint(0, m - 1)
            old_idx = selected[pos]
            
            # Try a random replacement
            new_idx = random.randint(0, len(all_edges) - 1)
            if new_idx == old_idx or new_idx in used:
                continue
            
            # Compute new score
            new_edges = list(edges)
            new_edges[pos] = all_edges[new_idx]
            
            new_sums = set()
            new_prods = set()
            for i, j in new_edges:
                new_sums.add(vals[i] + vals[j])
                new_prods.add(vals[i] * vals[j])
            
            new_score = max(len(new_sums), len(new_prods))
            
            if new_score < current_score:
                selected[pos] = new_idx
                used.discard(old_idx)
                used.add(new_idx)
                edges = new_edges
                current_score = new_score
                current_sum_set = new_sums
                current_prod_set = new_prods
                try_config(vals, edges)
    
    if best_result is None:
        vals = list(range(1, n + 1))
        edges = [list(e) for e in all_edges[:m]]
        best_result = (vals, edges)
    
    vals, edges = best_result
    vals = [int(v) for v in vals]
    edges = [[int(i), int(j)] for i, j in edges]
    
    return {'values': vals, 'edges': edges}
