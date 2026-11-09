import time
import random
from itertools import combinations

_DEADLINE = time.time() + 80  # 80s safety margin out of 90s

def _score(values, edges):
    """Compute max(distinct sums, distinct products) over edges."""
    sums = set()
    products = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        products.add(values[i] * values[j])
    return max(len(sums), len(products))

def _best_edges_for_values(values, n=14, m=35):
    """Find 35 edges that minimize max(distinct sums, distinct products)."""
    all_edges = list(combinations(range(n), 2))
    # Greedy: start with all edges, remove the one whose removal reduces score most
    # Or: start empty, add edge that increases score least
    # Let's try: start with all 91 edges, iteratively remove worst edge
    edges_set = set(all_edges)
    current_sums = set()
    current_products = set()
    for i, j in edges_set:
        current_sums.add(values[i] + values[j])
        current_products.add(values[i] * values[j])
    
    while len(edges_set) > m:
        best_remove = None
        best_score = _score(values, list(edges_set))
        for e in edges_set:
            new_edges_set = edges_set - {e}
            new_sums = set()
            new_products = set()
            for i, j in new_edges_set:
                new_sums.add(values[i] + values[j])
                new_products.add(values[i] * values[j])
            s = max(len(new_sums), len(new_products))
            if best_remove is None or s < best_score:
                best_score = s
                best_remove = e
        if best_remove is not None:
            edges_set.remove(best_remove)
        else:
            break
    
    return [list(e) for e in edges_set]

def _greedy_add_edges(values, n=14, m=35):
    """Start with no edges, greedily add the edge that increases score least."""
    all_edges = list(combinations(range(n), 2))
    edges = []
    current_sums = set()
    current_products = set()
    
    for _ in range(m):
        best_edge = None
        best_score = float('inf')
        for e in all_edges:
            if e in edges:
                continue
            i, j = e
            s = values[i] + values[j]
            p = values[i] * values[j]
            new_sums = current_sums | {s}
            new_products = current_products | {p}
            score = max(len(new_sums), len(new_products))
            if score < best_score:
                best_score = score
                best_edge = e
        if best_edge is None:
            break
        edges.append(best_edge)
        i, j = best_edge
        current_sums.add(values[i] + values[j])
        current_products.add(values[i] * values[j])
    
    return [list(e) for e in edges]

def _local_search(values, max_iters=5000):
    """Local search: mutate values, keep best edges."""
    n = len(values)
    best_values = list(values)
    best_edges = _greedy_add_edges(best_values)
    best_score = _score(best_values, best_edges)
    
    iters = 0
    while iters < max_iters and time.time() < _DEADLINE - 2:
        iters += 1
        new_values = list(best_values)
        idx = random.randint(0, n - 1)
        # Mutate: pick a new random value
        new_val = random.randint(1, 10000)
        # Ensure distinctness
        if new_val in new_values and new_values.count(new_val) > 0:
            # Remove old, add new
            new_values[idx] = new_val
            if len(set(new_values)) != n:
                continue
        else:
            new_values[idx] = new_val
            if len(set(new_values)) != n:
                continue
        
        new_edges = _greedy_add_edges(new_values)
        new_score = _score(new_values, new_edges)
        
        if new_score < best_score:
            best_values = new_values
            best_edges = new_edges
            best_score = new_score
    
    return best_values, best_edges, best_score

def construct():
    random.seed(42)
    n = 14
    m = 35
    
    # Try several initial configurations
    initial_sets = [
        list(range(1, n + 1)),
        list(range(2, n + 2)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30],
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
    ]
    
    best_overall = None
    best_overall_score = float('inf')
    
    for init in initial_sets:
        if time.time() > _DEADLINE - 5:
            break
        if len(set(init)) != n or any(v < 1 or v > 10000 for v in init):
            continue
        vals, edges, score = _local_search(init, max_iters=3000)
        if score < best_overall_score:
            best_overall_score = score
            best_overall = (vals, edges)
    
    if best_overall is None:
        # Fallback
        vals = list(range(1, n + 1))
        edges = _greedy_add_edges(vals)
        best_overall = (vals, edges)
    
    vals, edges = best_overall
    # Ensure types are correct
    vals = [int(v) for v in vals]
    edges = [[int(i), int(j)] for i, j in edges]
    
    return {'values': vals, 'edges': edges}
