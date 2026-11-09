"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_DEADLINE = time.time() + 85  # 5s safety margin before 90s kill

def _score(values, edges):
    """Compute raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _best_edges_for_values(values, m, n, max_iter=20000):
    """Local search to find best m-edge subset minimizing score."""
    all_edges = list(combinations(range(n), 2))
    # Start with random subset
    edges = set(random.sample(all_edges, m))
    current_score = _score(values, list(edges))
    best_score = current_score
    best_edges = set(edges)
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        if current_score <= 2:  # Can't do better than 2 (need at least 2 distinct)
            break
        # Try swapping one edge in, one out
        u, v = random.choice(list(edges))
        candidates = [e for e in all_edges if e not in edges]
        if not candidates:
            break
        x, y = random.choice(candidates)
        new_edges = set(edges)
        new_edges.discard((u, v))
        new_edges.add((x, y))
        new_score = _score(values, list(new_edges))
        if new_score <= current_score:
            edges = new_edges
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = set(edges)
    
    return best_score, [list(e) for e in best_edges]

def _try_value_set(values, m, n, max_iter=10000):
    """Try a value set with local search on edges."""
    if time.time() > _DEADLINE:
        return None
    score, edges = _best_edges_for_values(values, m, n, max_iter)
    return score, values, edges

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m  # worst case: all distinct
    best_result = None
    
    # Candidate value sets to try
    candidates = []
    
    # 1. Small consecutive integers
    candidates.append(list(range(1, n + 1)))
    
    # 2. Small even numbers
    candidates.append(list(range(2, 2 * n + 1, 2)))
    
    # 3. Highly composite-like small numbers
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    
    # 4. Powers of 2 (known to give unique sums/products, but try anyway)
    candidates.append([2 ** i for i in range(n)])
    
    # 5. Mixed small values with lots of collisions
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # 6. Try some random small value sets
    random.seed(42)
    for _ in range(20):
        vals = sorted(random.sample(range(1, 50), n))
        candidates.append(vals)
    
    # 7. Arithmetic progressions
    for start in range(1, 6):
        for step in range(1, 4):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val:
                candidates.append(vals)
    
    for vals in candidates:
        if time.time() > _DEADLINE:
            break
        # Validate
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > max_val:
            continue
        # Ensure all are Python int
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
        
        result = _try_value_set(vals, m, n, max_iter=5000)
        if result is None:
            continue
        score, values, edges = result
        if score < best_score:
            best_score = score
            best_result = (values, edges)
            if best_score <= 2:
                break
    
    if best_result is None:
        # Fallback
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
    
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
