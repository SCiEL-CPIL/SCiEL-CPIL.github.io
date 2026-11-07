"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def greedy_edges(values, all_edges, rng):
        """Greedy: pick edges that reuse existing sums/products."""
        sums_seen = set()
        prods_seen = set()
        edges = []
        remaining = list(all_edges)
        rng.shuffle(remaining)
        
        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            for e in remaining:
                u, v = e
                s = values[u] + values[v]
                p = values[u] * values[v]
                score = (0 if s in sums_seen else 1) + (0 if p in prods_seen else 1)
                if score < best_score:
                    best_score = score
                    best_edge = e
                    if best_score == 0:
                        break
            if best_edge is None:
                best_edge = remaining[0]
            edges.append(best_edge)
            remaining.remove(best_edge)
            u, v = best_edge
            sums_seen.add(values[u] + values[v])
            prods_seen.add(values[u] * values[v])
        
        return edges
    
    def local_search(values, edges, rng, max_iter=5000):
        """Swap edges and values to improve."""
        all_edges = list(combinations(range(n), 2))
        current_score = evaluate(values, edges)
        best_values = list(values)
        best_edges = [list(e) for e in edges]
        best_score = current_score
        edge_set = set(tuple(e) for e in edges)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if rng.random() < 0.5:
                # Try swapping an edge
                if not edges:
                    continue
                idx = rng.randrange(len(edges))
                old_edge = tuple(edges[idx])
                # Pick a random non-edge
                candidates = [e for e in all_edges if e not in edge_set]
                if not candidates:
                    continue
                new_edge = rng.choice(candidates)
                new_edges = list(edges)
                new_edges[idx] = new_edge
                new_edge_set = set(tuple(e) for e in new_edges)
                new_score = evaluate(values, new_edges)
                if new_score <= best_score:
                    best_score = new_score
                    best_edges = [list(e) for e in new_edges]
                    edges = new_edges
                    edge_set = new_edge_set
            else:
                # Try swapping two values
                i, j = rng.sample(range(n), 2)
                new_values = list(values)
                new_values[i], new_values[j] = new_values[j], new_values[i]
                if len(set(new_values)) != n:
                    continue
                new_score = evaluate(new_values, edges)
                if new_score <= best_score:
                    best_score = new_score
                    best_values = new_values
                    values = new_values
        
        return best_values, best_edges, best_score
    
    rng = random.Random(42)
    all_edges = list(combinations(range(n), 2))
    
    best_overall = float('inf')
    best_result = None
    
    # Try different value set strategies
    value_sets = []
    
    # Strategy 1: Small consecutive integers
    for start in range(1, 50):
        vals = list(range(start, start + n))
        if max(vals) <= max_val:
            value_sets.append(vals)
    
    # Strategy 2: Arithmetic progressions with small step
    for start in range(1, 100):
        for step in range(1, 5):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val and len(set(vals)) == n:
                value_sets.append(vals)
    
    # Strategy 3: Random values in small range
    for _ in range(100):
        lo = rng.randint(1, 200)
        hi = lo + rng.randint(n, 50)
        vals = rng.sample(range(lo, hi + 1), n)
        value_sets.append(vals)
    
    # Strategy 4: Powers and products
    for base in [2, 3, 4]:
        vals = [base ** i for i in range(n)]
        if max(vals) <= max_val:
            value_sets.append(vals)
    
    # Deduplicate
    seen_sets = set()
    unique_value_sets = []
    for vs in value_sets:
        key = tuple(sorted(vs))
        if key not in seen_sets:
            seen_sets.add(key)
            unique_value_sets.append(vs)
    
    # Evaluate each with greedy edges + local search
    for vs in unique_value_sets:
        if time.time() > deadline - 10:
            break
        edges = greedy_edges(vs, all_edges, rng)
        vals, eds, score = local_search(vs, edges, rng, max_iter=2000)
        if score < best_overall:
            best_overall = score
            best_result = (list(vals), [list(e) for e in eds])
    
    if best_result is None:
        # Fallback
        vals = list(range(1, n + 1))
        edges = [list(e) for e in all_edges[:m]]
    else:
        vals, edges = best_result
    
    # Final validation
    assert len(vals) == n
    assert len(set(vals)) == n
    assert all(isinstance(v, int) and 1 <= v <= max_val for v in vals)
    assert len(edges) == m
    assert all(isinstance(u, int) and isinstance(v, int) and 0 <= u < n and 0 <= v < n and u != v for u, v in edges)
    assert len(set(tuple(sorted(e)) for e in edges)) == m
    
    return {'values': vals, 'edges': edges}
# EVOLVE-BLOCK-END
