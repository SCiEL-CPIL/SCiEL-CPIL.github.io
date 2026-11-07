"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _score(vals, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        sums.add(vals[u] + vals[v])
        prods.add(vals[u] * vals[v])
    return max(len(sums), len(prods))

def _random_edges(n, m, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    all_edges = list(combinations(range(n), 2))
    rng.shuffle(all_edges)
    return [list(e) for e in all_edges[:m]]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin for 90s wall-clock limit
    
    n = P['n']
    m = P['m']
    
    # Initial solution: values 1..14, first 35 edges
    best_vals = list(range(1, n + 1))
    best_edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    best_score = _score(best_vals, best_edges)
    
    # Strategy: Try different value sets and edge sets
    # Key insight: We want to minimize max(distinct_sums, distinct_products)
    # 
    # For sums: if values are close together, sums have less spread
    # For products: if values are small, products are smaller and may collide more
    #
    # Let's try various value configurations:
    
    value_candidates = []
    
    # 1. Consecutive small values
    for start in range(1, 50):
        vals = list(range(start, start + n))
        if max(vals) <= P['maximum']:
            value_candidates.append(vals)
    
    # 2. Arithmetic progressions with small step
    for start in range(1, 30):
        for step in range(1, 5):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= P['maximum'] and len(set(vals)) == n:
                value_candidates.append(vals)
    
    # 3. Random distinct values in small range
    rng = random.Random(42)
    for _ in range(200):
        vals = rng.sample(range(1, 200), n)
        value_candidates.append(vals)
    
    # 4. Values with many common factors (to create product collisions)
    for base in range(2, 20):
        vals = []
        for i in range(n):
            v = base * (i + 1)
            if v <= P['maximum']:
                vals.append(v)
        if len(set(vals)) == n and len(vals) == n:
            value_candidates.append(vals)
    
    # 5. Powers of 2 (products are unique powers, but sums might collide)
    vals = [2**i for i in range(n)]
    if max(vals) <= P['maximum']:
        value_candidates.append(vals)
    
    # 6. Multiples with shared structure
    for mult in range(2, 15):
        vals = [mult * (i + 1) for i in range(n)]
        if max(vals) <= P['maximum'] and len(set(vals)) == n:
            value_candidates.append(vals)
    
    # 7. Random with bias toward small numbers
    for _ in range(500):
        vals = sorted(rng.sample(range(1, 100), n))
        value_candidates.append(vals)
    
    # 8. Very small values
    for start in range(1, 10):
        vals = list(range(start, start + n))
        if max(vals) <= P['maximum']:
            value_candidates.append(vals)
    
    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for vals in value_candidates:
        key = tuple(sorted(vals))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(vals)
    
    # For each value candidate, try multiple edge configurations
    # and use local search to improve edges
    
    for vals in unique_candidates:
        if time.time() > deadline:
            break
        
        # Try several random edge sets
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            if attempt == 0:
                edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
            else:
                edges = _random_edges(n, m, seed=attempt * 1000 + hash(tuple(vals)) % 10000)
            
            score = _score(vals, edges)
            if score < best_score:
                best_score = score
                best_vals = list(vals)
                best_edges = [list(e) for e in edges]
            
            # Local search on edges: try swapping edges
            if score <= best_score + 5:  # Only refine promising solutions
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    # Try removing one edge and adding another
                    for i in range(m):
                        if time.time() > deadline:
                            break
                        removed = edges[i]
                        # Try adding a random non-existing edge
                        all_pairs = list(combinations(range(n), 2))
                        existing = set(tuple(sorted(e)) for e in edges)
                        candidates = [e for e in all_pairs if tuple(sorted(e)) not in existing]
                        if not candidates:
                            continue
                        for _ in range(10):
                            if time.time() > deadline:
                                break
                            new_edge = list(rng.choice(candidates))
                            new_edges = edges[:i] + edges[i+1:] + [new_edge]
                            new_score = _score(vals, new_edges)
                            if new_score < score:
                                score = new_score
                                edges = new_edges
                                improved = True
                                if score < best_score:
                                    best_score = score
                                    best_vals = list(vals)
                                    best_edges = [list(e) for e in edges]
                                break
    
    # Final polish: try to improve best solution with more edge swaps
    if time.time() < deadline - 5:
        vals = best_vals
        edges = best_edges
        score = best_score
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            i = rng.randrange(m)
            removed = edges[i]
            all_pairs = list(combinations(range(n), 2))
            existing = set(tuple(sorted(e)) for e in edges)
            candidates = [e for e in all_pairs if tuple(sorted(e)) not in existing]
            if not candidates:
                break
            new_edge = list(rng.choice(candidates))
            new_edges = edges[:i] + edges[i+1:] + [new_edge]
            new_score = _score(vals, new_edges)
            if new_score < score:
                score = new_score
                edges = new_edges
                best_score = score
                best_edges = [list(e) for e in edges]
    
    # Ensure types are correct
    result_vals = [int(v) for v in best_vals]
    result_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': result_vals, 'edges': result_edges}
# EVOLVE-BLOCK-END
