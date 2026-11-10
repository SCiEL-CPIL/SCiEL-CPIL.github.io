"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 14
M = 35
MAX_VAL = 10000
DEADLINE = 85.0  # safety margin before 90s kill

def _score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _optimize_edges(values, time_budget):
    """Find 35 edges minimizing max(distinct_sums, distinct_products)."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sum and product for each edge
    edge_sums = {}
    edge_prods = {}
    for e in all_edges:
        i, j = e
        edge_sums[e] = values[i] + values[j]
        edge_prods[e] = values[i] * values[j]
    
    # Initial: pick edges with most "popular" sums and products
    # Count frequency of each sum and product
    from collections import Counter
    sum_counts = Counter(edge_sums[e] for e in all_edges)
    prod_counts = Counter(edge_prods[e] for e in all_edges)
    
    # Score each edge: prefer edges where sum and product are common
    def edge_priority(e):
        return (sum_counts[edge_sums[e]] + prod_counts[edge_prods[e]], -edge_sums[e])
    
    # Start with top 35 by priority
    sorted_edges = sorted(all_edges, key=edge_priority, reverse=True)
    current_edges = set(sorted_edges[:M])
    current_score = _score(values, list(current_edges))
    
    best_edges = set(current_edges)
    best_score = current_score
    
    start_time = time.time()
    
    # Local search: try swapping one edge in for one edge out
    in_edges = list(current_edges)
    out_edges = [e for e in all_edges if e not in current_edges]
    
    random.shuffle(in_edges)
    random.shuffle(out_edges)
    
    while time.time() - start_time < time_budget:
        improved = False
        for i in range(len(in_edges)):
            if time.time() - start_time >= time_budget:
                break
            e_in = in_edges[i]
            for j in range(len(out_edges)):
                e_out = out_edges[j]
                new_edges = (current_edges - {e_in}) | {e_out}
                s = _score(values, list(new_edges))
                if s < current_score:
                    current_edges = new_edges
                    current_score = s
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = set(current_edges)
                    improved = True
                    break
        if not improved:
            # Try random restarts or perturbation
            # Remove a random edge and add a random one
            e_remove = random.choice(list(current_edges))
            candidates = [e for e in all_edges if e not in current_edges]
            if not candidates:
                break
            e_add = random.choice(candidates)
            current_edges = (current_edges - {e_remove}) | {e_add}
            current_score = _score(values, list(current_edges))
            # Re-shuffle
            in_edges = list(current_edges)
            out_edges = [e for e in all_edges if e not in current_edges]
            random.shuffle(in_edges)
            random.shuffle(out_edges)
    
    return list(best_edges), best_score

def _try_value_set(values, time_budget):
    edges, score = _optimize_edges(values, time_budget)
    return values, edges, score

def construct():
    start_time = time.time()
    
    # Generate candidate value sets
    candidate_sets = []
    
    # Family 1: multiples of small numbers
    for base in [2, 3, 4, 5, 6, 8, 10, 12]:
        vals = [base * (k + 1) for k in range(N)]
        if max(vals) <= MAX_VAL:
            candidate_sets.append(vals)
    
    # Family 2: arithmetic progressions
    for a in range(1, 20):
        for d in range(1, 10):
            vals = [a + d * k for k in range(N)]
            if max(vals) <= MAX_VAL:
                candidate_sets.append(vals)
    
    # Family 3: near-constant with small variation
    for center in [10, 20, 50, 100, 200, 500]:
        for spread in [1, 2, 3, 5, 10]:
            vals = [center + (k - N // 2) * spread for k in range(N)]
            if all(v > 0 and v <= MAX_VAL for v in vals):
                candidate_sets.append(vals)
    
    # Family 4: factor-rich numbers
    factor_rich = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]
    candidate_sets.append(factor_rich)
    
    # Family 5: powers and multiples
    for base in [2, 3]:
        vals = []
        v = base
        while len(vals) < N and v <= MAX_VAL:
            vals.append(v)
            v *= base
        if len(vals) == N:
            candidate_sets.append(vals)
    
    # Family 6: small numbers with repeats of structure
    small_vals = list(range(1, N + 1))
    candidate_sets.append(small_vals)
    
    # Family 7: even numbers
    even_vals = list(range(2, 2 * N + 1, 2))
    candidate_sets.append(even_vals)
    
    # Family 8: numbers with common factors
    for g in [2, 3, 4, 6]:
        vals = [g * (2 * k + 1) for k in range(N)]  # odd multiples
        if max(vals) <= MAX_VAL:
            candidate_sets.append(vals)
        vals2 = [g * (2 * k + 2) for k in range(N)]  # even multiples
        if max(vals2) <= MAX_VAL:
            candidate_sets.append(vals2)
    
    # Remove duplicates
    seen = set()
    unique_sets = []
    for vs in candidate_sets:
        key = tuple(sorted(vs))
        if key not in seen:
            seen.add(key)
            unique_sets.append(vs)
    
    # Sort by initial heuristic: smaller max value might be better
    unique_sets.sort(key=lambda vs: max(vs))
    
    best_result = None
    best_score = float('inf')
    
    # Time per value set
    n_sets = len(unique_sets)
    time_per_set = min(3.0, (DEADLINE - (time.time() - start_time)) / max(1, n_sets))
    
    for idx, vs in enumerate(unique_sets):
        if time.time() - start_time > DEADLINE - 2:
            break
        remaining = DEADLINE - (time.time() - start_time) - 1
        if remaining < 0.5:
            break
        t_budget = min(time_per_set, remaining)
        
        values, edges, score = _try_value_set(vs, t_budget)
        
        if score < best_score:
            best_score = score
            best_result = (values, edges)
        
        # Early stop if we get a very good score
        if best_score <= 10:
            break
    
    # If no result found, use default
    if best_result is None:
        values = list(range(1, N + 1))
        edges = [list(e) for e in list(combinations(range(N), 2))[:M]]
        best_score = _score(values, edges)
    else:
        values, edges = best_result
    
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(i), int(j)] for i, j in edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
