"""Rewritten program with stronger algorithm for small sums/products on graph edges."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import Counter

N = 14
M = 35
MAX_VAL = 10000

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _evaluate(values, edges):
    """Return (score, n_sums, n_prods) for given values and edges."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    ns, np_ = len(sums), len(prods)
    return max(ns, np_), ns, np_

def _build_pair_tables(values):
    """Precompute sum and product for all pairs, plus group by (sum, prod)."""
    n = len(values)
    all_pairs = list(combinations(range(n), 2))
    pair_sum = {}
    pair_prod = {}
    pair_key = {}
    for (u, v) in all_pairs:
        a, b = values[u], values[v]
        s = a + b
        p = a * b
        pair_sum[(u, v)] = s
        pair_prod[(u, v)] = p
        pair_key[(u, v)] = (s, p)
    return all_pairs, pair_sum, pair_prod, pair_key

def _greedy_edges_collisions(values, n_starts=50, time_budget=0.5):
    """
    Greedy edge selection maximizing collision of (sum, product) pairs.
    Edges sharing the same (sum, prod) pair contribute zero new distinct values.
    """
    n = len(values)
    all_pairs, pair_sum, pair_prod, pair_key = _build_pair_tables(values)
    
    best_edges = None
    best_score = float('inf')
    best_ns = best_np = 0
    start_time = time.time()
    
    for start in range(n_starts):
        if time.time() - start_time > time_budget:
            break
        selected = []
        sel_set = set()
        cur_sums = set()
        cur_prods = set()
        
        # Candidate ordering: shuffle pairs for diversity
        cand = all_pairs[:]
        random.shuffle(cand)
        
        for _ in range(M):
            best_e = None
            best_metric = float('inf')
            best_new_s = 0
            best_new_p = 0
            best_ns = best_np = 0
            
            for e in cand:
                if e in sel_set:
                    continue
                s = pair_sum[e]
                p = pair_prod[e]
                new_s = 0 if s in cur_sums else 1
                new_p = 0 if p in cur_prods else 1
                ns = len(cur_sums) + new_s
                np_ = len(cur_prods) + new_p
                metric = max(ns, np_)
                if metric < best_metric:
                    best_metric = metric
                    best_e = e
                    best_new_s = new_s
                    best_new_p = new_p
                    best_ns = ns
                    best_np = np_
            
            if best_e is None:
                break
            selected.append(best_e)
            sel_set.add(best_e)
            cur_sums.add(pair_sum[best_e])
            cur_prods.add(pair_prod[best_e])
        
        if len(selected) == M:
            sc = max(best_ns, best_np)
            if sc < best_score:
                best_score = sc
                best_ns = best_ns
                best_np = best_np
                best_edges = selected[:]
    
    return best_edges, best_score, best_ns, best_np

def _edge_swap_improve(values, edges, max_iters=500000):
    """
    Aggressive local search: try swapping one edge for another to reduce the score.
    Uses precomputed tables for speed.
    """
    n = len(values)
    all_pairs, pair_sum, pair_prod, pair_key = _build_pair_tables(values)
    
    edge_set = set(edges)
    cur_sums = set()
    cur_prods = set()
    for e in edges:
        cur_sums.add(pair_sum[e])
        cur_prods.add(pair_prod[e])
    cur_ns = len(cur_sums)
    cur_np = len(cur_prods)
    cur_score = max(cur_ns, cur_np)
    
    # Count occurrences of each sum and product among current edges
    sum_count = Counter()
    prod_count = Counter()
    for e in edges:
        sum_count[pair_sum[e]] += 1
        prod_count[pair_prod[e]] += 1
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try removing each edge and finding best replacement
        best_swap = None
        best_swap_metric = cur_score
        
        for i in range(len(edges)):
            e = edges[i]
            s = pair_sum[e]
            p = pair_prod[e]
            
            # After removing e, what are the new counts?
            ns_after = cur_ns - (1 if sum_count[s] == 1 else 0)
            np_after = cur_np - (1 if prod_count[p] == 1 else 0)
            
            # Find best candidate to add
            best_add = None
            best_add_metric = float('inf')
            for cand in all_pairs:
                if cand in edge_set:
                    continue
                cs = pair_sum[cand]
                cp = pair_prod[cand]
                # Check if cs would be present after removal of e
                cs_will_be_present = (cs != s) or (sum_count[s] > 1)
                cp_will_be_present = (cp != p) or (prod_count[p] > 1)
                new_ns = ns_after + (0 if cs_will_be_present else 1)
                new_np = np_after + (0 if cp_will_be_present else 1)
                m = max(new_ns, new_np)
                if m < best_add_metric:
                    best_add_metric = m
                    best_add = cand
            
            if best_add is not None and best_add_metric < best_swap_metric:
                best_swap_metric = best_add_metric
                best_swap = (i, e, best_add)
        
        if best_swap is not None and best_swap_metric < cur_score:
            i, e, new_e = best_swap
            edges[i] = new_e
            edge_set.remove(e)
            edge_set.add(new_e)
            # Update counts
            sum_count[pair_sum[e]] -= 1
            if sum_count[pair_sum[e]] == 0:
                del sum_count[pair_sum[e]]
            prod_count[pair_prod[e]] -= 1
            if prod_count[pair_prod[e]] == 0:
                del prod_count[pair_prod[e]]
            sum_count[pair_sum[new_e]] = sum_count.get(pair_sum[new_e], 0) + 1
            prod_count[pair_prod[new_e]] = prod_count.get(pair_prod[new_e], 0) + 1
            # Recompute actual sets
            cur_sums = set(pair_sum[ed] for ed in edges)
            cur_prods = set(pair_prod[ed] for ed in edges)
            cur_ns = len(cur_sums)
            cur_np = len(cur_prods)
            cur_score = max(cur_ns, cur_np)
            improved = True
    
    return edges, cur_score, cur_ns, cur_np

def _gen_values_for_collisions(rng):
    """
    Generate values specifically designed to create many (sum, product) collisions.
    Strategy: use values that are multiples of small numbers, creating shared factors.
    """
    vals = set()
    strategy = rng.randint(0, 5)
    
    if strategy == 0:
        # Small arithmetic progression: many pairs share similar sums
        base = rng.randint(1, 15)
        step = rng.randint(1, 3)
        vals = set(base + i * step for i in range(N + 5))
    elif strategy == 1:
        # Multiples of 2 and 3: creates shared factor structure
        vals = set()
        for i in range(N * 2):
            vals.add(rng.choice([2, 3, 6]) * rng.randint(1, 200))
    elif strategy == 2:
        # Powers of 2 times small odd numbers
        vals = set()
        for i in range(N * 3):
            pow2 = 2 ** rng.randint(0, 5)
            odd = rng.choice([1, 3, 5, 7, 9, 11, 13])
            vals.add(pow2 * odd)
    elif strategy == 3:
        # Very small values: 1-20 with some repeats removed
        vals = set(rng.randint(1, 25) for _ in range(N * 3))
    elif strategy == 4:
        # Geometric progression with ratio 2
        base = rng.randint(1, 8)
        vals = set(base * (2 ** i) for i in range(N + 3))
    else:
        # Random small values
        vals = set(rng.randint(1, 50) for _ in range(N * 3))
    
    # Ensure we have exactly N distinct values in range
    if len(vals) < N:
        while len(vals) < N:
            vals.add(rng.randint(1, MAX_VAL))
    vals = list(v for v in vals if 1 <= v <= MAX_VAL)[:N]
    return vals

def _mutate_values(values, rng, num_changes):
    """Randomly change num_changes values."""
    new_vals = values[:]
    for _ in range(num_changes):
        idx = rng.randint(0, N - 1)
        new_val = rng.randint(1, MAX_VAL)
        if new_val not in new_vals:
            new_vals[idx] = new_val
    return new_vals

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin
    rng = random.Random(42)
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Phase 1: Generate many candidate value sets focused on collision-friendly structures
    value_candidates = []
    
    for _ in range(80):
        if time.time() > deadline - 30:
            break
        vals = _gen_values_for_collisions(rng)
        if len(set(vals)) == N and all(1 <= v <= MAX_VAL for v in vals):
            value_candidates.append(vals)
    
    # Also add some random small-value candidates
    for _ in range(30):
        if time.time() > deadline - 30:
            break
        vals = list(rng.sample(range(1, 100), N))
        value_candidates.append(vals)
    
    # Evaluate each candidate with greedy edges
    for vals in value_candidates:
        if time.time() > deadline - 25:
            break
        edges, score, ns, np_ = _greedy_edges_collisions(vals, n_starts=30, time_budget=0.3)
        if edges is None:
            continue
        if score < best_score:
            best_score = score
            best_values = vals[:]
            best_edges = edges[:]
    
    if best_values is None:
        best_values = list(range(1, N + 1))
        best_edges, best_score, ns, np_ = _greedy_edges_collisions(best_values, n_starts=50, time_budget=2.0)
    
    # Phase 2: Simulated annealing on values with edge re-optimization
    current_values = best_values[:]
    current_edges = best_edges[:]
    current_score = best_score
    
    temperature = 30.0
    cooling_rate = 0.96
    min_temperature = 0.5
    
    while time.time() < deadline - 15.0 and temperature > min_temperature:
        # Choose mutation: mostly small changes
        if rng.random() < 0.6:
            num_changes = rng.randint(1, 2)
        else:
            num_changes = rng.randint(2, 4)
        
        new_values = _mutate_values(current_values, rng, num_changes)
        
        if len(set(new_values)) != N:
            continue
        if not all(1 <= v <= MAX_VAL for v in new_values):
            continue
        
        # Find best edges for new values
        new_edges, new_score, ns, np_ = _greedy_edges_collisions(new_values, n_starts=20, time_budget=0.2)
        if new_edges is None:
            continue
        
        # Acceptance with simulated annealing
        delta = new_score - current_score
        if delta < 0 or (temperature > 0 and rng.random() < pow(2.71828, -delta / temperature)):
            current_values = new_values
            current_edges = new_edges
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_values = current_values[:]
                best_edges = current_edges[:]
        
        temperature *= cooling_rate
    
    # Phase 3: Aggressive edge swap improvement on the best values
    if time.time() < deadline - 8.0:
        improved_edges, improved_score, ns, np_ = _edge_swap_improve(
            best_values, best_edges[:], max_iters=300000
        )
        if improved_score < best_score:
            best_score = improved_score
            best_edges = improved_edges[:]
    
    # Phase 4: Final re-optimize edges with more starts
    if time.time() < deadline - 3.0:
        final_edges, final_score, ns, np_ = _greedy_edges_collisions(
            best_values, n_starts=100, time_budget=5.0
        )
        if final_edges is not None and final_score < best_score:
            best_score = final_score
            best_edges = final_edges
    
    # Phase 5: One more edge swap pass
    if time.time() < deadline - 1.0:
        improved_edges, improved_score, ns, np_ = _edge_swap_improve(
            best_values, best_edges[:], max_iters=200000
        )
        if improved_score < best_score:
            best_score = improved_score
            best_edges = improved_edges[:]
    
    # Validation
    assert len(best_values) == N
    assert len(set(best_values)) == N
    assert all(isinstance(v, int) and 1 <= v <= MAX_VAL for v in best_values)
    assert len(best_edges) == M
    edge_tuples = set()
    for u, v in best_edges:
        assert isinstance(u, int) and isinstance(v, int)
        assert 0 <= u < N and 0 <= v < N
        assert u != v
        t = (min(u, v), max(u, v))
        assert t not in edge_tuples
        edge_tuples.add(t)
    
    return {'values': list(best_values), 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
