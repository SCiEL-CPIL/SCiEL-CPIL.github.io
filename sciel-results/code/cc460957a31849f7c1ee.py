"""Rewritten program with stronger algorithm for small sums/products on graph edges."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

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

def _greedy_edges(values, n_starts=30, time_budget=0.4):
    """
    Greedy edge selection: pick M edges minimizing max(distinct sums, distinct products).
    Multiple random starts with randomized candidate ordering.
    """
    n = len(values)
    all_pairs = list(combinations(range(n), 2))
    
    # Precompute sum and product for each pair
    pair_sum = {}
    pair_prod = {}
    for (u, v) in all_pairs:
        a, b = values[u], values[v]
        pair_sum[(u, v)] = a + b
        pair_prod[(u, v)] = a * b
    
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
                best_edges = selected
    
    return best_edges, best_score, best_ns, best_np

def _edge_swap_improve(values, edges, max_iters=200000):
    """
    Local search: try swapping one edge for another to reduce the score.
    Returns (edges, score, ns, np).
    """
    n = len(values)
    all_pairs = list(combinations(range(n), 2))
    pair_sum = {}
    pair_prod = {}
    for (u, v) in all_pairs:
        a, b = values[u], values[v]
        pair_sum[(u, v)] = a + b
        pair_prod[(u, v)] = a * b
    
    edge_set = set(edges)
    cur_sums = set()
    cur_prods = set()
    for e in edges:
        cur_sums.add(pair_sum[e])
        cur_prods.add(pair_prod[e])
    cur_ns = len(cur_sums)
    cur_np = len(cur_prods)
    cur_score = max(cur_ns, cur_np)
    
    # For each edge, count how many edges share its sum/product (to know removal effect)
    sum_count = {}
    prod_count = {}
    for e in edges:
        s = pair_sum[e]
        p = pair_prod[e]
        sum_count[s] = sum_count.get(s, 0) + 1
        prod_count[p] = prod_count.get(p, 0) + 1
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try removing each edge
        for i in range(len(edges)):
            e = edges[i]
            s = pair_sum[e]
            p = pair_prod[e]
            # After removing e
            ns_after = cur_ns - (1 if sum_count[s] == 1 else 0)
            np_after = cur_np - (1 if prod_count[p] == 1 else 0)
            base_metric = max(ns_after, np_after)
            
            # Find a candidate edge to add (not in edge_set) that keeps metric minimal
            best_add = None
            best_add_metric = float('inf')
            for cand in all_pairs:
                if cand in edge_set:
                    continue
                cs = pair_sum[cand]
                cp = pair_prod[cand]
                # Is cs already present after removal?
                cs_present = (cs in sum_count and sum_count[cs] > (1 if cs == s else 0))
                cp_present = (cp in prod_count and prod_count[cp] > (1 if cp == p else 0))
                new_ns = ns_after + (0 if cs_present else 1)
                new_np = np_after + (0 if cp_present else 1)
                m = max(new_ns, new_np)
                if m < best_add_metric:
                    best_add_metric = m
                    best_add = cand
                    if m < cur_score:
                        # good enough, but keep searching for better
                        pass
            
            if best_add is not None and best_add_metric < cur_score:
                # Apply swap
                edges[i] = best_add
                edge_set.remove(e)
                edge_set.add(best_add)
                # Update counts
                sum_count[s] -= 1
                if sum_count[s] == 0:
                    del sum_count[s]
                prod_count[p] -= 1
                if prod_count[p] == 0:
                    del prod_count[p]
                cs = pair_sum[best_add]
                cp = pair_prod[best_add]
                sum_count[cs] = sum_count.get(cs, 0) + 1
                prod_count[cp] = prod_count.get(cp, 0) + 1
                cur_sums = set(pair_sum[ed] for ed in edges)
                cur_prods = set(pair_prod[ed] for ed in edges)
                cur_ns = len(cur_sums)
                cur_np = len(cur_prods)
                cur_score = max(cur_ns, cur_np)
                improved = True
                break
    return edges, cur_score, cur_ns, cur_np

def _gen_values_structured(rng):
    """
    Generate values with structure to encourage sum/product collisions.
    Strategy: pick a base, create arithmetic/geometric progressions,
    and values sharing common factors.
    """
    vals = set()
    strategy = rng.randint(0, 5)
    
    if strategy == 0:
        # Two arithmetic progressions interleaved
        base = rng.randint(2, 40)
        step = rng.randint(1, 3)
        vals = set(base + i * step for i in range(N))
    elif strategy == 1:
        # Geometric-ish: powers times a base
        base = rng.randint(2, 15)
        vals = set()
        for i in range(N):
            vals.add(base * (i + 1) * rng.randint(1, 3))
    elif strategy == 2:
        # Multiples of a few shared factors
        factors = [rng.randint(2, 20) for _ in range(3)]
        vals = set()
        for i in range(N):
            f = factors[i % len(factors)]
            vals.add(f * rng.randint(1, 200))
    elif strategy == 3:
        # Small values with some structure
        base = rng.randint(1, 10)
        vals = set(base + i * rng.randint(1, 4) for i in range(N))
    elif strategy == 4:
        # Random with a common divisor
        d = rng.randint(2, 12)
        vals = set()
        for _ in range(N * 3):
            vals.add(d * rng.randint(1, MAX_VAL // d))
    else:
        # All values are multiples of a small number, with structured multipliers
        d = rng.randint(2, 10)
        # Use small multipliers to keep products small and create collisions
        mults = []
        for i in range(N):
            if i < N // 2:
                mults.append(rng.randint(1, 10))
            else:
                mults.append(rng.randint(1, 30))
        vals = set(d * m for m in mults)
    
    # Fill up to N distinct values in range
    if len(vals) < N:
        while len(vals) < N:
            vals.add(rng.randint(1, MAX_VAL))
    # Trim to N
    vals = list(vals)[:N]
    return vals

def _gen_values_random(rng):
    """Generate N distinct random values."""
    vals = set()
    while len(vals) < N:
        vals.add(rng.randint(1, MAX_VAL))
    return list(vals)

def _gen_values_common_factor(rng):
    """
    Generate values that all share a common factor, maximizing product collisions.
    If values = d * a_i, then products = d^2 * a_i * a_j.
    We want the a_i * a_j to have few distinct values.
    """
    d = rng.randint(2, 30)
    # Generate small multipliers with structure
    strategy = rng.randint(0, 3)
    if strategy == 0:
        # Arithmetic progression of multipliers
        start = rng.randint(1, 5)
        step = rng.randint(1, 3)
        mults = [start + i * step for i in range(N)]
    elif strategy == 1:
        # Powers of small numbers
        base = rng.randint(2, 4)
        mults = [base ** i for i in range(N)]
    elif strategy == 2:
        # Small random multipliers
        mults = [rng.randint(1, 50) for _ in range(N)]
    else:
        # Mix of small and medium
        mults = [rng.randint(1, 20) if i < N//2 else rng.randint(1, 100) for i in range(N)]
    
    vals = set()
    for m in mults:
        v = d * m
        if 1 <= v <= MAX_VAL:
            vals.add(v)
    
    # Fill up if needed
    if len(vals) < N:
        while len(vals) < N:
            v = d * rng.randint(1, MAX_VAL // d)
            if 1 <= v <= MAX_VAL:
                vals.add(v)
    
    return list(vals)[:N]

def _mutate_values(values, rng, num_changes):
    """Randomly change num_changes values."""
    new_vals = values[:]
    for _ in range(num_changes):
        idx = rng.randint(0, N - 1)
        new_val = rng.randint(1, MAX_VAL)
        if new_val not in new_vals:
            new_vals[idx] = new_val
    return new_vals

def _mutate_structured(values, rng):
    """
    Mutate values while preserving some structure: pick a subset and
    regenerate using a structured approach around existing values.
    """
    new_vals = values[:]
    num_changes = rng.randint(1, 4)
    for _ in range(num_changes):
        idx = rng.randint(0, N - 1)
        strategy = rng.randint(0, 3)
        if strategy == 0:
            # Scale a random existing value
            base = values[rng.randint(0, N - 1)]
            new_val = base * rng.randint(2, 4)
        elif strategy == 1:
            # Arithmetic offset from a random value
            base = values[rng.randint(0, N - 1)]
            new_val = base + rng.randint(1, 50)
        elif strategy == 2:
            # Common factor with a random value
            base = values[rng.randint(0, N - 1)]
            d = rng.randint(2, 6)
            new_val = base * d
        else:
            new_val = rng.randint(1, MAX_VAL)
        if 1 <= new_val <= MAX_VAL and new_val not in new_vals:
            new_vals[idx] = new_val
    return new_vals

def _mutate_common_factor(values, rng):
    """
    Mutate values to maintain or create a common factor structure.
    """
    new_vals = values[:]
    num_changes = rng.randint(1, 3)
    # Pick a common factor
    d = rng.randint(2, 20)
    for _ in range(num_changes):
        idx = rng.randint(0, N - 1)
        new_val = d * rng.randint(1, MAX_VAL // d)
        if 1 <= new_val <= MAX_VAL and new_val not in new_vals:
            new_vals[idx] = new_val
    return new_vals

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    rng = random.Random(12345)
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Phase 1: Generate candidate value sets, find best edges for each
    value_candidates = []
    
    # Structured generators
    for _ in range(30):
        if time.time() > deadline:
            break
        vals = _gen_values_structured(rng)
        if len(set(vals)) == N:
            value_candidates.append(vals)
    
    # Common factor generators (key insight: shared factors -> product collisions)
    for _ in range(30):
        if time.time() > deadline:
            break
        vals = _gen_values_common_factor(rng)
        if len(set(vals)) == N:
            value_candidates.append(vals)
    
    # Random generators
    for _ in range(15):
        if time.time() > deadline:
            break
        vals = _gen_values_random(rng)
        value_candidates.append(vals)
    
    # Evaluate each candidate
    for vals in value_candidates:
        if time.time() > deadline:
            break
        edges, score, ns, np_ = _greedy_edges(vals, n_starts=20, time_budget=0.3)
        if edges is None:
            continue
        if score < best_score:
            best_score = score
            best_values = vals[:]
            best_edges = edges[:]
    
    if best_values is None:
        # Fallback
        best_values = list(range(1, N + 1))
        best_edges, best_score, ns, np_ = _greedy_edges(best_values, n_starts=30, time_budget=1.0)
    
    # Phase 2: Local search on values + edges
    current_values = best_values[:]
    current_edges = best_edges[:]
    current_score = best_score
    
    temperature = 50.0
    cooling_rate = 0.97
    min_temperature = 1.0
    
    while time.time() < deadline - 10.0 and temperature > min_temperature:
        # Choose mutation strategy
        r = rng.random()
        if r < 0.4:
            new_values = _mutate_structured(current_values, rng)
        elif r < 0.7:
            new_values = _mutate_common_factor(current_values, rng)
        else:
            num_changes = rng.randint(1, 3)
            new_values = _mutate_values(current_values, rng, num_changes)
        
        if len(set(new_values)) != N:
            continue
        if not all(1 <= v <= MAX_VAL for v in new_values):
            continue
        
        # Find best edges for new values
        new_edges, new_score, ns, np_ = _greedy_edges(new_values, n_starts=15, time_budget=0.25)
        if new_edges is None:
            continue
        
        # Acceptance
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
    
    # Phase 3: Edge swap improvement on the best values
    if time.time() < deadline - 5.0:
        improved_edges, improved_score, ns, np_ = _edge_swap_improve(
            best_values, best_edges[:], max_iters=100000
        )
        if improved_score < best_score:
            best_score = improved_score
            best_edges = improved_edges
    
    # Phase 4: Final re-optimize edges for best values
    if time.time() < deadline - 2.0:
        final_edges, final_score, ns, np_ = _greedy_edges(
            best_values, n_starts=50, time_budget=3.0
        )
        if final_edges is not None and final_score < best_score:
            best_score = final_score
            best_edges = final_edges
    
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
