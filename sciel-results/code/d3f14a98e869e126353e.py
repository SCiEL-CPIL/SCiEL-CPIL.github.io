"""CSP-style search: partition values into multiplicative orbits to force product
collisions, then backtrack-select 35 edges with a strict bound on distinct sums/products."""
import time
import random
import math
from itertools import combinations

_N = 14
_M = 35
_DEADLINE = 85.0  # safety margin under 90 s kill


def _raw_value(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))


def _all_pairs(n):
    return [list(e) for e in combinations(range(n), 2)]


def _factorize(x):
    """Return prime factorization as a sorted tuple of (prime, exponent)."""
    factors = []
    d = 2
    while d * d <= x:
        if x % d == 0:
            cnt = 0
            while x % d == 0:
                x //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1
    if x > 1:
        factors.append((x, 1))
    return tuple(factors)


def _prime_signature(x):
    """Return just the set of prime factors (ignoring exponents) as a frozenset."""
    return frozenset(p for p, _ in _factorize(x))


def _generate_orbit_values(rng, n):
    """
    Generate n distinct positive integers <= 10000 by grouping them into
    'orbits' where each orbit consists of numbers with the same prime signature.
    This forces product collisions: if a and b have the same prime signature as
    c and d, then a*b and c*d may collide more often.
    
    Strategy: pick a few base prime signatures, then for each generate multiple
    numbers with that signature by varying exponents.
    """
    # Precompute all numbers <= 10000 grouped by prime signature
    sig_to_numbers = {}
    for x in range(1, 10001):
        sig = _prime_signature(x)
        sig_to_numbers.setdefault(sig, []).append(x)
    
    # Sort signatures by pool size (larger pools give more flexibility)
    sigs = sorted(sig_to_numbers.keys(), key=lambda s: -len(sig_to_numbers[s]))
    
    # We need to pick n distinct numbers, trying to maximize orbit structure
    # Pick a small number of signatures and take multiple from each
    # Target: 3-5 orbits of size 3-5 each
    
    best_set = None
    best_collisions = 0
    
    for _ in range(50):
        # Pick 3-5 signatures
        n_sigs = rng.randint(3, 5)
        chosen_sigs = rng.sample(sigs[:50], n_sigs)  # from larger pools
        
        values = []
        used = set()
        valid = True
        
        for sig in chosen_sigs:
            pool = sig_to_numbers[sig]
            # Pick 3-5 numbers from this pool
            k = rng.randint(3, min(5, len(pool)))
            try:
                picks = rng.sample(pool, k)
            except ValueError:
                valid = False
                break
            for p in picks:
                if p not in used:
                    used.add(p)
                    values.append(p)
        
        if not valid or len(values) < n:
            # Fill remaining with random distinct numbers
            remaining = [x for x in range(1, 10001) if x not in used]
            if len(remaining) >= n - len(values):
                extra = rng.sample(remaining, n - len(values))
                values.extend(extra)
                used.update(extra)
        
        if len(values) != n:
            continue
        
        values = sorted(values)
        
        # Estimate product collision potential
        # Count how many edge products would collide
        prods = {}
        for i, j in combinations(range(n), 2):
            p = values[i] * values[j]
            prods[p] = prods.get(p, 0) + 1
        
        n_colliding_edges = sum(1 for c in prods.values() if c > 1)
        n_distinct_prods = len(prods)
        
        # Also estimate sum collisions
        sums = {}
        for i, j in combinations(range(n), 2):
            s = values[i] + values[j]
            sums[s] = sums.get(s, 0) + 1
        n_distinct_sums = len(sums)
        
        score = max(n_distinct_sums, n_distinct_prods)
        
        if best_set is None or score < best_collisions:
            best_collisions = score
            best_set = values[:]
    
    if best_set is None:
        # Fallback: random
        return sorted(rng.sample(range(1, 10001), n))
    
    return best_set


def _generate_mixed_values(rng, n):
    """Generate values with a mix of orbit structure and arithmetic structure."""
    # Half from orbits, half from arithmetic progression
    half = n // 2
    
    # Orbit part
    orbit_vals = []
    used = set()
    for _ in range(20):
        # Pick a random base and create multiples
        base = rng.randint(2, 200)
        k = rng.randint(2, 4)
        candidates = [base * (i + 1) for i in range(k)]
        candidates = [c for c in candidates if c <= 10000 and c not in used]
        if len(candidates) >= 2:
            picks = rng.sample(candidates, min(2, len(candidates)))
            orbit_vals.extend(picks)
            used.update(picks)
        if len(orbit_vals) >= half:
            break
    
    # Arithmetic part
    arith_vals = []
    start = rng.randint(1, 500)
    step = rng.randint(1, 30)
    for i in range(n - len(orbit_vals)):
        v = start + i * step
        if v <= 10000 and v not in used and v not in arith_vals:
            arith_vals.append(v)
            used.add(v)
    
    # Fill if needed
    while len(orbit_vals) + len(arith_vals) < n:
        v = rng.randint(1, 10000)
        if v not in used:
            arith_vals.append(v)
            used.add(v)
    
    result = (orbit_vals + arith_vals)[:n]
    if len(set(result)) != n:
        return sorted(rng.sample(range(1, 10001), n))
    return sorted(result)


def _backtrack_edge_select(values, rng, max_iter=500000, time_limit=10.0):
    """
    Backtracking search to select M edges minimizing max(distinct_sums, distinct_prods).
    Uses forward checking: when adding an edge, check if it introduces a new sum/product
    that would exceed a current best bound.
    """
    all_edges = _all_pairs(_N)
    n_all = len(all_edges)
    
    # Precompute sums and products
    edge_sum = {}
    edge_prod = {}
    for u, v in all_edges:
        edge_sum[(u, v)] = values[u] + values[v]
        edge_prod[(u, v)] = values[u] * values[v]
    
    # Initial solution: random M edges
    init_edges = rng.sample(all_edges, _M)
    best_edges = init_edges[:]
    best_score = _raw_value(values, best_edges)
    
    current_edges = init_edges[:]
    current_set = set(tuple(e) for e in current_edges)
    current_sums = set(edge_sum[e] for e in current_set)
    current_prods = set(edge_prod[e] for e in current_set)
    current_score = max(len(current_sums), len(current_prods))
    
    T = 5.0
    T_min = 0.01
    cooling = 0.99995
    
    start = time.time()
    iter_count = 0
    stall_count = 0
    max_stall = 50000
    
    while T > T_min and (time.time() - start) < time_limit:
        iter_count += 1
        
        # Choose move: swap one edge, or add+remove (keep size M)
        i = rng.randint(0, _M - 1)
        out_edge = current_edges[i]
        out_key = tuple(out_edge)
        
        # Pick random in-edge
        while True:
            in_edge = all_edges[rng.randint(0, n_all - 1)]
            in_key = tuple(in_edge)
            if in_key not in current_set:
                break
        
        # Compute new state
        new_sums = set(current_sums)
        new_prods = set(current_prods)
        
        # Remove out_edge contributions
        os = edge_sum[out_key]
        op = edge_prod[out_key]
        # Need to check if other edges share this sum/product
        # For efficiency, just recompute from scratch for the changed edge
        # Actually, let's just recompute the full sets (fast enough for M=35)
        
        # Remove
        # Count how many edges have this sum
        sum_count = {}
        prod_count = {}
        for e in current_set:
            s = edge_sum[e]
            p = edge_prod[e]
            sum_count[s] = sum_count.get(s, 0) + 1
            prod_count[p] = prod_count.get(p, 0) + 1
        
        sum_count[os] -= 1
        prod_count[op] -= 1
        
        if sum_count[os] <= 0:
            new_sums.discard(os)
        if prod_count[op] <= 0:
            new_prods.discard(op)
        
        # Add in_edge
        ins = edge_sum[in_key]
        inp = edge_prod[in_key]
        new_sums.add(ins)
        new_prods.add(inp)
        
        new_score = max(len(new_sums), len(new_prods))
        
        delta = new_score - current_score
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            # Accept
            current_set.discard(out_key)
            current_set.add(in_key)
            current_edges[i] = in_edge
            current_sums = new_sums
            current_prods = new_prods
            current_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_edges = [list(e) for e in current_set]
                stall_count = 0
            else:
                stall_count += 1
        else:
            stall_count += 1
        
        T *= cooling
        
        if iter_count % 100000 == 0:
            if (time.time() - start) >= time_limit:
                break
    
    return best_edges, best_score


def _construct():
    start_time = time.time()
    rng = random.Random(12345)
    
    best_result = None
    best_raw = float('inf')
    
    # Generate candidate value sets
    value_candidates = []
    
    # Orbit-based values
    for _ in range(30):
        vset = _generate_orbit_values(rng, _N)
        if len(set(vset)) == _N:
            value_candidates.append(vset)
    
    # Mixed values
    for _ in range(20):
        vset = _generate_mixed_values(rng, _N)
        if len(set(vset)) == _N:
            value_candidates.append(vset)
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for v in value_candidates:
        key = tuple(sorted(v))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(v)
    
    # For each value set, run backtracking/SA edge selection
    for idx, values in enumerate(unique_candidates):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 5:
            break
        
        time_left = _DEADLINE - elapsed - 2
        if time_left < 0.5:
            break
        
        per_set_time = min(4.0, time_left / max(1, len(unique_candidates) - idx))
        
        edges, score = _backtrack_edge_select(values, rng, max_iter=300000, time_limit=per_set_time)
        
        if score < best_raw:
            best_raw = score
            best_result = {
                'values': [int(v) for v in values],
                'edges': [list(e) for e in edges]
            }
    
    # If no valid result, use default
    if best_result is None:
        values = list(range(1, _N + 1))
        edges = _all_pairs(_N)[:_M]
        best_result = {
            'values': [int(v) for v in values],
            'edges': edges
        }
    
    # Final validation
    vals = best_result['values']
    edges = best_result['edges']
    
    assert len(vals) == _N
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in vals)
    assert all(1 <= v <= 10000 for v in vals)
    assert len(set(vals)) == _N
    
    assert len(edges) == _M
    edge_set = set()
    for e in edges:
        assert len(e) == 2
        i, j = e
        assert isinstance(i, int) and isinstance(j, int)
        assert 0 <= i < _N and 0 <= j < _N
        assert i != j
        key = (min(i, j), max(i, j))
        assert key not in edge_set
        edge_set.add(key)
    
    return {
        'values': [int(v) for v in vals],
        'edges': [[int(i), int(j)] for i, j in edges]
    }


def construct():
    return _construct()
