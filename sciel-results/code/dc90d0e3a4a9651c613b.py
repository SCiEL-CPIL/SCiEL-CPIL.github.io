"""Rewritten program: joint optimization of values and edges for sum/product collisions."""
import time
import random
from itertools import combinations

_N = 14
_M = 35
_DEADLINE = 80.0  # seconds (safety margin under 90 s kill)


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


def _random_prime_power_set(rng, n):
    """Generate a set of n distinct prime powers <= 10000."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    # Build candidate pool of prime powers
    pool = set()
    for p in primes:
        x = p
        while x <= 10000:
            pool.add(x)
            x *= p
    pool = list(pool)
    if len(pool) < n:
        return None
    chosen = rng.sample(pool, n)
    return sorted(chosen)


def _random_structured_set(rng, n):
    """Generate a set with many sum collisions: arithmetic-ish + duplicates of pair sums."""
    # Base: small integers with some structure
    base = list(range(1, n + 1))
    # Perturb to create sum collisions
    for _ in range(3):
        i, j = rng.sample(range(n), 2)
        # Make values[i] + values[j] match another pair
        k, l = rng.sample(range(n), 2)
        if i != k and i != l and j != k and j != l:
            base[i] = base[k] + base[l] - base[j]
            if base[i] < 1:
                base[i] = 1
            if base[i] > 10000:
                base[i] = rng.randint(1, 10000)
    # Ensure distinctness
    seen = set()
    result = []
    for v in base:
        if v in seen:
            v = rng.randint(1, 10000)
        seen.add(v)
        result.append(v)
    # Ensure all distinct
    while len(set(result)) < n:
        i = rng.randint(0, n - 1)
        result[i] = rng.randint(1, 10000)
    return sorted(result)


def _random_set(rng, n):
    """Random distinct positive integers <= 10000."""
    while True:
        vals = rng.sample(range(1, 10001), n)
        if len(set(vals)) == n:
            return sorted(vals)


def _sa_edge_select(values, rng, time_limit, max_iter=200000):
    """Simulated annealing to select _M edges minimizing raw_value."""
    all_edges = _all_pairs(_N)
    n_edges = len(all_edges)
    
    # Precompute sums and products for all edges
    edge_sum = {}
    edge_prod = {}
    for u, v in all_edges:
        key = (u, v)
        edge_sum[key] = values[u] + values[v]
        edge_prod[key] = values[u] * values[v]
    
    # Initial: random selection of _M edges
    initial = rng.sample(all_edges, _M)
    best_edges = initial[:]
    best_score = _raw_value(values, best_edges)
    
    current_edges = initial[:]
    current_score = best_score
    current_set = set(tuple(e) for e in current_edges)
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9999
    
    start = time.time()
    iter_count = 0
    while T > T_min and (time.time() - start) < time_limit:
        iter_count += 1
        # Pick an edge in current and an edge not in current, swap
        if len(current_edges) < _M:
            break
        i = rng.randint(0, len(current_edges) - 1)
        out_edge = current_edges[i]
        # Pick a random edge not in current
        while True:
            in_edge = all_edges[rng.randint(0, n_edges - 1)]
            if tuple(in_edge) not in current_set:
                break
        
        # Compute new score
        new_set = current_set - {tuple(out_edge)} | {tuple(in_edge)}
        new_sums = set()
        new_prods = set()
        for e in new_set:
            new_sums.add(edge_sum[e])
            new_prods.add(edge_prod[e])
        new_score = max(len(new_sums), len(new_prods))
        
        delta = new_score - current_score
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
            # Accept
            current_set = new_set
            current_edges[i] = in_edge
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = [list(e) for e in new_set]
        
        T *= cooling
        # Check time
        if iter_count % 10000 == 0:
            if (time.time() - start) >= time_limit:
                break
    
    return best_edges, best_score


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_result = None
    best_raw = float('inf')
    
    # Phase 1: Try various value sets
    value_candidates = []
    
    # Prime power sets (good for product collisions)
    for _ in range(20):
        vset = _random_prime_power_set(rng, _N)
        if vset and len(set(vset)) == _N:
            value_candidates.append(vset)
    
    # Structured sets (good for sum collisions)
    for _ in range(20):
        vset = _random_structured_set(rng, _N)
        if len(set(vset)) == _N:
            value_candidates.append(vset)
    
    # Random sets
    for _ in range(10):
        vset = _random_set(rng, _N)
        value_candidates.append(vset)
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for v in value_candidates:
        key = tuple(sorted(v))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(v)
    
    # Phase 2: For each value set, run SA edge selection
    for idx, values in enumerate(unique_candidates):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 5:
            break
        
        time_left = _DEADLINE - elapsed - 2  # safety margin
        if time_left < 0.5:
            break
        
        # Limit time per value set
        per_set_time = min(3.0, time_left / max(1, len(unique_candidates) - idx))
        
        edges, score = _sa_edge_select(values, rng, per_set_time, max_iter=100000)
        
        if score < best_raw:
            best_raw = score
            best_result = {
                'values': [int(v) for v in values],
                'edges': [list(e) for e in edges]
            }
    
    # If no valid result found, use default
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
    
    assert len(vals) == _N, f"Expected {_N} values, got {len(vals)}"
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in vals)
    assert all(1 <= v <= 10000 for v in vals)
    assert len(set(vals)) == _N, "Values must be distinct"
    
    assert len(edges) == _M, f"Expected {_M} edges, got {len(edges)}"
    edge_set = set()
    for e in edges:
        assert len(e) == 2
        i, j = e
        assert isinstance(i, int) and isinstance(j, int)
        assert 0 <= i < _N and 0 <= j < _N
        assert i != j, "No loops"
        key = (min(i, j), max(i, j))
        assert key not in edge_set, "No duplicate edges"
        edge_set.add(key)
    
    return {
        'values': [int(v) for v in vals],
        'edges': [[int(i), int(j)] for i, j in edges]
    }
