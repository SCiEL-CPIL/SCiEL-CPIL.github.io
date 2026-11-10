"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _raw_value(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return max(len(sums), len(prods))

def _greedy_edges(values, m):
    """Select m edges greedily to minimize max(distinct_sums, distinct_prods)."""
    n = len(values)
    sums = set()
    prods = set()
    chosen = []
    
    # Precompute all possible edges with their sum and product
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            s = values[i] + values[j]
            p = values[i] * values[j]
            all_edges.append((i, j, s, p))
    
    # Sort by: prefer edges where both sum and product already exist,
    # then by smaller sum, then smaller product
    for _ in range(m):
        best_idx = -1
        best_key = None
        for idx, (i, j, s, p) in enumerate(all_edges):
            if (i, j) in [(c[0], c[1]) for c in chosen]:
                continue
            # Score: how many new distinct values this edge adds
            new_s = 0 if s in sums else 1
            new_p = 0 if p in prods else 1
            total_new = new_s + new_p
            # Prefer edges that add fewer new values
            # Tie-break by smaller sum, then product
            key = (total_new, s, p)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx
        
        if best_idx >= 0:
            i, j, s, p = all_edges[best_idx]
            chosen.append((i, j))
            sums.add(s)
            prods.add(p)
        else:
            break
    
    return [list(e) for e in chosen]

def _local_search(values, edges, max_iters=2000):
    """Improve values via local search."""
    best_vals = list(values)
    best_raw = _raw_value(best_vals, edges)
    n = len(best_vals)
    
    for it in range(max_iters):
        # Pick a random vertex and try changing its value
        vi = random.randint(0, n - 1)
        old_val = best_vals[vi]
        # Try nearby values
        candidates = []
        for delta in range(-20, 21):
            nv = old_val + delta
            if 1 <= nv <= 10000 and nv != old_val:
                candidates.append(nv)
        # Also try some random values
        for _ in range(5):
            nv = random.randint(1, 10000)
            if nv != old_val:
                candidates.append(nv)
        
        for nv in candidates:
            best_vals[vi] = nv
            # Check distinctness
            if len(set(best_vals)) < n:
                best_vals[vi] = old_val
                continue
            raw = _raw_value(best_vals, edges)
            if raw < best_raw:
                best_raw = raw
                break
            best_vals[vi] = old_val
        else:
            continue
        # Accept if improved
        if best_raw < _raw_value(best_vals, edges):
            pass  # already set
    
    return best_vals, best_raw

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s limit
    
    best_result = None
    best_raw = float('inf')
    
    # Strategy 1: Powers of 2 for multiplicative structure
    # 2^0, 2^1, ..., 2^13 -> products are 2^(a+b), sums are 2^a + 2^b
    # Try different base values
    for base in [2, 3]:
        if time.time() > deadline:
            break
        values = [base ** i for i in range(14)]
        # Check max value
        if max(values) > 10000:
            continue
        edges = _greedy_edges(values, 35)
        raw = _raw_value(values, edges)
        if raw < best_raw:
            best_raw = raw
            best_result = (list(values), [list(e) for e in edges])
    
    # Strategy 2: Mixed approach - small values with some structure
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Generate values: mix of small numbers
        # Use a geometric-like sequence
        vals = []
        for i in range(14):
            v = random.randint(1, 50)
            while v in vals:
                v = random.randint(1, 50)
            vals.append(v)
        
        edges = _greedy_edges(vals, 35)
        raw = _raw_value(vals, edges)
        if raw < best_raw:
            best_raw = raw
            best_result = (list(vals), [list(e) for e in edges])
        
        # Local search on values
        improved_vals, improved_raw = _local_search(vals, edges, max_iters=500)
        if improved_raw < best_raw:
            best_raw = improved_raw
            best_result = (list(improved_vals), [list(e) for e in edges])
    
    # Strategy 3: More structured - use values that create many sum collisions
    # Try arithmetic progressions and their variants
    for start in [1, 2, 3, 5]:
        if time.time() > deadline:
            break
        for step in [1, 2, 3, 4, 5]:
            vals = [start + i * step for i in range(14)]
            if max(vals) > 10000:
                continue
            if len(set(vals)) < 14:
                continue
            edges = _greedy_edges(vals, 35)
            raw = _raw_value(vals, edges)
            if raw < best_raw:
                best_raw = raw
                best_result = (list(vals), [list(e) for e in edges])
    
    # Final: try to improve best result with more local search
    if best_result and time.time() < deadline - 5:
        vals, edges = best_result
        improved_vals, improved_raw = _local_search(vals, edges, max_iters=3000)
        if improved_raw < best_raw:
            best_raw = improved_raw
            best_result = (list(improved_vals), [list(e) for e in edges])
    
    if best_result is None:
        # Fallback
        values = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        best_result = (values, edges)
    
    values, edges = best_result
    # Ensure correct types
    return {'values': [int(v) for v in values], 'edges': [[int(i), int(j)] for i, j in edges]}
# EVOLVE-BLOCK-END
