"""Search-based: minimize max(distinct_sums, distinct_products) over 35 edges of 14 vertices."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

DEADLINE = 85.0  # safety margin before 90s kill


def _raw_score(values, edges):
    """Compute max(#distinct sums, #distinct products) over edges."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))


def _all_edges(n):
    return list(itertools.combinations(range(n), 2))


def _greedy_edges(values, target_m):
    """Greedy edge selection: pick edges that minimize growth in distinct sums and products."""
    n = len(values)
    all_e = _all_edges(n)
    sums = set()
    prods = set()
    edges = []
    
    # Precompute sum and product for each edge
    edge_data = []
    for u, v in all_e:
        s = values[u] + values[v]
        p = values[u] * values[v]
        edge_data.append((u, v, s, p))
    
    remaining = list(range(len(edge_data)))
    
    for _ in range(target_m):
        if not remaining:
            break
        best_idx = None
        best_score = None
        for idx in remaining:
            u, v, s, p = edge_data[idx]
            # Count new distinct sums and products
            new_sums = 0 if s in sums else 1
            new_prods = 0 if p in prods else 1
            score = new_sums + new_prods
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        u, v, s, p = edge_data[best_idx]
        edges.append([u, v])
        sums.add(s)
        prods.add(p)
        remaining.remove(best_idx)
    
    return edges


def _try_complete_graph(values):
    """Try the complete graph (all 91 edges, but we only need 35)."""
    n = len(values)
    edges = _greedy_edges(values, 35)
    return _raw_score(values, edges)


def _generate_values_ap(start, step, n=14):
    """Generate arithmetic progression values."""
    return [start + i * step for i in range(n)]


def _generate_values_gcd(start, step, n=14):
    """Generate values with common factor to reduce distinct products."""
    return [start + i * step for i in range(n)]


def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    # Strategy 1: Arithmetic progressions with various parameters
    # Try different starts and steps
    for start in range(1, 50):
        for step in range(1, 500):
            if time.time() - start_time > DEADLINE - 5:
                break
            values = _generate_values_ap(start, step, 14)
            # Check all values are <= 10000 and distinct positive
            if any(v < 1 or v > 10000 for v in values):
                continue
            if len(set(values)) != 14:
                continue
            
            edges = _greedy_edges(values, 35)
            score = _raw_score(values, edges)
            
            if score < best_score:
                best_score = score
                best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
        
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Strategy 2: Try values with a common factor (e.g., all even, all multiples of 3)
    for base in range(2, 20):
        for step in range(1, 200):
            if time.time() - start_time > DEADLINE - 5:
                break
            values = [base * (1 + i * step) for i in range(14)]
            if any(v < 1 or v > 10000 for v in values):
                continue
            if len(set(values)) != 14:
                continue
            
            edges = _greedy_edges(values, 35)
            score = _raw_score(values, edges)
            
            if score < best_score:
                best_score = score
                best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
        
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Strategy 3: Random search with local improvement
    if time.time() - start_time < DEADLINE - 10:
        random.seed(42)
        for trial in range(500):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Generate random values
            values = random.sample(range(1, 10001), 14)
            values.sort()
            
            edges = _greedy_edges(values, 35)
            score = _raw_score(values, edges)
            
            if score < best_score:
                best_score = score
                best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
    
    # Ensure we have a valid result
    if best_result is None:
        # Fallback: simple arithmetic progression
        values = list(range(1, 15))
        edges = _greedy_edges(values, 35)
        best_result = {'values': values, 'edges': [list(e) for e in edges]}
    
    # Final validation
    vals = best_result['values']
    eds = best_result['edges']
    assert len(vals) == 14
    assert len(set(vals)) == 14
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in vals)
    assert all(1 <= v <= 10000 for v in vals)
    assert len(eds) == 35
    assert all(len(e) == 2 for e in eds)
    assert all(isinstance(u, int) and isinstance(v, int) for e in eds for u, v in e)
    assert all(0 <= u < 14 and 0 <= v < 14 for e in eds for u, v in e)
    assert all(u != v for e in eds for u, v in e)
    edge_set = set(tuple(sorted(e)) for e in eds)
    assert len(edge_set) == 35
    
    return best_result

# EVOLVE-BLOCK-END
