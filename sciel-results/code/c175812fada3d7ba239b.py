"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

DEADLINE = 85.0

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

def _greedy_edge_selection(values, target_m):
    """Greedy: pick edges that minimize growth in distinct sums and products."""
    n = len(values)
    all_e = _all_edges(n)
    random.shuffle(all_e)
    sums = set()
    prods = set()
    edges = []
    for e in all_e:
        if len(edges) >= target_m:
            break
        u, v = e
        a, b = values[u], values[v]
        new_sums = len(sums | {a + b})
        new_prods = len(prods | {a * b})
        cost = max(new_sums, new_prods)
        # Accept if it doesn't increase the max too much
        if cost <= max(len(sums), len(prods)) + 1:
            sums.add(a + b)
            prods.add(a * b)
            edges.append([u, v])
    # If we didn't get enough edges, fill with remaining
    if len(edges) < target_m:
        existing = set(tuple(sorted(e)) for e in edges)
        for e in all_e:
            if len(edges) >= target_m:
                break
            if tuple(sorted(e)) not in existing:
                u, v = e
                sums.add(values[u] + values[v])
                prods.add(values[u] * values[v])
                edges.append([u, v])
                existing.add(tuple(sorted(e)))
    return edges

def _local_search(values, edges, time_limit):
    """Local search: try swapping edges to reduce objective."""
    n = len(values)
    all_e = _all_edges(n)
    start_time = time.time()
    current_edges = [list(e) for e in edges]
    current_score = _raw_score(values, current_edges)
    best_edges = [list(e) for e in current_edges]
    best_score = current_score
    edge_set = set(tuple(sorted(e)) for e in current_edges)
    
    while time.time() - start_time < time_limit:
        # Try random edge swaps
        for _ in range(500):
            if time.time() - start_time > time_limit:
                break
            # Pick a random existing edge to remove
            idx = random.randint(0, len(current_edges) - 1)
            old_edge = current_edges[idx]
            old_key = tuple(sorted(old_edge))
            
            # Pick a random non-existing edge to add
            new_edge = random.choice(all_e)
            new_key = tuple(sorted(new_edge))
            if new_key == old_key or new_key in edge_set:
                continue
            
            # Perform swap
            current_edges[idx] = list(new_edge)
            edge_set.discard(old_key)
            edge_set.add(new_key)
            
            new_score = _raw_score(values, current_edges)
            if new_score < best_score:
                best_score = new_score
                best_edges = [list(e) for e in current_edges]
            else:
                # Revert
                current_edges[idx] = list(old_edge)
                edge_set.discard(new_key)
                edge_set.add(old_key)
    
    return best_edges, best_score

def construct():
    n = 14
    m = 35
    max_val = 10000
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    # Try multiple random value assignments
    for attempt in range(20):
        if time.time() - start_time > DEADLINE:
            break
        
        # Generate distinct positive integers
        # Use small values to keep sums and products manageable
        if attempt == 0:
            values = list(range(1, n + 1))
        else:
            # Random selection from small range
            values = random.sample(range(1, max_val + 1), n)
            random.shuffle(values)
        
        # Greedy edge selection
        edges = _greedy_edge_selection(values, m)
        
        # Local search with remaining time budget
        time_left = DEADLINE - (time.time() - start_time)
        search_time = min(3.0, time_left * 0.4)
        
        if search_time > 0.5:
            edges, score = _local_search(values, edges, search_time)
        else:
            score = _raw_score(values, edges)
        
        if score < best_score:
            best_score = score
            best_result = {'values': values, 'edges': edges}
    
    # Ensure we have a valid result
    if best_result is None:
        values = list(range(1, n + 1))
        all_e = _all_edges(n)
        edges = [list(e) for e in all_e[:m]]
        best_result = {'values': values, 'edges': edges}
    
    # Final validation
    vals = best_result['values']
    eds = best_result['edges']
    
    # Check distinctness
    assert len(set(vals)) == n, "Values must be distinct"
    assert all(isinstance(v, int) for v in vals), "Values must be int"
    assert all(1 <= v <= max_val for v in vals), "Values out of range"
    
    # Check edges
    assert len(eds) == m, f"Expected {m} edges, got {len(eds)}"
    edge_tuples = set()
    for e in eds:
        assert len(e) == 2, "Edge must have 2 elements"
        i, j = e
        assert 0 <= i < n and 0 <= j < n, "Edge indices out of range"
        assert i != j, "No loops allowed"
        key = tuple(sorted([i, j]))
        assert key not in edge_tuples, "Duplicate edge"
        edge_tuples.add(key)
    
    # Convert to ensure proper types
    best_result['values'] = [int(v) for v in vals]
    best_result['edges'] = [[int(i), int(j)] for i, j in eds]
    
    return best_result
# EVOLVE-BLOCK-END
