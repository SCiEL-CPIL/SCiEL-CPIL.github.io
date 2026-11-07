"""Rewritten program: optimize both values and edges to minimize distinct sums/products."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _raw_value(values, edges):
    """Compute max(distinct_sums, distinct_products) for given values and edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _evaluate(values, edges):
    """Return (raw_value, num_sums, num_prods)."""
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods)), len(sums), len(prods)

def _optimize_edges_for_values(values, m=35, deadline=None, seed=42):
    """Find m edges minimizing max(distinct_sums, distinct_products)."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    rng = random.Random(seed)
    
    # Precompute sums and products for all edges
    edge_sums = {}
    edge_prods = {}
    for e in all_edges:
        edge_sums[e] = values[e[0]] + values[e[1]]
        edge_prods[e] = values[e[0]] * values[e[1]]
    
    best_raw = float('inf')
    best_edges = None
    best_sums = None
    best_prods = None
    
    # Simulated annealing on edge selection
    # State: set of m edges
    # Move: swap one edge in with one edge out
    
    # Initial solution: greedy - pick edges that minimize new distinct sums/products
    current_edges = set()
    current_sums = set()
    current_prods = set()
    
    # Sort edges by how "common" their sum/product is (lower rank = more common)
    sum_counts = {}
    prod_counts = {}
    for e in all_edges:
        s = edge_sums[e]
        p = edge_prods[e]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    # Greedy initialization
    remaining = list(all_edges)
    rng.shuffle(remaining)
    for e in remaining:
        if len(current_edges) >= m:
            break
        s = edge_sums[e]
        p = edge_prods[e]
        # Prefer edges whose sum/product are already in our sets
        if s in current_sums and p in current_prods:
            current_edges.add(e)
        elif s in current_sums:
            current_edges.add(e)
        elif p in current_prods:
            current_edges.add(e)
    
    # Fill if needed
    for e in remaining:
        if len(current_edges) >= m:
            break
        if e not in current_edges:
            current_edges.add(e)
    
    # If still not enough, add more
    if len(current_edges) < m:
        for e in all_edges:
            if len(current_edges) >= m:
                break
            if e not in current_edges:
                current_edges.add(e)
    
    # Recompute sets
    current_sums = set()
    current_prods = set()
    for e in current_edges:
        current_sums.add(edge_sums[e])
        current_prods.add(edge_prods[e])
    
    current_raw = max(len(current_sums), len(current_prods))
    best_raw = current_raw
    best_edges = set(current_edges)
    best_sums = set(current_sums)
    best_prods = set(current_prods)
    
    # Simulated annealing
    if deadline is not None:
        start_time = time.time()
    
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    
    while T > min_T:
        if deadline is not None and time.time() - start_time > deadline - 5:
            break
        
        # Pick a random edge in and a random edge out
        in_edge = rng.choice(all_edges)
        if in_edge in current_edges:
            continue
        
        out_edge = rng.choice(list(current_edges))
        
        # Evaluate the swap
        new_sums = set(current_sums)
        new_prods = set(current_prods)
        
        # Remove out_edge's contributions (need to check if other edges have same sum/prod)
        out_s = edge_sums[out_edge]
        out_p = edge_prods[out_edge]
        
        # Count how many edges in current_edges have this sum/prod
        sum_count = 0
        prod_count = 0
        for e in current_edges:
            if e == out_edge:
                continue
            if edge_sums[e] == out_s:
                sum_count += 1
            if edge_prods[e] == out_p:
                prod_count += 1
        
        new_sums = set(current_sums)
        new_prods = set(current_prods)
        
        if sum_count == 0:
            new_sums.discard(out_s)
        if prod_count == 0:
            new_prods.discard(out_p)
        
        # Add in_edge's contributions
        in_s = edge_sums[in_edge]
        in_p = edge_prods[in_edge]
        new_sums.add(in_s)
        new_prods.add(in_p)
        
        new_raw = max(len(new_sums), len(new_prods))
        
        if new_raw <= current_raw or (T > min_T and random.random() < (new_raw - current_raw) / (-T)):
            # Accept
            current_edges.discard(out_edge)
            current_edges.add(in_edge)
            current_sums = new_sums
            current_prods = new_prods
            current_raw = new_raw
            
            if current_raw < best_raw:
                best_raw = current_raw
                best_edges = set(current_edges)
                best_sums = set(current_sums)
                best_prods = set(current_prods)
        
        T *= cooling
    
    return best_raw, list(best_edges)

def _optimize_values(deadline, seed=0):
    """Optimize values using local search."""
    n = 14
    m = 35
    
    best_overall_raw = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    rng = random.Random(seed)
    
    # Start with a reasonable initial set
    # Use geometric progression or similar to keep products close
    base_values = list(range(1, n + 1))
    
    start_time = time.time()
    max_time = deadline - 10  # Safety margin
    
    # Try multiple random restarts
    num_restarts = 0
    while time.time() - start_time < max_time:
        num_restarts += 1
        
        # Generate random values
        if num_restarts == 1:
            values = base_values[:]
        else:
            # Random values, biased toward smaller numbers to keep products small
            values = []
            for i in range(n):
                if i == 0:
                    values.append(1)
                else:
                    # Geometric-like with noise
                    base = int(1.3 ** i)
                    noise = rng.randint(-base // 4, base // 4)
                    v = max(1, base + noise)
                    v = min(v, 10000)
                    # Ensure distinctness
                    while v in values:
                        v = max(1, v + rng.randint(-2, 2))
                        v = min(v, 10000)
                    values.append(v)
        
        # Ensure all distinct and positive
        if len(set(values)) != n or any(v <= 0 or v > 10000 for v in values):
            continue
        
        # Optimize edges for these values
        remaining_time = max_time - (time.time() - start_time)
        if remaining_time < 2:
            break
        
        raw, edges = _optimize_edges_for_values(values, m, deadline=remaining_time, seed=seed + num_restarts)
        
        if raw < best_overall_raw:
            best_overall_raw = raw
            best_overall_values = values[:]
            best_overall_edges = edges[:]
        
        # Local search on values
        if time.time() - start_time < max_time - 5:
            improved = True
            while improved and time.time() - start_time < max_time - 5:
                improved = False
                # Try modifying one value at a time
                for idx in range(n):
                    if time.time() - start_time > max_time - 5:
                        break
                    orig_val = values[idx]
                    # Try a few nearby values
                    for delta in [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10]:
                        new_val = orig_val + delta
                        if new_val <= 0 or new_val > 10000:
                            continue
                        if new_val in values and new_val != orig_val:
                            continue
                        values[idx] = new_val
                        raw2, edges2 = _optimize_edges_for_values(values, m, deadline=1, seed=seed + num_restarts)
                        if raw2 < best_overall_raw:
                            best_overall_raw = raw2
                            best_overall_values = values[:]
                            best_overall_edges = edges2[:]
                            improved = True
                        else:
                            values[idx] = orig_val
                    values[idx] = orig_val
    
    return best_overall_raw, best_overall_values, best_overall_edges

def construct():
    deadline = 85  # Safety margin from 90s limit
    
    start = time.time()
    
    best_raw = float('inf')
    best_values = None
    best_edges = None
    
    # First, try to optimize values and edges together
    try:
        raw, values, edges = _optimize_values(deadline, seed=42)
        if values is not None and edges is not None:
            best_raw = raw
            best_values = values
            best_edges = edges
    except Exception:
        pass
    
    # If we didn't get a good result, fall back to simple approach
    if best_values is None or best_edges is None:
        values = list(range(1, 15))
        all_edges = list(combinations(range(14), 2))[:35]
        edges = [list(e) for e in all_edges]
        best_raw = _raw_value(values, edges)
        best_values = values
        best_edges = edges
    
    # Final validation and type conversion
    values = [int(v) for v in best_values]
    edges = [[int(i), int(j)] for i, j in best_edges]
    
    # Ensure exactly 35 edges
    if len(edges) > 35:
        edges = edges[:35]
    elif len(edges) < 35:
        all_edges = list(combinations(range(14), 2))
        existing = set(tuple(e) for e in edges)
        for e in all_edges:
            if len(edges) >= 35:
                break
            if e not in existing:
                edges.append(list(e))
    
    # Final check
    raw = _raw_value(values, edges)
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
