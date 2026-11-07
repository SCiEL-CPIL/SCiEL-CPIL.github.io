"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 14
M = 35
MAX_VAL = 10000
DEADLINE = 75.0  # safety margin before 90s kill

def _raw_value(values, edges):
    """Compute max(distinct_sums, distinct_products) for given values and edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def _optimize_edges_for_values(values, deadline):
    """Find 35 edges that minimize max(distinct_sums, distinct_products) for given values."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sum and product for each edge
    edge_data = []
    for i, j in all_edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_data.append((s, p, i, j))
    
    # Sort edges by how "common" their sum and product are
    # Count frequency of each sum and product
    sum_counts = {}
    prod_counts = {}
    for s, p, i, j in edge_data:
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    # Score each edge: prefer edges with common sums and products
    def edge_score(s, p):
        return -sum_counts.get(s, 0) * 2 - prod_counts.get(p, 0) * 2
    
    scored_edges = [(edge_score(s, p), s, p, i, j) for s, p, i, j in edge_data]
    scored_edges.sort(key=lambda x: x[0])
    
    # Start with top M edges
    best_edges = [list(scored_edges[k][3:]) for k in range(M)]
    best_raw = _raw_value(values, best_edges)
    
    # Local search: try swapping edges
    in_set = set(tuple(e) for e in best_edges)
    not_in_set = [e for e in scored_edges if tuple(e[3:]) not in in_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(not_in_set)
        for out_idx, out_edge in enumerate(best_edges):
            if time.time() > deadline:
                break
            out_key = tuple(out_edge)
            for in_edge in not_in_set:
                in_key = (in_edge[3], in_edge[4])
                if in_key == out_key or in_key == tuple(reversed(out_key)):
                    continue
                # Try swap
                new_edges = best_edges.copy()
                new_edges[out_idx] = list(in_key)
                new_raw = _raw_value(values, new_edges)
                if new_raw < best_raw:
                    best_raw = new_raw
                    best_edges = new_edges
                    in_set.discard(out_key)
                    in_set.add(in_key)
                    not_in_set.remove(in_edge)
                    improved = True
                    break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    return best_edges, best_raw

def _generate_value_set(deadline, rng):
    """Generate a random set of 14 distinct positive integers <= 10000."""
    # Try different strategies
    strategy = rng.random()
    
    if strategy < 0.3:
        # Small consecutive values
        start = rng.randint(1, 20)
        vals = list(range(start, start + N))
        if max(vals) > MAX_VAL:
            vals = list(range(1, N + 1))
    elif strategy < 0.5:
        # Arithmetic progression
        a = rng.randint(1, 50)
        d = rng.randint(1, 100)
        vals = [a + i * d for i in range(N)]
        if max(vals) > MAX_VAL:
            vals = list(range(1, N + 1))
    elif strategy < 0.7:
        # Random small values
        vals = rng.sample(range(1, 500), N)
    elif strategy < 0.85:
        # Random medium values
        vals = rng.sample(range(1, 2000), N)
    else:
        # Random larger values
        vals = rng.sample(range(1, MAX_VAL + 1), N)
    
    # Ensure distinct
    vals = list(set(vals))
    while len(vals) < N:
        v = rng.randint(1, MAX_VAL)
        if v not in vals:
            vals.append(v)
    
    return vals[:N]

def construct():
    """Construct 14 distinct positive integers and 35 edges minimizing distinct sums/products."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = random.Random(42)
    
    best_overall_raw = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    # Try multiple value sets
    num_attempts = 0
    while time.time() < deadline:
        num_attempts += 1
        values = _generate_value_set(deadline, rng)
        
        # Ensure values are valid Python ints
        values = [int(v) for v in values]
        
        # Optimize edges for this value set
        edge_deadline = min(deadline - 2.0, time.time() + 15.0)  # leave time for next iterations
        edges, raw = _optimize_edges_for_values(values, edge_deadline)
        
        if raw < best_overall_raw:
            best_overall_raw = raw
            best_overall_values = values
            best_overall_edges = edges
        
        # If we've found a very good solution, we can be more conservative
        if best_overall_raw <= 10:
            break
        
        # Check if we have time for more
        if time.time() > deadline - 5.0:
            break
    
    # If no valid solution found (shouldn't happen), use default
    if best_overall_values is None:
        best_overall_values = list(range(1, N + 1))
        best_overall_edges = [list(e) for e in list(combinations(range(N), 2))[:M]]
    
    # Convert to proper types
    best_overall_values = [int(v) for v in best_overall_values]
    best_overall_edges = [[int(i), int(j)] for i, j in best_overall_edges]
    
    return {'values': best_overall_values, 'edges': best_overall_edges}
# EVOLVE-BLOCK-END
