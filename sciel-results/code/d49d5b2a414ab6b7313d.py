"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

def _evaluate_fixed_edges(values, edges):
    """
    Evaluate the score for a fixed set of edges and given values.
    Returns max(distinct_sums, distinct_products).
    """
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _generate_regular_graph(n, k, seed):
    """
    Generate a random k-regular graph on n nodes.
    Returns a list of edges [i, j].
    """
    rng = random.Random(seed)
    # Simple method: shuffle neighbors
    # For n=14, k=5.
    # We can use the configuration model or just random sampling with rejection.
    # Given small n, rejection sampling is fine.
    while True:
        degrees = [0] * n
        edges = []
        # Create a list of potential edges
        all_edges = list(itertools.combinations(range(n), 2))
        rng.shuffle(all_edges)
        
        # Greedy assignment of edges to satisfy degree constraints
        # This might fail, so we retry.
        valid = True
        current_edges = []
        for e in all_edges:
            u, v = e
            if degrees[u] < k and degrees[v] < k:
                current_edges.append(e)
                degrees[u] += 1
                degrees[v] += 1
            if all(d == k for d in degrees):
                break
        
        if all(d == k for d in degrees):
            return current_edges
        # If not, retry with new shuffle

def _optimize_values_for_edges(edges, n, time_limit, rng):
    """
    Local search to find values that minimize score for fixed edges.
    """
    # Initialize with random small integers
    values = [rng.randint(1, 20) for _ in range(n)]
    # Ensure distinctness
    while len(set(values)) < n:
        dupes = [v for v in set(values) if values.count(v) > 1]
        for d in dupes:
            idx = values.index(d)
            values[idx] = rng.randint(1, 50)
    
    current_score = _evaluate_fixed_edges(values, edges)
    best_values = values[:]
    best_score = current_score
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Pick a random index to perturb
        idx = rng.randint(0, n - 1)
        old_val = values[idx]
        
        # Perturb: change to a random small integer
        # Bias towards small numbers as they create more collisions
        new_val = rng.randint(1, 30)
        
        # Ensure distinctness
        if new_val in values and values[idx] != new_val:
             # Try a few times to find a unique value
             for _ in range(10):
                 new_val = rng.randint(1, 50)
                 if new_val not in values or values[idx] == new_val:
                     break
             if new_val in values and values[idx] != new_val:
                 continue

        values[idx] = new_val
        new_score = _evaluate_fixed_edges(values, edges)
        
        if new_score < best_score:
            best_score = new_score
            best_values = values[:]
        else:
            # Revert if not better (Steepest Descent / First Improvement hybrid)
            # To allow escaping local minima, we could use SA, but simple descent is faster per iteration.
            # Let's add a small probability of accepting worse moves (SA-like)
            if rng.random() < 0.05:
                # Accept worse move to escape local minima
                pass
            else:
                values[idx] = old_val

    return best_values, best_score

def construct():
    n = 14
    m = 35
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin
    
    rng = random.Random(42)
    
    best_global_score = float('inf')
    best_global_values = []
    best_global_edges = []
    
    # Generate a few different 5-regular graph topologies
    # Each topology is a set of 35 edges.
    num_topologies = 5
    
    # Time per topology
    time_per_topology = (time_limit - 5.0) / num_topologies
    
    for t in range(num_topologies):
        # Generate a random 5-regular graph
        # Note: _generate_regular_graph might take a few retries, but n is small.
        edges = _generate_regular_graph(n, 5, seed=42 + t)
        
        if not edges:
            continue
            
        # Optimize values for this specific topology
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 1.0:
            break
            
        values, score = _optimize_values_for_edges(edges, n, remaining_time - 1.0, rng)
        
        if score < best_global_score:
            best_global_score = score
            best_global_values = values
            best_global_edges = edges
    
    # Fallback if no topology worked or time ran out
    if not best_global_values:
        best_global_values = list(range(1, n + 1))
        best_global_edges = _generate_regular_graph(n, 5, seed=999)
        # Re-evaluate to ensure consistency
        # (Though the score might not be optimal, it's a valid structure)
        
    # Ensure types are correct
    final_values = [int(v) for v in best_global_values]
    final_edges = [[int(i), int(j)] for i, j in best_global_edges]
    
    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
