"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_value(values, edges):
    """Compute the objective: max(#distinct sums, #distinct products)."""
    sums = set()
    prods = set()
    for i, j in edges:
        v = values[i]
        w = values[j]
        sums.add(v + w)
        prods.add(v * w)
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    n = P['n']
    m = P['m']
    maximum = P['maximum']
    
    # Generate all possible edges
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)
    
    # Initial random solution
    def random_values():
        vals = random.sample(range(1, maximum + 1), n)
        return vals
    
    def random_edges():
        edges = random.sample(all_edges, m)
        return [list(e) for e in edges]
    
    # Find a good initial solution
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Try several random initializations
    num_initials = 20
    for _ in range(num_initials):
        vals = random_values()
        edges = random_edges()
        score = _raw_value(vals, edges)
        if score < best_score:
            best_score = score
            best_values = vals[:]
            best_edges = [e[:] for e in edges]
    
    # Local search: try to improve by modifying values or edges
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_score = best_score
        current_values = best_values[:]
        current_edges = [e[:] for e in best_edges]
        
        # Try modifying a single value
        for trial in range(100):
            if time.time() > deadline - 5:
                break
            idx = random.randint(0, n - 1)
            old_val = current_values[idx]
            new_val = random.randint(1, maximum)
            if new_val in current_values:
                continue
            current_values[idx] = new_val
            new_score = _raw_value(current_values, current_edges)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current_values[idx] = old_val
        
        # Try swapping one edge for another
        if time.time() < deadline - 5:
            for trial in range(100):
                if time.time() > deadline - 5:
                    break
                edge_idx = random.randint(0, m - 1)
                old_edge = tuple(current_edges[edge_idx])
                new_edge = random.choice(all_edges)
                if new_edge in set(tuple(e) for e in current_edges):
                    continue
                current_edges[edge_idx] = list(new_edge)
                new_score = _raw_value(current_values, current_edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current_edges[edge_idx] = list(old_edge)
        
        # Try replacing two edges
        if time.time() < deadline - 5:
            for trial in range(50):
                if time.time() > deadline - 5:
                    break
                idx1 = random.randint(0, m - 1)
                idx2 = random.randint(0, m - 1)
                if idx1 == idx2:
                    continue
                old_edge1 = tuple(current_edges[idx1])
                old_edge2 = tuple(current_edges[idx2])
                new_edge1 = random.choice(all_edges)
                new_edge2 = random.choice(all_edges)
                edge_set = set(tuple(e) for e in current_edges)
                if new_edge1 in edge_set or new_edge2 in edge_set or new_edge1 == new_edge2:
                    continue
                current_edges[idx1] = list(new_edge1)
                current_edges[idx2] = list(new_edge2)
                new_score = _raw_value(current_values, current_edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current_edges[idx1] = list(old_edge1)
                    current_edges[idx2] = list(old_edge2)
        
        if current_score < best_score:
            best_score = current_score
            best_values = current_values[:]
            best_edges = [e[:] for e in current_edges]
    
    # If no good result found, use a default
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = [list(e) for e in all_edges[:m]]
    
    # Ensure types are correct
    result = {
        'values': [int(v) for v in best_values],
        'edges': [[int(i), int(j)] for i, j in best_edges]
    }
    
    return result
# EVOLVE-BLOCK-END
