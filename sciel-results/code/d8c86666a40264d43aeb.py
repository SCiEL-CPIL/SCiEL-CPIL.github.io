"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _raw_value(values, edges):
    """Compute max(distinct_sums, distinct_products) for given values and edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def _optimize_edges_for_values(values, m=35, deadline=None):
    """Find m edges that minimize max(distinct_sums, distinct_products)."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    total_edges = len(all_edges)
    
    best_raw = float('inf')
    best_edges = None
    
    # Multiple random restarts with greedy improvement
    for restart in range(200):
        if deadline and time.time() > deadline:
            break
        
        # Random initial set of m edges
        idx = list(range(total_edges))
        random.shuffle(idx)
        current_set = set(idx[:m])
        current_edges = [all_edges[i] for i in current_set]
        
        # Compute current raw value
        sums = set()
        prods = set()
        for i, j in current_edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        current_raw = max(len(sums), len(prods))
        
        if current_raw < best_raw:
            best_raw = current_raw
            best_edges = current_edges[:]
        
        # Greedy improvement: try swapping each edge with a non-edge
        improved = True
        while improved:
            if deadline and time.time() > deadline:
                break
            improved = False
            for pos in range(m):
                if deadline and time.time() > deadline:
                    break
                curr_edge_idx = idx[pos]
                curr_i, curr_j = all_edges[curr_edge_idx]
                
                # Try swapping with a random non-edge
                for _ in range(50):
                    if deadline and time.time() > deadline:
                        break
                    new_idx = random.randint(0, total_edges - 1)
                    if new_idx in current_set:
                        continue
                    new_i, new_j = all_edges[new_idx]
                    
                    # Remove current edge, add new edge
                    new_sums = set()
                    new_prods = set()
                    for k in range(m):
                        if k == pos:
                            new_sums.add(values[new_i] + values[new_j])
                            new_prods.add(values[new_i] * values[new_j])
                        else:
                            ei, ej = all_edges[idx[k]]
                            new_sums.add(values[ei] + values[ej])
                            new_prods.add(values[ei] * values[ej])
                    
                    new_raw = max(len(new_sums), len(new_prods))
                    
                    if new_raw < current_raw:
                        current_raw = new_raw
                        idx[pos] = new_idx
                        current_set.discard(curr_edge_idx)
                        current_set.add(new_idx)
                        current_edges = [all_edges[k] for k in idx]
                        improved = True
                        break
        
        if best_raw <= 1:
            break
    
    return best_raw, best_edges

def _generate_value_set(n=14, max_val=10000):
    """Generate a random set of n distinct positive integers <= max_val."""
    return random.sample(range(1, max_val + 1), n)

def construct():
    deadline = time.time() + 80  # 80 second deadline (10s safety margin)
    
    best_raw = float('inf')
    best_values = None
    best_edges = None
    
    # Strategy 1: Try structured value sets
    # Arithmetic progressions with small differences
    for start in range(1, 20):
        for diff in range(1, 5):
            vals = [start + i * diff for i in range(14)]
            if max(vals) <= 10000:
                raw, edges = _optimize_edges_for_values(vals, 35, deadline)
                if raw < best_raw:
                    best_raw = raw
                    best_values = vals
                    best_edges = edges
                if best_raw <= 2:
                    break
        if best_raw <= 2:
            break
        if deadline and time.time() > deadline:
            break
    
    # Strategy 2: Random value sets with optimization
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        vals = _generate_value_set(14, 10000)
        raw, edges = _optimize_edges_for_values(vals, 35, deadline)
        if raw < best_raw:
            best_raw = raw
            best_values = vals
            best_edges = edges
        if best_raw <= 2:
            break
        # Limit random attempts to avoid spending too much time
        if attempts >= 50:
            break
    
    # Fallback: use the seed-like solution if nothing better found
    if best_values is None:
        best_values = list(range(1, 15))
        best_edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
    
    # Ensure types are correct
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
