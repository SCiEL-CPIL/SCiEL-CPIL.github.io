import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    maximum = 10000
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)  # 91
    
    # Precompute sum and product for each edge given values
    def compute_edge_props(values, edges):
        """Return (sums_set, prods_set, sum_count, prod_count) for given edges."""
        sums = set()
        prods = set()
        for i, j in edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            sums.add(s)
            prods.add(p)
        return len(sums), len(prods), max(len(sums), len(prods))
    
    # Start with values 1..14
    best_values = list(range(1, n + 1))
    
    # Find best edge set for given values using local search
    def optimize_edges(values, time_limit):
        # Greedy initialization: start with all edges, remove worst ones
        # Or start with a random subset and improve
        current_edges = list(all_edges[:m])  # First 35 edges
        current_score = compute_edge_props(values, current_edges)[2]
        
        # Precompute sum and product for all edges
        edge_sum = {}
        edge_prod = {}
        for i, j in all_edges:
            edge_sum[(i, j)] = values[i] + values[j]
            edge_prod[(i, j)] = values[i] * values[j]
        
        # Local search: try swapping edges
        best_edges = list(current_edges)
        best_score = current_score
        no_improve_count = 0
        
        while time.time() < time_limit:
            improved = False
            # Try all possible swaps: remove one edge, add one not in current set
            current_edge_set = set(current_edges)
            
            for rem_idx, rem_edge in enumerate(current_edges):
                if time.time() >= time_limit:
                    break
                for add_edge in all_edges:
                    if add_edge in current_edge_set:
                        continue
                    # Try swap
                    new_edges = list(current_edges)
                    new_edges[rem_idx] = add_edge
                    new_score = compute_edge_props(values, new_edges)[2]
                    if new_score < best_score:
                        best_score = new_score
                        best_edges = new_edges
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # Try single additions/removals if we're stuck? No, we need exactly m edges.
                # Try random restarts within this values set
                break
            
            current_edges = best_edges
        
        return best_edges, best_score
    
    # Joint optimization: try different value sets
    best_result = None
    best_raw = float('inf')
    
    # Value set candidates
    value_candidates = [
        list(range(1, n + 1)),  # 1..14
    ]
    
    # Also try some perturbed value sets
    random.seed(42)
    for _ in range(20):
        base = list(range(1, n + 1))
        # Perturb: swap some values, or use a different small set
        perm = base[:]
        random.shuffle(perm)
        # Ensure distinct and positive
        if len(set(perm)) == n and all(1 <= x <= maximum for x in perm):
            value_candidates.append(perm)
    
    # Also try arithmetic progressions and other structured sets
    for start in range(1, 5):
        for step in range(1, 3):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= maximum and len(set(vals)) == n:
                value_candidates.append(vals)
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for vc in value_candidates:
        key = tuple(sorted(vc))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(vc)
    
    for values in unique_candidates:
        if time.time() >= deadline:
            break
        
        # Optimize edges for this value set
        edges, score = optimize_edges(values, deadline)
        
        if score < best_raw:
            best_raw = score
            best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
    
    # If we haven't found anything, use a default
    if best_result is None:
        best_result = {'values': list(range(1, n + 1)), 'edges': [list(e) for e in all_edges[:m]]}
    
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
