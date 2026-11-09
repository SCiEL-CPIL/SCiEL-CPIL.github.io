import time
import random
import math
from itertools import combinations

# Deadline guard: 90s limit, use 85s for safety
START_TIME = time.time()
DEADLINE = START_TIME + 85.0

def _check_time():
    return time.time() > DEADLINE

def _score(values, edges):
    """Calculate the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _optimize_edges(values, n_edges=35, max_iters=20000):
    """
    Find the best set of n_edges edges for a given set of values
    using Simulated Annealing with incremental score updates.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_all = len(all_edges)
    
    # Precompute sums and products for all possible edges
    edge_props = {}
    for e in all_edges:
        u, v = e
        edge_props[e] = (values[u] + values[v], values[u] * values[v])
        
    # Initialize with a greedy approach
    # Sort edges by how 'common' their sum and product are
    sum_counts_all = {}
    prod_counts_all = {}
    for s, p in edge_props.values():
        sum_counts_all[s] = sum_counts_all.get(s, 0) + 1
        prod_counts_all[p] = prod_counts_all.get(p, 0) + 1
        
    def edge_priority(e):
        s, p = edge_props[e]
        return (sum_counts_all[s], prod_counts_all[p])
        
    sorted_edges = sorted(all_edges, key=edge_priority, reverse=True)
    
    current_edges = set(sorted_edges[:n_edges])
    
    # Initialize counts for current edges
    s_counts = {}
    p_counts = {}
    for e in current_edges:
        s, p = edge_props[e]
        s_counts[s] = s_counts.get(s, 0) + 1
        p_counts[p] = p_counts.get(p, 0) + 1
        
    current_score = max(len(s_counts), len(p_counts))
    
    best_edges = set(current_edges)
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    iter_count = 0
    while T > min_T and iter_count < max_iters:
        if _check_time():
            break
            
        # Pick a random edge to remove
        u, v = random.choice(list(current_edges))
        s_old, p_old = edge_props[(u, v)]
        
        # Temporarily remove
        s_counts[s_old] -= 1
        if s_counts[s_old] == 0: del s_counts[s_old]
        p_counts[p_old] -= 1
        if p_counts[p_old] == 0: del p_counts[p_old]
        
        base_s_count = len(s_counts)
        base_p_count = len(p_counts)
        
        # Pick a random edge to add (not currently in set)
        # Efficiently pick from non-edges
        cand_e = None
        attempts = 0
        while cand_e is None or cand_e in current_edges:
            cand_e = random.choice(all_edges)
            attempts += 1
            if attempts > 100:
                break
                
        if cand_e is None or cand_e in current_edges:
            # Restore old edge
            s_counts[s_old] = s_counts.get(s_old, 0) + 1
            p_counts[p_old] = p_counts.get(p_old, 0) + 1
            continue
            
        s_new, p_new = edge_props[cand_e]
        
        # Calculate potential new score
        new_s_count = base_s_count + (0 if s_new in s_counts else 1)
        new_p_count = base_p_count + (0 if p_new in p_counts else 1)
        new_score = max(new_s_count, new_p_count)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept move
            current_edges.discard((u, v))
            current_edges.add(cand_e)
            
            # Update counts for new edge
            s_counts[s_new] = s_counts.get(s_new, 0) + 1
            p_counts[p_new] = p_counts.get(p_new, 0) + 1
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
        else:
            # Reject move, restore old edge
            s_counts[s_old] = s_counts.get(s_old, 0) + 1
            p_counts[p_old] = p_counts.get(p_old, 0) + 1
            
        T *= cooling
        iter_count += 1
        
    return best_score, list(best_edges)

def _generate_candidates():
    """
    Generate value sets with strong multiplicative structure.
    """
    candidates = []
    n = 14
    
    # Grid 2^a 3^b
    grid_23 = []
    for a in range(0, 5):
        for b in range(0, 5):
            val = (2 ** a) * (3 ** b)
            if val <= 10000:
                grid_23.append(val)
    grid_23 = sorted(set(grid_23))
    
    # Smallest 14
    if len(grid_23) >= 14:
        candidates.append(grid_23[:14])
        
    # Random subsets of grid_23
    random.seed(42)
    for _ in range(20):
        if len(grid_23) >= 14:
            vals = random.sample(grid_23, 14)
            vals.sort()
            candidates.append(vals)
            
    # Grid 2^a 3^b 5^c
    grid_235 = []
    for a in range(0, 4):
        for b in range(0, 4):
            for c in range(0, 3):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 10000:
                    grid_235.append(val)
    grid_235 = sorted(set(grid_235))
    
    if len(grid_235) >= 14:
        candidates.append(grid_235[:14])
        
    for _ in range(20):
        if len(grid_235) >= 14:
            vals = random.sample(grid_235, 14)
            vals.sort()
            candidates.append(vals)
            
    # Hardcoded sets from previous good runs
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36])
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 48])
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24])
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18])
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16])
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        if len(c) != 14: continue
        if len(set(c)) != 14: continue
        if min(c) < 1 or max(c) > 10000: continue
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(list(c))
            
    return unique_candidates

def construct():
    n = 14
    m = 35
    
    candidates = _generate_candidates()
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Iterate through candidates
    for vals in candidates:
        if _check_time():
            break
            
        score, edges = _optimize_edges(vals, n_edges=m, max_iters=10000)
        
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = edges
            
            if best_score <= 5:
                break
                
    # If we have time, try a few more random value perturbations on the best found
    if best_values is not None and not _check_time():
        current_vals = best_values[:]
        current_edges = best_edges
        current_score = best_score
        
        # Simple local search on values: swap one value with a small random one
        for _ in range(100):
            if _check_time():
                break
            idx = random.randint(0, n - 1)
            old_val = current_vals[idx]
            new_val = random.randint(1, 50)
            if new_val == old_val or new_val in current_vals:
                continue
                
            new_vals = current_vals[:]
            new_vals[idx] = new_val
            
            # Re-optimize edges for this new value set? That's expensive.
            # Instead, just check score with current edges.
            new_score = _score(new_vals, current_edges)
            
            if new_score < current_score:
                current_vals = new_vals
                current_score = new_score
                # Try to improve edges for this new value set
                opt_score, opt_edges = _optimize_edges(current_vals, n_edges=m, max_iters=2000)
                if opt_score < current_score:
                    current_score = opt_score
                    current_edges = opt_edges
                    best_score = current_score
                    best_values = current_vals
                    best_edges = current_edges

    # Final validation
    if best_values is None:
        best_values = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
        best_edges = list(combinations(range(n), 2))[:m]
        
    values = [int(v) for v in best_values]
    edges = [[int(u), int(v)] for u, v in best_edges]
    
    # Sanity checks
    assert len(values) == n
    assert len(set(values)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in values)
    assert min(values) >= 1 and max(values) <= 10000
    assert len(edges) == m
    assert all(u != v for u, v in edges)
    assert len(set(tuple(sorted(e)) for e in edges)) == m
    
    return {'values': values, 'edges': edges}
