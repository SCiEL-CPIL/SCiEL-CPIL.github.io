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

def _collision_frequency_score(values, m=35):
    """Compute collision frequency for all possible edges.
    Returns: (edge_sums, edge_prods, sum_freq, prod_freq, all_edges)
    where sum_freq[s] = number of edges with sum s, prod_freq[p] = number of edges with product p
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    edge_sums = []
    edge_prods = []
    for i, j in all_edges:
        edge_sums.append(values[i] + values[j])
        edge_prods.append(values[i] * values[j])
    
    sum_freq = {}
    prod_freq = {}
    for s in edge_sums:
        sum_freq[s] = sum_freq.get(s, 0) + 1
    for p in edge_prods:
        prod_freq[p] = prod_freq.get(p, 0) + 1
    
    return all_edges, edge_sums, edge_prods, sum_freq, prod_freq

def _greedy_collision_select(all_edges, edge_sums, edge_prods, sum_freq, prod_freq, m=35, deadline=None):
    """Greedy selection: pick edges with highest collision frequency first.
    An edge's 'value' is min(sum_freq[s], prod_freq[p]) - edges that are in dense clusters.
    """
    n = len(all_edges)
    # Score each edge by its collision potential
    edge_scores = []
    for idx in range(n):
        s = edge_sums[idx]
        p = edge_prods[idx]
        # Higher frequency = more collisions = better
        score = sum_freq[s] + prod_freq[p]
        edge_scores.append((score, idx))
    
    # Sort by score descending
    edge_scores.sort(reverse=True)
    
    selected = set()
    cur_sums = set()
    cur_prods = set()
    
    for score, idx in edge_scores:
        if len(selected) >= m:
            break
        if deadline and time.time() > deadline:
            break
        s = edge_sums[idx]
        p = edge_prods[idx]
        selected.add(idx)
        cur_sums.add(s)
        cur_prods.add(p)
    
    return list(selected), cur_sums, cur_prods

def _swap_local_search(selected, all_edges, edge_sums, edge_prods, sum_freq, prod_freq, m=35, deadline=None, max_iter=30):
    """Local search: try swapping selected edges with unselected ones to improve score."""
    selected = set(selected)
    outside = set(range(len(all_edges))) - selected
    
    cur_sums = set()
    cur_prods = set()
    for idx in selected:
        cur_sums.add(edge_sums[idx])
        cur_prods.add(edge_prods[idx])
    cur_score = max(len(cur_sums), len(cur_prods))
    
    improved = True
    it = 0
    while improved and it < max_iter:
        if deadline and time.time() > deadline:
            break
        improved = False
        it += 1
        
        # Try all possible swaps
        for e_in in list(selected):
            if deadline and time.time() > deadline:
                break
            s_in = edge_sums[e_in]
            p_in = edge_prods[e_in]
            
            # Count how many edges have this sum/prod
            sum_count = sum(1 for idx in selected if edge_sums[idx] == s_in)
            prod_count = sum(1 for idx in selected if edge_prods[idx] == p_in)
            
            for e_out in outside:
                if deadline and time.time() > deadline:
                    break
                s_out = edge_sums[e_out]
                p_out = edge_prods[e_out]
                
                # Simulate swap
                new_sums = set(cur_sums)
                new_prods = set(cur_prods)
                
                # Remove e_in
                if sum_count == 1:
                    new_sums.discard(s_in)
                if prod_count == 1:
                    new_prods.discard(p_in)
                
                # Add e_out
                new_sums.add(s_out)
                new_prods.add(p_out)
                
                trial_score = max(len(new_sums), len(new_prods))
                
                if trial_score < cur_score:
                    selected.discard(e_in)
                    selected.add(e_out)
                    outside.discard(e_out)
                    outside.add(e_in)
                    
                    cur_sums = new_sums
                    cur_prods = new_prods
                    cur_score = trial_score
                    improved = True
                    break
            
            if improved:
                break
    
    return list(selected), cur_score

def _optimize_edges(values, m=35, deadline=None, num_restarts=10):
    """Full edge optimization: greedy init + local search, multiple restarts."""
    all_edges, edge_sums, edge_prods, sum_freq, prod_freq = _collision_frequency_score(values, m)
    
    best_score = float('inf')
    best_selected = None
    
    for restart in range(num_restarts):
        if deadline and time.time() > deadline:
            break
        
        # Greedy init
        selected, cur_sums, cur_prods = _greedy_collision_select(
            all_edges, edge_sums, edge_prods, sum_freq, prod_freq, m, deadline
        )
        
        if len(selected) < m:
            continue
        
        # Local search
        selected, score = _swap_local_search(
            selected, all_edges, edge_sums, edge_prods, sum_freq, prod_freq, m, deadline, max_iter=25
        )
        
        if score < best_score:
            best_score = score
            best_selected = list(selected)
    
    if best_selected is None:
        # Fallback: first m edges
        best_selected = list(range(m))
        best_score = _raw_value(values, [all_edges[i] for i in best_selected])
    
    edges = [list(all_edges[idx]) for idx in best_selected]
    return edges, best_score

def _value_perturb_for_collisions(values, edges, deadline=None, num_trials=20):
    """Perturb values to create more sum/product collisions."""
    n = len(values)
    best_vals = list(values)
    best_raw = _raw_value(values, edges)
    
    for trial in range(num_trials):
        if deadline and time.time() > deadline:
            break
        
        trial_vals = list(values)
        # Perturb 1-3 random values
        num_perturb = random.randint(1, 3)
        idxs = random.sample(range(n), num_perturb)
        
        for idx in idxs:
            delta = random.choice([-3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10, 12])
            new_v = trial_vals[idx] + delta
            if 1 <= new_v <= 10000 and new_v not in trial_vals:
                trial_vals[idx] = new_v
        
        if len(set(trial_vals)) != n:
            continue
        
        # Optimize edges for new values
        new_edges, new_raw = _optimize_edges(trial_vals, 35, deadline, num_restarts=5)
        
        if new_raw < best_raw:
            best_raw = new_raw
            best_vals = trial_vals
    
    return best_vals, best_raw

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    best_result = None
    best_raw = float('inf')
    
    # Structured value sets designed for collision promotion
    # Key: numbers with many additive/multiplicative relations
    value_sets = [
        # Small numbers with many additive relations
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        # Numbers with many divisors (promote product collisions)
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        # Consecutive
        list(range(1, 15)),
        # Mixed small
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20],
        # Powers and composites
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        # More structured
        [1, 2, 3, 4, 6, 8, 12, 16, 18, 24, 30, 36, 48, 60],
        # Geometric-ish
        [1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 72, 96, 144],
    ]
    
    for vals in value_sets:
        if deadline and time.time() > deadline:
            break
        if len(set(vals)) != 14 or any(v <= 0 or v > 10000 for v in vals):
            continue
        
        edges, raw = _optimize_edges(vals, 35, deadline, num_restarts=10)
        if edges and raw < best_raw:
            best_raw = raw
            best_result = {'values': list(vals), 'edges': [list(e) for e in edges]}
    
    # Joint optimization: perturb values to find better collision structures
    if best_result and deadline and time.time() < deadline:
        for _ in range(3):
            if deadline and time.time() > deadline:
                break
            new_vals, new_raw = _value_perturb_for_collisions(
                best_result['values'], best_result['edges'], deadline, num_trials=15
            )
            if new_raw < best_raw:
                best_raw = new_raw
                new_edges, _ = _optimize_edges(new_vals, 35, deadline, num_restarts=10)
                best_result = {'values': list(new_vals), 'edges': [list(e) for e in new_edges]}
            else:
                break
    
    # Final polish: try more perturbations
    if best_result and deadline and time.time() < deadline - 5:
        for _ in range(2):
            if deadline and time.time() > deadline:
                break
            new_vals, new_raw = _value_perturb_for_collisions(
                best_result['values'], best_result['edges'], deadline, num_trials=25
            )
            if new_raw < best_raw:
                best_raw = new_raw
                new_edges, _ = _optimize_edges(new_vals, 35, deadline, num_restarts=15)
                best_result = {'values': list(new_vals), 'edges': [list(e) for e in new_edges]}
    
    if best_result is None:
        vals = list(range(1, 15))
        edges, _ = _optimize_edges(vals, 35, deadline, num_restarts=5)
        best_result = {'values': vals, 'edges': [list(e) for e in edges]}
    
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
