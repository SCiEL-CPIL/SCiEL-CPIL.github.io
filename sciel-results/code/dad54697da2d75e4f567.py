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

def _fast_edge_optimize(values, m=35, deadline=None, num_restarts=15):
    """Fast edge optimization with incremental score updates."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_all = len(all_edges)
    
    edge_sums = [0] * num_all
    edge_prods = [0] * num_all
    for idx, (i, j) in enumerate(all_edges):
        edge_sums[idx] = values[i] + values[j]
        edge_prods[idx] = values[i] * values[j]
    
    best_score = float('inf')
    best_edges = None
    
    for restart in range(num_restarts):
        if deadline and time.time() > deadline:
            break
        
        # Greedy init: pick edges that maximize collisions
        current = set()
        cur_sums = set()
        cur_prods = set()
        candidates = list(range(num_all))
        random.shuffle(candidates)
        
        for _ in range(m):
            best_idx = -1
            best_key = None
            for idx in candidates:
                if idx in current:
                    continue
                ns = cur_sums | {edge_sums[idx]}
                np_ = cur_prods | {edge_prods[idx]}
                key = (max(len(ns), len(np_)), len(ns) + len(np_))
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
            if best_idx == -1:
                break
            current.add(best_idx)
            cur_sums.add(edge_sums[best_idx])
            cur_prods.add(edge_prods[best_idx])
            candidates.remove(best_idx)
        
        if len(current) < m:
            continue
        
        # Compute actual score
        actual_sums = set()
        actual_prods = set()
        for idx in current:
            actual_sums.add(edge_sums[idx])
            actual_prods.add(edge_prods[idx])
        cur_score = max(len(actual_sums), len(actual_prods))
        
        # Local search with swaps
        improved = True
        max_iter = 50
        it = 0
        while improved and it < max_iter and (not deadline or time.time() < deadline):
            improved = False
            it += 1
            cur_list = list(current)
            outside = [x for x in range(num_all) if x not in current]
            random.shuffle(cur_list)
            random.shuffle(outside)
            
            for e_in in cur_list:
                if deadline and time.time() > deadline:
                    break
                s_in = edge_sums[e_in]
                p_in = edge_prods[e_in]
                
                # Build sum/prod counts for proper removal
                sum_counts = {}
                prod_counts = {}
                for idx in current:
                    s = edge_sums[idx]
                    p = edge_prods[idx]
                    sum_counts[s] = sum_counts.get(s, 0) + 1
                    prod_counts[p] = prod_counts.get(p, 0) + 1
                
                for e_out in outside:
                    s_out = edge_sums[e_out]
                    p_out = edge_prods[e_out]
                    
                    # New sum count after swap
                    new_sum_counts = dict(sum_counts)
                    new_prod_counts = dict(prod_counts)
                    
                    # Remove e_in
                    new_sum_counts[s_in] -= 1
                    if new_sum_counts[s_in] == 0:
                        del new_sum_counts[s_in]
                    new_prod_counts[p_in] -= 1
                    if new_prod_counts[p_in] == 0:
                        del new_prod_counts[p_in]
                    
                    # Add e_out
                    new_sum_counts[s_out] = new_sum_counts.get(s_out, 0) + 1
                    new_prod_counts[p_out] = new_prod_counts.get(p_out, 0) + 1
                    
                    trial_score = max(len(new_sum_counts), len(new_prod_counts))
                    
                    if trial_score < cur_score:
                        current.discard(e_in)
                        current.add(e_out)
                        cur_score = trial_score
                        improved = True
                        if cur_score < best_score:
                            best_score = cur_score
                            best_edges = [list(all_edges[idx]) for idx in current]
                        break
                if improved:
                    break
        
        if cur_score < best_score:
            best_score = cur_score
            best_edges = [list(all_edges[idx]) for idx in current]
    
    return best_edges, best_score

def _optimize_edges(values, m=35, deadline=None):
    """Optimize edge selection for given values."""
    return _fast_edge_optimize(values, m, deadline)

def _joint_optimize(start_values, deadline, max_rounds=8):
    """Jointly optimize values and edges through iterative perturbation."""
    best_raw = float('inf')
    best_vals = None
    best_edges = None
    
    vals = list(start_values)
    
    for rnd in range(max_rounds):
        if deadline and time.time() > deadline:
            break
        
        # Optimize edges for current values
        edges, raw = _fast_edge_optimize(vals, 35, deadline, num_restarts=10)
        
        if edges and raw < best_raw:
            best_raw = raw
            best_vals = list(vals)
            best_edges = [list(e) for e in edges]
        
        # Perturb values: try swapping some values with nearby numbers
        # to create more sum/product collisions
        new_vals = list(vals)
        for _ in range(3):  # try a few perturbations
            if deadline and time.time() > deadline:
                break
            trial_vals = list(new_vals)
            # Pick 2-3 random indices and perturb
            idxs = random.sample(range(14), random.randint(2, 3))
            for idx in idxs:
                delta = random.choice([-3, -2, -1, 1, 2, 3, 4, 5])
                new_v = trial_vals[idx] + delta
                if 1 <= new_v <= 10000 and new_v not in trial_vals:
                    trial_vals[idx] = new_v
            
            if len(set(trial_vals)) != 14:
                continue
            
            t_edges, t_raw = _fast_edge_optimize(trial_vals, 35, deadline, num_restarts=5)
            if t_edges and t_raw < best_raw:
                best_raw = t_raw
                best_vals = list(trial_vals)
                best_edges = [list(e) for e in t_edges]
                new_vals = list(trial_vals)
        
        vals = new_vals
    
    if best_vals is None:
        return None, float('inf')
    return {'values': best_vals, 'edges': best_edges}, best_raw

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (90s limit)
    
    best_result = None
    best_raw = float('inf')
    
    # Structured value sets designed for collision promotion
    value_sets = [
        # Small numbers with many additive/multiplicative relations
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        # Numbers with many divisors (promote product collisions)
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        # Consecutive
        list(range(1, 15)),
        # Mixed small
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20],
        # Powers and composites
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
    ]
    
    for vals in value_sets:
        if deadline and time.time() > deadline:
            break
        if len(set(vals)) != 14 or any(v <= 0 or v > 10000 for v in vals):
            continue
        
        # Quick edge optimization first
        edges, raw = _fast_edge_optimize(vals, 35, deadline, num_restarts=8)
        if edges and raw < best_raw:
            best_raw = raw
            best_result = {'values': list(vals), 'edges': [list(e) for e in edges]}
    
    # Joint optimization from best found so far
    if best_result and deadline and time.time() < deadline:
        jr, jr_raw = _joint_optimize(best_result['values'], deadline, max_rounds=5)
        if jr and jr_raw < best_raw:
            best_raw = jr_raw
            best_result = jr
    
    # Additional joint optimization from other starting points if time permits
    if deadline and time.time() < deadline - 5:
        for vals in value_sets[:2]:
            if deadline and time.time() > deadline:
                break
            jr, jr_raw = _joint_optimize(vals, deadline, max_rounds=3)
            if jr and jr_raw < best_raw:
                best_raw = jr_raw
                best_result = jr
    
    if best_result is None:
        vals = list(range(1, 15))
        edges = list(combinations(range(14), 2))[:35]
        best_result = {'values': vals, 'edges': edges}
    
    # Format output
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
