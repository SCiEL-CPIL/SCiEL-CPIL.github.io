import time
import random
import math
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def get_score(values, edges):
    """Compute the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def compute_edge_data(values):
    """Precompute sums and products for all possible edges."""
    n = len(values)
    edge_data = {}
    for i in range(n):
        for j in range(i+1, n):
            edge_data[(i, j)] = (values[i] + values[j], values[i] * values[j])
    return edge_data

def fast_score_with_counts(values, edges, edge_data):
    """Fast score computation using precomputed edge data."""
    sum_counts = {}
    prod_counts = {}
    for i, j in edges:
        s, p = edge_data[(i, j)]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    return max(len(sum_counts), len(prod_counts))

def simulated_annealing(values, edges, deadline, rng):
    """
    Jointly optimize values and edges using simulated annealing.
    Returns (best_values, best_edges, best_score).
    """
    n = len(values)
    m = P['m']
    
    current_values = list(values)
    current_edges = set(tuple(e) for e in edges)
    edge_data = compute_edge_data(current_values)
    
    current_score = fast_score_with_counts(current_values, current_edges, edge_data)
    best_values = list(current_values)
    best_edges = set(current_edges)
    best_score = current_score
    
    # Temperature schedule
    T_start = 50.0
    T_end = 0.1
    total_steps = 5000
    
    all_possible_edges = list(combinations(range(n), 2))
    
    step = 0
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() < deadline and no_improve < max_no_improve:
        step += 1
        T = T_start * (T_end / T_start) ** (step / total_steps)
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.4:
            # Perturb a value
            i = rng.randint(0, n-1)
            old_val = current_values[i]
            step_size = rng.choice([1, 2, 3, 5, 10, 20, 50])
            direction = rng.choice([-1, 1])
            new_val = old_val + direction * step_size
            
            if new_val <= 0 or new_val > 10000 or new_val in current_values:
                continue
            
            current_values[i] = new_val
            # Recompute edge data for affected edges
            for j in range(n):
                if j != i:
                    key = (min(i, j), max(i, j))
                    edge_data[key] = (current_values[i] + current_values[j], 
                                      current_values[i] * current_values[j])
            
            new_score = fast_score_with_counts(current_values, current_edges, edge_data)
            delta = new_score - current_score
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = list(current_values)
                    best_edges = set(current_edges)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_values[i] = old_val
                no_improve += 1
                
        elif move_type < 0.7:
            # Swap two values
            i, j = rng.sample(range(n), 2)
            current_values[i], current_values[j] = current_values[j], current_values[i]
            
            if len(set(current_values)) < n:
                current_values[i], current_values[j] = current_values[j], current_values[i]
                continue
            
            # Recompute edge data for affected edges
            for k in range(n):
                if k != i and k != j:
                    for idx in [i, j]:
                        key = (min(idx, k), max(idx, k))
                        edge_data[key] = (current_values[idx] + current_values[k], 
                                          current_values[idx] * current_values[k])
            key_ij = (min(i, j), max(i, j))
            edge_data[key_ij] = (current_values[i] + current_values[j], 
                                  current_values[i] * current_values[j])
            
            new_score = fast_score_with_counts(current_values, current_edges, edge_data)
            delta = new_score - current_score
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = list(current_values)
                    best_edges = set(current_edges)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_values[i], current_values[j] = current_values[j], current_values[i]
                no_improve += 1
                
        else:
            # Swap an edge: remove one, add another
            if len(current_edges) < m:
                # Add an edge
                candidates = [e for e in all_possible_edges if e not in current_edges]
                if not candidates:
                    continue
                e_in = rng.choice(candidates)
                current_edges.add(e_in)
                new_score = fast_score_with_counts(current_values, current_edges, edge_data)
                delta = new_score - current_score
                
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_values = list(current_values)
                        best_edges = set(current_edges)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    current_edges.remove(e_in)
                    no_improve += 1
            else:
                # Remove one, add another
                e_out = rng.choice(list(current_edges))
                candidates = [e for e in all_possible_edges if e not in current_edges]
                if not candidates:
                    continue
                e_in = rng.choice(candidates)
                
                current_edges.remove(e_out)
                current_edges.add(e_in)
                
                new_score = fast_score_with_counts(current_values, current_edges, edge_data)
                delta = new_score - current_score
                
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_values = list(current_values)
                        best_edges = set(current_edges)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    current_edges.remove(e_in)
                    current_edges.add(e_out)
                    no_improve += 1
    
    return best_values, list(best_edges), best_score

def greedy_select_edges(values):
    """Greedy edge selection as initial solution."""
    n = len(values)
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            s = values[i] + values[j]
            p = values[i] * values[j]
            all_edges.append((i, j, s, p))
    
    chosen_edges = []
    chosen_sums = set()
    chosen_prods = set()
    
    remaining = list(range(len(all_edges)))
    
    for _ in range(P['m']):
        best_idx = -1
        best_score = float('inf')
        best_delta_sum = float('inf')
        best_delta_prod = float('inf')
        
        for idx in remaining:
            i, j, s, p = all_edges[idx]
            
            delta_s = 0 if s in chosen_sums else 1
            delta_p = 0 if p in chosen_prods else 1
            
            new_k_s = len(chosen_sums) + delta_s
            new_k_p = len(chosen_prods) + delta_p
            
            new_max = max(new_k_s, new_k_p)
            
            if new_max < best_score or (new_max == best_score and (delta_s + delta_p) < best_delta_sum + best_delta_prod):
                best_score = new_max
                best_idx = idx
                best_delta_sum = delta_s
                best_delta_prod = delta_p
        
        if best_idx != -1:
            i, j, s, p = all_edges[best_idx]
            chosen_edges.append([i, j])
            chosen_sums.add(s)
            chosen_prods.add(p)
            remaining.remove(best_idx)
            
    return chosen_edges, max(len(chosen_sums), len(chosen_prods))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    rng = random.Random(42)
    
    # Generate diverse initial value sets
    initial_sets = [
        list(range(1, n+1)),
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],
        sorted(rng.sample(range(1, 100), n)),
        sorted(rng.sample(range(1, 500), n)),
        [2, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    ]
    
    # Ensure distinctness and positive integers
    valid_initial_sets = []
    for s in initial_sets:
        if len(s) == n and len(set(s)) == n and all(x > 0 for x in s) and all(x <= 10000 for x in s):
            valid_initial_sets.append(s)
            
    if not valid_initial_sets:
        valid_initial_sets = [list(range(1, n+1))]

    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # For each initial set, run simulated annealing
    for vals in valid_initial_sets:
        if time.time() > deadline - 5.0:
            break
        
        # Get initial edges via greedy
        edges, _ = greedy_select_edges(vals)
        
        # Run simulated annealing with remaining time
        sa_deadline = min(deadline - 2.0, start_time + 80.0)
        v, e, score = simulated_annealing(vals, edges, sa_deadline, rng)
        
        if score < best_score:
            best_score = score
            best_values = v
            best_edges = e
            
    # Final fallback
    if best_values is None:
        best_values = list(range(1, n+1))
        best_edges, best_score = greedy_select_edges(best_values)
        
    return {'values': [int(v) for v in best_values], 'edges': [[int(a), int(b)] for a, b in best_edges]}
