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

def _optimize_edges(values, m=35, deadline=None):
    """Optimize edge selection for given values using fast local search."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)
    
    # Precompute sums and products for all edges
    edge_sums = [0] * num_all_edges
    edge_prods = [0] * num_all_edges
    edge_index = {}
    for idx, (i, j) in enumerate(all_edges):
        edge_sums[idx] = values[i] + values[j]
        edge_prods[idx] = values[i] * values[j]
        edge_index[(i, j)] = idx
    
    # Map edge tuple to index for fast lookup
    edge_to_idx = {}
    for idx, e in enumerate(all_edges):
        edge_to_idx[e] = idx
    
    def get_score_from_indices(edge_indices):
        """Compute score from edge index list."""
        sums = set()
        prods = set()
        for idx in edge_indices:
            sums.add(edge_sums[idx])
            prods.add(edge_prods[idx])
        return max(len(sums), len(prods))
    
    best_score = float('inf')
    best_edges = None
    
    # Use multiple restarts with faster greedy init and local search
    num_restarts = 30
    
    for restart in range(num_restarts):
        if deadline and time.time() > deadline:
            break
        
        # Greedy Init with randomization
        current_edge_indices = set()
        current_sums = set()
        current_prods = set()
        
        # Random order for greedy
        candidate_indices = list(range(num_all_edges))
        random.shuffle(candidate_indices)
        
        for _ in range(m):
            best_idx = -1
            best_score_candidate = float('inf')
            best_sums_candidate = None
            best_prods_candidate = None
            
            for idx in candidate_indices:
                if idx in current_edge_indices:
                    continue
                s = edge_sums[idx]
                p = edge_prods[idx]
                new_sums = current_sums | {s}
                new_prods = current_prods | {p}
                score = max(len(new_sums), len(new_prods))
                tie_breaker = len(new_sums) + len(new_prods)
                
                if best_idx == -1 or (score, tie_breaker) < (best_score_candidate, 
                    len(best_sums_candidate) + len(best_prods_candidate) if best_sums_candidate else float('inf')):
                    best_score_candidate = score
                    best_idx = idx
                    best_sums_candidate = new_sums
                    best_prods_candidate = new_prods
                    
            if best_idx == -1:
                break
                
            current_edge_indices.add(best_idx)
            current_sums = best_sums_candidate
            current_prods = best_prods_candidate
            candidate_indices.remove(best_idx)
            
        if len(current_edge_indices) < m:
            continue
            
        current_score = get_score_from_indices(list(current_edge_indices))
        
        # Local Search: Fast swaps using incremental updates
        improved = True
        max_iterations = 200
        iteration = 0
        while improved and iteration < max_iterations and (not deadline or time.time() < deadline):
            improved = False
            iteration += 1
            
            # Try swapping each edge in current set with each edge outside
            current_list = list(current_edge_indices)
            outside_indices = [idx for idx in range(num_all_edges) if idx not in current_edge_indices]
            random.shuffle(current_list)
            random.shuffle(outside_indices)
            
            for i, e_in_idx in enumerate(current_list):
                if deadline and time.time() > deadline:
                    break
                s_in = edge_sums[e_in_idx]
                p_in = edge_prods[e_in_idx]
                
                for e_out_idx in outside_indices:
                    s_out = edge_sums[e_out_idx]
                    p_out = edge_prods[e_out_idx]
                    
                    # Compute new sums and prods
                    new_sums = set(current_sums)
                    new_prods = set(current_prods)
                    
                    # Remove old edge's contribution
                    if s_in in new_sums:
                        new_sums.discard(s_in)
                    if p_in in new_prods:
                        new_prods.discard(p_in)
                    
                    # Add new edge's contribution
                    new_sums.add(s_out)
                    new_prods.add(p_out)
                    
                    # But we need to be careful: other edges might have same sum/prod
                    # Recompute properly
                    trial_sums = set()
                    trial_prods = set()
                    for idx in current_edge_indices:
                        if idx == e_in_idx:
                            continue
                        trial_sums.add(edge_sums[idx])
                        trial_prods.add(edge_prods[idx])
                    trial_sums.add(s_out)
                    trial_prods.add(p_out)
                    
                    trial_score = max(len(trial_sums), len(trial_prods))
                    
                    if trial_score < current_score:
                        # Accept the swap
                        current_edge_indices.remove(e_in_idx)
                        current_edge_indices.add(e_out_idx)
                        current_score = trial_score
                        current_sums = trial_sums
                        current_prods = trial_prods
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_edges = [all_edges[idx] for idx in current_edge_indices]
                        break
                if improved:
                    break
        
        if current_score < best_score:
            best_score = current_score
            best_edges = [all_edges[idx] for idx in current_edge_indices]
            
    return best_edges, best_score

def _optimize_values_and_edges(deadline):
    """Try multiple value sets and optimize edges for each."""
    best_raw = float('inf')
    best_result = None
    
    # Generate candidate value sets
    value_sets = []
    
    # 1. Consecutive small integers
    value_sets.append(list(range(1, 15)))
    
    # 2. Powers of 2 (for product collisions)
    value_sets.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    
    # 3. Geometric progression with smaller ratio
    value_sets.append([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
    
    # 4. Small primes
    value_sets.append([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43])
    
    # 5. Mixed: some small, some larger for variety
    value_sets.append([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40])
    
    # 6. Arithmetic progression
    value_sets.append(list(range(1, 15, 1)))
    
    # 7. Squares
    value_sets.append([i*i for i in range(1, 15)])
    
    # 8. Custom: try to maximize collisions
    value_sets.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 36, 48])
    
    for vals in value_sets:
        if deadline and time.time() > deadline:
            break
        
        # Ensure values are distinct and positive <= 10000
        if len(set(vals)) != 14 or any(v <= 0 or v > 10000 for v in vals):
            continue
            
        edges, raw = _optimize_edges(vals, 35, deadline)
        
        if edges and raw < best_raw:
            best_raw = raw
            best_result = {'values': vals, 'edges': edges}
            
    return best_result, best_raw

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin
    
    best_result, best_raw = _optimize_values_and_edges(deadline)
    
    if best_result is None:
        # Fallback
        vals = list(range(1, 15))
        edges = list(combinations(range(14), 2))[:35]
        best_result = {'values': vals, 'edges': edges}
        
    # Format output
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
