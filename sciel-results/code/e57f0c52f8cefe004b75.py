import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Keep well under 90s

    n = 14
    m = 35
    
    # Candidate value sets
    value_sets = [
        list(range(1, n + 1)),
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16],
        [6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 26],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
    ]

    best_result = None
    best_score = float('inf')

    for values in value_sets:
        if time.time() > deadline - 5:
            break
            
        # Precompute edge properties
        all_edges = []
        edge_sums = []
        edge_prods = []
        for i in range(n):
            for j in range(i + 1, n):
                u, v = values[i], values[j]
                s = u + v
                p = u * v
                all_edges.append((i, j))
                edge_sums.append(s)
                edge_prods.append(p)
                
        num_total_edges = len(all_edges)
        
        # Run multiple SA restarts for each value set
        num_restarts = 5
        steps_per_restart = 200000 # Adjust based on time
        
        for r in range(num_restarts):
            if time.time() > deadline - 2:
                break
                
            # Random initial set of m edges
            init_indices = random.sample(range(num_total_edges), m)
            current_edges = set(init_indices)
            
            # Counters
            sum_counts = {}
            prod_counts = {}
            active_sums = set()
            active_prods = set()
            
            for idx in current_edges:
                s = edge_sums[idx]
                p = edge_prods[idx]
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1
                active_sums.add(s)
                active_prods.add(p)
                
            current_score = max(len(active_sums), len(active_prods))
            
            # Simulated Annealing
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            
            # Precompute list of edge indices for faster access
            all_indices = list(range(num_total_edges))
            
            for step in range(steps_per_restart):
                if time.time() > deadline - 1:
                    break
                    
                # Pick a random edge to remove
                e_out = random.choice(list(current_edges))
                
                # Pick a random edge to add (not in current)
                e_in = random.choice(all_indices)
                if e_in in current_edges:
                    continue
                    
                # Calculate delta score
                s_out = edge_sums[e_out]
                p_out = edge_prods[e_out]
                s_in = edge_sums[e_in]
                p_in = edge_prods[e_in]
                
                # Tentative removal
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0:
                    active_sums.discard(s_out)
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0:
                    active_prods.discard(p_out)
                
                # Tentative addition
                sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
                if sum_counts[s_in] == 1:
                    active_sums.add(s_in)
                prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
                if prod_counts[p_in] == 1:
                    active_prods.add(p_in)
                
                new_score = max(len(active_sums), len(active_prods))
                
                if new_score <= current_score:
                    # Accept
                    current_edges.remove(e_out)
                    current_edges.add(e_in)
                    current_score = new_score
                else:
                    # Reject? Or accept with prob?
                    if random.random() < math.exp(-(new_score - current_score) / T):
                        current_edges.remove(e_out)
                        current_edges.add(e_in)
                        current_score = new_score
                    else:
                        # Revert
                        sum_counts[s_in] -= 1
                        if sum_counts[s_in] == 0:
                            active_sums.discard(s_in)
                        prod_counts[p_in] -= 1
                        if prod_counts[p_in] == 0:
                            active_prods.discard(p_in)
                        
                        sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
                        if sum_counts[s_out] == 1:
                            active_sums.add(s_out)
                        prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
                        if prod_counts[p_out] == 1:
                            active_prods.add(p_out)
                            
                T *= cooling
                if T < min_T:
                    T = min_T

            # Update global best
            if current_score < best_score:
                best_score = current_score
                result_edges = [ [all_edges[idx][0], all_edges[idx][1]] for idx in current_edges ]
                # Ensure types are correct
                best_result = {
                    'values': [int(v) for v in values],
                    'edges': result_edges
                }
                
    if best_result is None:
        # Fallback
        values = list(range(1, n + 1))
        edges = [ [i, j] for i in range(n) for j in range(i+1, n) if i+j < 15 ] # Just some edges
        # Ensure 35 edges
        all_e = [ [i, j] for i in range(n) for j in range(i+1, n) ]
        random.shuffle(all_e)
        edges = all_e[:35]
        best_result = {
            'values': values,
            'edges': edges
        }

    return best_result
