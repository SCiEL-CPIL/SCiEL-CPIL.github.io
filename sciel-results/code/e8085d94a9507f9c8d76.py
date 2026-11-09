"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _score(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    n = 14
    m = 35
    max_val = 10000
    
    rng = random.Random(42)
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Candidate base value sets
    base_sets = [
        list(range(1, n + 1)),
        list(range(2, n + 2)),
        list(range(1, n + 1, 2)) + list(range(n - 1, 1, -2)), # odd then even? no, just distinct
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], # Powers of 2
        [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377], # Fibonacci (remove dup 1)
    ]
    
    # Ensure distinctness and validity for base sets
    valid_base_sets = []
    for bs in base_sets:
        if len(set(bs)) == n and all(1 <= v <= max_val for v in bs):
            valid_base_sets.append(bs)
    
    if not valid_base_sets:
        valid_base_sets.append(list(range(1, n + 1)))
    
    all_possible_edges = list(combinations(range(n), 2))
    
    # Simulated Annealing parameters
    T_start = 10.0
    T_end = 0.01
    alpha = 0.9999
    
    for base_vals in valid_base_sets:
        if time.time() > deadline:
            break
            
        # Initialize with random permutation of base values
        vals = base_vals[:]
        rng.shuffle(vals)
        
        # Initialize with random edges
        edges = [set(e) for e in rng.sample(all_possible_edges, m)]
        edge_set = set(edges)
        
        current_score = _score(vals, edges)
        best_score_in_run = current_score
        best_vals_in_run = vals[:]
        best_edges_in_run = [sorted(e) for e in edges]
        
        T = T_start
        iter_count = 0
        max_iters = 50000 # Adjust based on time
        
        while T > T_end and time.time() < deadline:
            iter_count += 1
            move_type = rng.random()
            
            if move_type < 0.5:
                # Swap two values
                i, j = rng.sample(range(n), 2)
                old_vi, old_vj = vals[i], vals[j]
                vals[i], vals[j] = old_vj, old_vi
                
                new_score = _score(vals, edges)
                
                delta = new_score - current_score
                if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                    current_score = new_score
                    if new_score < best_score_in_run:
                        best_score_in_run = new_score
                        best_vals_in_run = vals[:]
                        best_edges_in_run = [sorted(e) for e in edges]
                else:
                    vals[i], vals[j] = old_vi, old_vj
            else:
                # Swap an edge
                if not edges:
                    continue
                rem_idx = rng.randrange(len(edges))
                rem_edge = edges[rem_idx]
                
                # Find a random new edge not in the set
                # Optimization: pick random from all, check if in set
                new_edge = None
                for _ in range(100):
                    cand = set(rng.choice(all_possible_edges))
                    if cand not in edge_set:
                        new_edge = cand
                        break
                
                if new_edge is None:
                    continue
                
                # Apply move
                edges[rem_idx] = new_edge
                edge_set.remove(rem_edge)
                edge_set.add(new_edge)
                
                new_score = _score(vals, edges)
                
                delta = new_score - current_score
                if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                    current_score = new_score
                    if new_score < best_score_in_run:
                        best_score_in_run = new_score
                        best_vals_in_run = vals[:]
                        best_edges_in_run = [sorted(e) for e in edges]
                else:
                    # Revert
                    edges[rem_idx] = rem_edge
                    edge_set.remove(new_edge)
                    edge_set.add(rem_edge)
            
            T *= alpha
        
        # Update global best
        if best_score_in_run < best_score:
            best_score = best_score_in_run
            best_values = best_vals_in_run[:]
            best_edges = [list(e) for e in best_edges_in_run]
            
        # Early stop if perfect or very good
        if best_score <= 20:
            break

    # Fallback
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = [list(e) for e in all_possible_edges[:m]]
        best_score = _score(best_values, best_edges)
        
    # Final validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= max_val for v in best_values)
    assert len(best_edges) == m
    edge_set = set()
    for e in best_edges:
        assert len(e) == 2
        i, j = e
        assert isinstance(i, int) and isinstance(j, int)
        assert i != j
        assert 0 <= i < n and 0 <= j < n
        key = (min(i, j), max(i, j))
        assert key not in edge_set
        edge_set.add(key)
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
