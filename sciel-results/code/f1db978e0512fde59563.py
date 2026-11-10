import time
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    n = 14
    m = 35
    maximum = 10000
    
    def raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def all_edges():
        return [list(e) for e in combinations(range(n), 2)]

    # Candidate value sets
    candidates_vals = []
    
    # 1. Powers of 2
    candidates_vals.append([2**i for i in range(14)])
    
    # 2. Arithmetic Progression
    candidates_vals.append([1 + i*10 for i in range(14)]) # Smaller range for more collisions
    
    # 3. Small integers
    candidates_vals.append(list(range(1, 15)))
    
    # 4. Symmetric around center
    # 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140
    candidates_vals.append([10 * (i + 1) for i in range(14)])

    # 5. Geometric with ratio 3 (small)
    # 1, 3, 9, 27, 81, 243, 729, 2187, 6561 (only 9 fit well, need 14 distinct)
    # Let's stick to AP and small ints which are better for sum collisions.

    best_result = None
    best_score = float('inf')
    
    all_possible_edges = all_edges()
    
    # Pre-compute sums and products for all possible edges for each candidate?
    # It's better to do it on the fly or pre-compute for speed in local search.
    
    for vals in candidates_vals:
        # Check validity of vals
        if len(set(vals)) != n: continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals): continue
        if not all(1 <= v <= maximum for v in vals): continue
        
        # Pre-compute sums and products for all 91 edges
        edge_data = []
        for e in all_possible_edges:
            i, j = e
            s = vals[i] + vals[j]
            p = vals[i] * vals[j]
            edge_data.append((e, s, p))
        
        # Local Search to find best 35 edges
        # Initialize with a random subset of 35 edges
        # Or greedy: start with edges that have "central" sums/products?
        
        # Let's use a simple local search with random restarts
        num_restarts = 50
        steps_per_restart = 100
        
        for r in range(num_restarts):
            # Random initialization
            selected_indices = set(random.sample(range(len(all_possible_edges)), m))
            selected_edges = [all_possible_edges[i] for i in selected_indices]
            
            # Calculate initial score
            cur_sums = set()
            cur_prods = set()
            for e in selected_edges:
                i, j = e
                cur_sums.add(vals[i] + vals[j])
                cur_prods.add(vals[i] * vals[j])
            cur_score = max(len(cur_sums), len(cur_prods))
            
            # Local search
            improved = True
            while improved and (time.time() < deadline):
                improved = False
                # Try swapping one edge in with one edge out
                # To be fast, we can just pick random in/out pairs
                for _ in range(steps_per_restart):
                    if time.time() > deadline:
                        break
                    
                    # Pick a random edge to remove
                    idx_out = random.choice(list(selected_indices))
                    # Pick a random edge to add
                    available = [i for i in range(len(all_possible_edges)) if i not in selected_indices]
                    if not available: continue
                    idx_in = random.choice(available)
                    
                    # Simulate swap
                    e_out = all_possible_edges[idx_out]
                    e_in = all_possible_edges[idx_in]
                    
                    # Recalculate sums/prods efficiently? 
                    # For 35 edges, full recalc is 35 ops, which is very fast.
                    # Let's just recalculate.
                    new_edges = [all_possible_edges[i] for i in selected_indices if i != idx_out]
                    new_edges.append(all_possible_edges[idx_in])
                    
                    new_sums = set()
                    new_prods = set()
                    for e in new_edges:
                        i, j = e
                        new_sums.add(vals[i] + vals[j])
                        new_prods.add(vals[i] * vals[j])
                    new_score = max(len(new_sums), len(new_prods))
                    
                    if new_score < cur_score:
                        cur_score = new_score
                        selected_indices.remove(idx_out)
                        selected_indices.add(idx_in)
                        improved = True
                        # Update current sets for next iteration? 
                        # Or just rely on recalculation in next loop? 
                        # Recalculation is cheap.
                        # But we need `cur_sums` and `cur_prods` to be consistent if we use them.
                        # Since we recalc every time, we can just update the score.
                        # However, the `new_edges` list construction is O(m).
                        # We can optimize by keeping the sets and updating them incrementally, 
                        # but with m=35, full recalc is < 1ms.
                        
                    # If we found improvement, break inner loop to restart random picks?
                    # No, just continue.
                    
            # End local search for this restart
            
            if cur_score < best_score:
                best_score = cur_score
                best_edges = [all_possible_edges[i] for i in selected_indices]
                best_vals = vals

    if best_result is None:
        # Fallback: return something valid
        best_vals = candidates_vals[0]
        best_edges = all_possible_edges[:35]
        
    # Final validation
    assert len(best_vals) == n
    assert len(set(best_vals)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in best_vals)
    assert all(1 <= v <= maximum for v in best_vals)
    assert len(best_edges) == m
    assert len(set(tuple(sorted(e)) for e in best_edges)) == m
    
    return {'values': list(best_vals), 'edges': [list(e) for e in best_edges]}
