"""
Maximize number of triples in a 3-uniform hypergraph on 16 vertices
such that no three disjoint pairs have all 8 transversal triples present.

Strategy:
1. Precompute all K(2,2,2) configurations (each is a set of 8 triples).
2. Use simulated annealing with temperature-controlled swaps and periodic
   greedy restarts to escape local optima.
3. Use degree-based greedy initialization for diverse starting points.
4. Time-bounded with multiple restarts.
"""

import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_total = len(all_triples)  # 560
    
    # Precompute all K(2,2,2) configurations
    k222_configs = []  # list of frozensets of 8 triple indices
    triple_to_k222 = [[] for _ in range(num_total)]
    
    for six_vertices in itertools.combinations(range(N), 6):
        a, b, c, d, e, f = six_vertices
        partitions = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for p1, p2, p3 in partitions:
            transversals = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        transversals.append(t)
            trans_indices = frozenset(triple_to_idx[t] for t in transversals)
            k222_configs.append(trans_indices)
            config_idx = len(k222_configs) - 1
            for ti in trans_indices:
                triple_to_k222[ti].append(config_idx)
    
    num_k222 = len(k222_configs)
    
    def time_check():
        return time.time() >= deadline
    
    best_result = []
    best_count = 0
    
    # Greedy initialization
    def greedy_init(rng):
        present = [False] * num_total
        k222_count = [0] * num_k222
        count = 0
        
        order = list(range(num_total))
        rng.shuffle(order)
        
        for ti in order:
            can_add = True
            for ci in triple_to_k222[ti]:
                if k222_count[ci] == 7:
                    can_add = False
                    break
            if can_add:
                present[ti] = True
                count += 1
                for ci in triple_to_k222[ti]:
                    k222_count[ci] += 1
        return present, k222_count, count
    
    # Simulated annealing
    def simulated_annealing(present, k222_count, count, rng, max_steps=500000):
        steps = 0
        T = 1.0
        T_min = 0.01
        cooling = 0.99995
        best_local = count
        best_local_present = present.copy()
        best_local_k222 = k222_count.copy()
        no_improve = 0
        max_no_improve = 50000
        
        while steps < max_steps and no_improve < max_no_improve:
            if time_check():
                break
            
            # Try to add a triple (always accept if possible)
            moved = False
            for _ in range(30):
                ti = rng.randint(0, num_total - 1)
                if present[ti]:
                    continue
                can_add = True
                for ci in triple_to_k222[ti]:
                    if k222_count[ci] == 7:
                        can_add = False
                        break
                if can_add:
                    present[ti] = True
                    count += 1
                    for ci in triple_to_k222[ti]:
                        k222_count[ci] += 1
                    moved = True
                    no_improve = 0
                    if count > best_local:
                        best_local = count
                        best_local_present = present.copy()
                        best_local_k222 = k222_count.copy()
                    break
            
            if moved:
                steps += 1
                T = max(T_min, T * cooling)
                continue
            
            # Try a swap: remove one triple, add another
            swap_done = False
            for _ in range(30):
                ti_remove = rng.randint(0, num_total - 1)
                if not present[ti_remove]:
                    continue
                
                # Remove
                for ci in triple_to_k222[ti_remove]:
                    k222_count[ci] -= 1
                present[ti_remove] = False
                count -= 1
                
                # Try to add a different triple
                found_add = False
                ti_add = -1
                for _ in range(50):
                    ti_candidate = rng.randint(0, num_total - 1)
                    if present[ti_candidate] or ti_candidate == ti_remove:
                        continue
                    can_add = True
                    for ci in triple_to_k222[ti_candidate]:
                        if k222_count[ci] == 7:
                            can_add = False
                            break
                    if can_add:
                        ti_add = ti_candidate
                        found_add = True
                        break
                
                if found_add:
                    # Accept swap (neutral move)
                    present[ti_add] = True
                    count += 1
                    for ci in triple_to_k222[ti_add]:
                        k222_count[ci] += 1
                    swap_done = True
                    no_improve = 0
                else:
                    # Revert removal
                    for ci in triple_to_k222[ti_remove]:
                        k222_count[ci] += 1
                    present[ti_remove] = True
                    count += 1
            
            if swap_done:
                steps += 1
                T = max(T_min, T * cooling)
                continue
            
            # If we couldn't add or swap, try a random perturbation
            # Remove a random triple to break free
            perturbed = False
            for _ in range(20):
                ti_remove = rng.randint(0, num_total - 1)
                if not present[ti_remove]:
                    continue
                
                # Remove
                for ci in triple_to_k222[ti_remove]:
                    k222_count[ci] -= 1
                present[ti_remove] = False
                count -= 1
                perturbed = True
                no_improve += 1
                break
            
            if not perturbed:
                no_improve += 1
            
            steps += 1
            T = max(T_min, T * cooling)
        
        # Restore best local solution
        if best_local > count:
            present = best_local_present
            k222_count = best_local_k222
            count = best_local
        
        return present, k222_count, count
    
    # Main loop with multiple restarts
    rng = random.Random(42)
    
    for restart in range(300):
        if time_check():
            break
        
        # Greedy init with randomization
        present, k222_count, count = greedy_init(rng)
        
        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]
        
        # Simulated annealing
        present, k222_count, count = simulated_annealing(present, k222_count, count, rng)
        
        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]
        
        if time_check():
            break
    
    if not best_result:
        # Fallback: return a simple valid solution
        best_result = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_result
