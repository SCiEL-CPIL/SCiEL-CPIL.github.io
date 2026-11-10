"""Improved program for hypergraph Ramsey R(4,5;3) task."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit

    def get_precomputed_indices(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4 triple indices
        k4_triples = []
        for k4 in combinations(range(n), 4):
            ts = []
            for i in range(4):
                t = tuple(sorted(k4[:i] + k4[i+1:]))
                ts.append(triple_to_idx[t])
            k4_triples.append(ts)
        k4_arr = np.array(k4_triples, dtype=np.int32)
        
        # Precompute K5 triple indices
        k5_triples = []
        for k5 in combinations(range(n), 5):
            ts = []
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[i] for i in combo]))
                ts.append(triple_to_idx[t])
            k5_triples.append(ts)
        k5_arr = np.array(k5_triples, dtype=np.int32)
        
        return triples, num_triples, k4_arr, k5_arr

    def count_violations_vec(colors, k4_arr, k5_arr):
        """
        colors: 1D NumPy array of 0s and 1s.
        k4_arr: (num_k4, 4) array of triple indices.
        k5_arr: (num_k5, 10) array of triple indices.
        Returns total number of monochromatic K4 (red) and K5 (blue).
        """
        # Red K4: all 4 triples are 0
        red_k4_count = np.sum(colors[k4_arr], axis=1)
        red_violations = np.sum(red_k4_count == 0)
        
        # Blue K5: all 10 triples are 1
        blue_k5_count = np.sum(colors[k5_arr], axis=1)
        blue_violations = np.sum(blue_k5_count == 10)
        
        return int(red_violations + blue_violations)

    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # We want to maximize n. Published construction is 34.
    # Try from 34 down to 15, using SA with multiple restarts.
    # For each n, allocate remaining time / remaining_n_values, with a minimum of 3s per n.
    
    n_values = list(range(34, 14, -1))
    num_n_values = len(n_values)
    
    for idx, n in enumerate(n_values):
        if time.time() > deadline:
            break
            
        remaining_time = deadline - time.time()
        remaining_n = num_n_values - idx
        
        # Allocate time: at least 3s per n, but if we have extra, give more to current n
        time_budget = max(3.0, remaining_time / max(1, remaining_n) * 1.5)
        time_budget = min(time_budget, remaining_time - 1.0)  # leave 1s for next iterations
        
        if time_budget < 1.0:
            break
            
        n_start_time = time.time()
        
        # Get precomputed indices
        try:
            triples, num_triples, k4_arr, k5_arr = get_precomputed_indices(n)
        except MemoryError:
            # If n is too large for memory, skip to smaller n
            continue
            
        # Run SA with multiple restarts
        found_valid = False
        num_restarts = 3  # Try up to 3 restarts for each n
        
        for restart in range(num_restarts):
            if time.time() > n_start_time + time_budget:
                break
                
            # Initialize colors
            colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
            
            # Simulated Annealing
            current_v = count_violations_vec(colors, k4_arr, k5_arr)
            best_v = current_v
            best_cols = colors.copy()
            
            # SA Parameters
            T = max(10.0, num_triples * 0.01)
            T_end = 0.01
            cooling_rate = 0.998
            no_improve_steps = 0
            max_no_improve = 5000
            
            while time.time() < n_start_time + time_budget:
                # Pick random triple
                idx_trip = random.randint(0, num_triples - 1)
                
                # Flip
                old_val = colors[idx_trip]
                colors[idx_trip] = 1 - old_val
                
                new_v = count_violations_vec(colors, k4_arr, k5_arr)
                delta_v = new_v - current_v
                
                if delta_v <= 0:
                    # Improvement or equal
                    current_v = new_v
                    if new_v < best_v:
                        best_v = new_v
                        best_cols = colors.copy()
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    # Worse move
                    if T > T_end:
                        prob = np.exp(-delta_v / T)
                        if random.random() < prob:
                            current_v = new_v
                        else:
                            # Revert
                            colors[idx_trip] = old_val
                    else:
                        # Revert
                        colors[idx_trip] = old_val
                    no_improve_steps += 1
                
                # Cool down
                T *= cooling_rate
                
                if best_v == 0:
                    break
                    
                # Restart if stuck too long
                if no_improve_steps > max_no_improve:
                    T = max(10.0, num_triples * 0.01)
                    no_improve_steps = 0
                    # Randomize some fraction of colors to escape local minimum
                    num_random = max(1, num_triples // 10)
                    rand_indices = np.random.choice(num_triples, size=num_random, replace=False)
                    colors[rand_indices] = np.random.randint(0, 2, size=num_random).astype(np.int8)
                    current_v = count_violations_vec(colors, k4_arr, k5_arr)
                
            if best_v == 0:
                # Found valid coloring for n
                best_n = n
                best_colors = best_cols.tolist()
                found_valid = True
                break
                
        if found_valid:
            # We found a valid coloring for n, try n-1? No, we want to maximize n.
            # But we're iterating from high to low, so if we find n=34 valid, that's best.
            # If we don't find 34, we try 33, etc.
            # Since we found a valid one for this n, and we're going down, 
            # we should keep this as best and stop (since smaller n won't beat it).
            # Actually, we want the MAXIMUM n, so if we find n=30 valid, that's good.
            # But maybe we could have found n=34 with more time?
            # The loop goes 34, 33, 32, ... so if we find 33 valid, we've already failed on 34.
            # So we should stop here since any smaller n won't beat n=33.
            break
            
    # Fallback: ensure we have at least n=9
    if best_n < 9:
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        best_n = n
    
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
