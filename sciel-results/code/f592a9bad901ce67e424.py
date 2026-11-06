import random
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def try_find_coloring(n, time_limit):
        triples = list(itertools.combinations(range(n), 3))
        m = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        four_sets = list(itertools.combinations(range(n), 4))
        five_sets = list(itertools.combinations(range(n), 5))
        
        # Mapping: triple_index -> array of 4-set indices
        # Mapping: triple_index -> array of 5-set indices
        triples_in_four = [np.array([], dtype=np.int32) for _ in range(m)]
        triples_in_five = [np.array([], dtype=np.int32) for _ in range(m)]
        
        # We need to build these arrays efficiently.
        # List of lists first, then convert to arrays.
        t_in_f_list = [[] for _ in range(m)]
        t_in_5_list = [[] for _ in range(m)]
        
        for i, s in enumerate(four_sets):
            # 4 triples in s
            for j in range(4):
                t = (s[0], s[1], s[2]) if j==0 else (s[0], s[1], s[3]) if j==1 else (s[0], s[2], s[3]) if j==2 else (s[1], s[2], s[3])
                ti = triple_idx[t]
                t_in_f_list[ti].append(i)
                
        for i, s in enumerate(five_sets):
            for t in itertools.combinations(s, 3):
                ti = triple_idx[t]
                t_in_5_list[ti].append(i)
                
        for i in range(m):
            triples_in_four[i] = np.array(t_in_f_list[i], dtype=np.int32)
            triples_in_five[i] = np.array(t_in_5_list[i], dtype=np.int32)

        colors = np.random.randint(0, 2, m, dtype=np.int8)
        
        red_count_4 = np.zeros(len(four_sets), dtype=np.int8)
        blue_count_5 = np.zeros(len(five_sets), dtype=np.int8)
        
        # Initial counts
        for t_idx, t in enumerate(triples):
            c = colors[t_idx]
            if c == 0:
                red_count_4[triples_in_four[t_idx]] += 1
            else:
                blue_count_5[triples_in_five[t_idx]] += 1
                
        energy = np.sum(red_count_4 == 4) + np.sum(blue_count_5 == 10)
        
        if energy == 0:
            return {'n': n, 'colors': colors.tolist()}
            
        best_energy = energy
        best_colors = colors.copy()
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() < start_time + time_limit:
            if T < min_T:
                T = min_T
            
            t_idx = random.randint(0, m - 1)
            old_c = colors[t_idx]
            new_c = 1 - old_c
            
            delta_e = 0
            
            if old_c == 0: # Red -> Blue
                # Red counts decrease
                idx4s = triples_in_four[t_idx]
                old_vals = red_count_4[idx4s]
                # Count how many were 4
                delta_e -= np.sum(old_vals == 4)
                
                # Blue counts increase
                idx5s = triples_in_five[t_idx]
                old_vals = blue_count_5[idx5s]
                # Count how many were 9
                delta_e += np.sum(old_vals == 9)
                
            else: # Blue -> Red
                # Blue counts decrease
                idx5s = triples_in_five[t_idx]
                old_vals = blue_count_5[idx5s]
                # Count how many were 10
                delta_e -= np.sum(old_vals == 10)
                
                # Red counts increase
                idx4s = triples_in_four[t_idx]
                old_vals = red_count_4[idx4s]
                # Count how many were 3
                delta_e += np.sum(old_vals == 3)
                
            new_energy = energy + delta_e
            
            if new_energy < energy or (new_energy > energy and random.random() < np.exp(-(new_energy - energy) / T)):
                colors[t_idx] = new_c
                energy = new_energy
                
                if old_c == 0:
                    red_count_4[triples_in_four[t_idx]] -= 1
                    blue_count_5[triples_in_five[t_idx]] += 1
                else:
                    blue_count_5[triples_in_five[t_idx]] -= 1
                    red_count_4[triples_in_four[t_idx]] += 1
                    
                if energy < best_energy:
                    best_energy = energy
                    best_colors = colors.copy()
                    if energy == 0:
                        return {'n': n, 'colors': colors.tolist()}
                        
            T *= cooling
            
        return None

    # Try n=34
    result = try_find_coloring(34, 120.0)
    if result:
        return result
        
    # Fallback
    from itertools import combinations
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
