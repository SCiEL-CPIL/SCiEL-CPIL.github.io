import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Try n=34 first
    for n in [34, 33, 32]:
        if time.time() > deadline:
            break
            
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4_list = list(combinations(range(n), 4))
        k5_list = list(combinations(range(n), 5))
        num_k4 = len(k4_list)
        num_k5 = len(k5_list)
        
        # Precompute incidence
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        for i, k4 in enumerate(k4_list):
            for j in range(4):
                t = tuple(sorted(k4[:j] + k4[j+1:]))
                idx = triple_to_idx[t]
                triple_to_k4s[idx].append(i)
                
        for i, k5 in enumerate(k5_list):
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[j] for j in combo]))
                idx = triple_to_idx[t]
                triple_to_k5s[idx].append(i)
                
        # Greedy Initialization
        colors = np.zeros(num_triples, dtype=np.int8)
        k4_counts = np.zeros(num_k4, dtype=np.int32)
        k5_counts = np.zeros(num_k5, dtype=np.int32)
        
        for t_idx in range(num_triples):
            delta_red = 0
            for k4_idx in triple_to_k4s[t_idx]:
                if k4_counts[k4_idx] == 3:
                    delta_red += 1
            delta_blue = 0
            for k5_idx in triple_to_k5s[t_idx]:
                if k5_counts[k5_idx] == 9:
                    delta_blue += 1
            
            if delta_red < delta_blue:
                colors[t_idx] = 0
                for k4_idx in triple_to_k4s[t_idx]:
                    k4_counts[k4_idx] += 1
            else:
                colors[t_idx] = 1
                for k5_idx in triple_to_k5s[t_idx]:
                    k5_counts[k5_idx] += 1
                    
        energy = np.sum(k4_counts == 4) + np.sum(k5_counts == 10)
        
        if energy == 0:
            return {'n': n, 'colors': colors.tolist()}
            
        best_energy = energy
        best_colors = colors.copy()
        
        # Simulated Annealing
        # We will run multiple restarts of SA
        max_restarts = 10
        for r in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # If not first restart, perturb best
            if r > 0:
                colors = best_colors.copy()
                k4_counts = np.zeros(num_k4, dtype=np.int32)
                k5_counts = np.zeros(num_k5, dtype=np.int32)
                for t_idx in range(num_triples):
                    if colors[t_idx] == 0:
                        for k4_idx in triple_to_k4s[t_idx]:
                            k4_counts[k4_idx] += 1
                    else:
                        for k5_idx in triple_to_k5s[t_idx]:
                            k5_counts[k5_idx] += 1
                energy = np.sum(k4_counts == 4) + np.sum(k5_counts == 10)
                
                # Perturb
                for _ in range(50):
                    idx = random.randint(0, num_triples - 1)
                    colors[idx] = 1 - colors[idx]
                    # Update counts manually for perturbation is slow, so we just recount?
                    # No, we should update. But for simplicity in restart, let's just recount.
                # Recount after perturbation
                k4_counts = np.zeros(num_k4, dtype=np.int32)
                k5_counts = np.zeros(num_k5, dtype=np.int32)
                for t_idx in range(num_triples):
                    if colors[t_idx] == 0:
                        for k4_idx in triple_to_k4s[t_idx]:
                            k4_counts[k4_idx] += 1
                    else:
                        for k5_idx in triple_to_k5s[t_idx]:
                            k5_counts[k5_idx] += 1
                energy = np.sum(k4_counts == 4) + np.sum(k5_counts == 10)
                
            T = 5.0
            cooling = 0.9995
            min_T = 0.01
            steps = 0
            max_steps = 50000 # Adjust based on time
            
            while steps < max_steps and time.time() < deadline - 5.0:
                idx = random.randint(0, num_triples - 1)
                
                old_val = colors[idx]
                new_val = 1 - old_val
                
                delta_E = 0
                delta_red = 1 if new_val == 0 else -1
                for k4_idx in triple_to_k4s[idx]:
                    old_c = k4_counts[k4_idx]
                    new_c = old_c + delta_red
                    delta_E += (1 if new_c == 4 else 0) - (1 if old_c == 4 else 0)
                    
                delta_blue = 1 if new_val == 1 else -1
                for k5_idx in triple_to_k5s[idx]:
                    old_c = k5_counts[k5_idx]
                    new_c = old_c + delta_blue
                    delta_E += (1 if new_c == 10 else 0) - (1 if old_c == 10 else 0)
                    
                if delta_E < 0 or (T > 0 and random.random() < np.exp(-delta_E / T)):
                    colors[idx] = new_val
                    for k4_idx in triple_to_k4s[idx]:
                        k4_counts[k4_idx] += delta_red
                    for k5_idx in triple_to_k5s[idx]:
                        k5_counts[k5_idx] += delta_blue
                    energy += delta_E
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_colors = colors.copy()
                        
                    if energy == 0:
                        return {'n': n, 'colors': colors.tolist()}
                        
                T = max(min_T, T * cooling)
                steps += 1
                
        if best_energy == 0:
            return {'n': n, 'colors': best_colors.tolist()}
            
    # Fallback
    n_fallback = 9
    colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fallback), 3)]
    return {'n': n_fallback, 'colors': colors_fallback}
