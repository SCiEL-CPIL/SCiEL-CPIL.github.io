import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 34
    
    # Precompute indices for fast lookup
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    k4_list = list(combinations(range(n), 4))
    num_k4 = len(k4_list)
    k5_list = list(combinations(range(n), 5))
    num_k5 = len(k5_list)
    
    k4_data = []
    triple_to_k4s = [[] for _ in range(num_triples)]
    for i, k4 in enumerate(k4_list):
        ts = []
        for j in range(4):
            t = tuple(sorted(k4[:j] + k4[j+1:]))
            idx = triple_to_idx[t]
            ts.append(idx)
            triple_to_k4s[idx].append(i)
        k4_data.append(ts)
        
    k5_data = []
    triple_to_k5s = [[] for _ in range(num_triples)]
    for i, k5 in enumerate(k5_list):
        ts = []
        for combo in combinations(range(5), 3):
            t = tuple(sorted([k5[j] for j in combo]))
            idx = triple_to_idx[t]
            ts.append(idx)
            triple_to_k5s[idx].append(i)
        k5_data.append(ts)

    k4_arr = np.array(k4_data, dtype=np.int32)
    k5_arr = np.array(k5_data, dtype=np.int32)
    
    # Local variables for speed in inner loop
    t2k4 = triple_to_k4s
    t2k5 = triple_to_k5s
    
    def init_state(colors):
        counts_k4 = np.sum(colors[k4_arr], axis=1)
        counts_k5 = np.sum(colors[k5_arr], axis=1)
        violations = np.sum(counts_k4 == 4) + np.sum(counts_k5 == 10)
        return counts_k4, counts_k5, violations

    def flip_state(colors, counts_k4, counts_k5, violations, idx):
        old_val = colors[idx]
        new_val = 1 - old_val
        delta = new_val - old_val
        
        # Update K4s
        for k4_idx in t2k4[idx]:
            old_count = counts_k4[k4_idx]
            new_count = old_count + delta
            counts_k4[k4_idx] = new_count
            old_viol = 1 if old_count == 4 else 0
            new_viol = 1 if new_count == 4 else 0
            violations += (new_viol - old_viol)
            
        # Update K5s
        for k5_idx in t2k5[idx]:
            old_count = counts_k5[k5_idx]
            new_count = old_count + delta
            counts_k5[k5_idx] = new_count
            old_viol = 1 if old_count == 10 else 0
            new_viol = 1 if new_count == 10 else 0
            violations += (new_viol - old_viol)
            
        colors[idx] = new_val
        return violations

    best_violations = float('inf')
    best_colors = None
    
    # Min-Conflicts / Random Walk Hybrid
    max_restarts = 2000
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        counts_k4, counts_k5, violations = init_state(colors)
        
        # Local Search
        max_steps = 2000
        no_improve = 0
        best_viol_in_run = violations
        best_col_in_run = colors.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if violations == 0:
                best_violations = 0
                best_colors = colors.copy()
                break
                
            if violations < best_viol_in_run:
                best_viol_in_run = violations
                best_col_in_run = colors.copy()
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve > 500:
                break # Stuck
                
            # Find a random violation
            # Use np.where efficiently
            red_k4s = np.where(counts_k4 == 4)[0]
            blue_k5s = np.where(counts_k5 == 10)[0]
            
            if len(red_k4s) == 0 and len(blue_k5s) == 0:
                break # Should be 0 violations
            
            # Pick one
            if len(red_k4s) > 0 and (len(blue_k5s) == 0 or random.random() < len(red_k4s) / (len(red_k4s) + len(blue_k5s))):
                k4_idx = red_k4s[random.randint(0, len(red_k4s) - 1)]
                edges = k4_data[k4_idx]
            else:
                k5_idx = blue_k5s[random.randint(0, len(blue_k5s) - 1)]
                edges = k5_data[k5_idx]
                
            # Try up to 3 random edges from this violation
            best_delta = float('inf')
            best_edge = None
            
            # Shuffle edges to pick random ones
            # Since edges list is small (4 or 10), we can just pick random indices
            # To avoid duplicates in the 3 tries, we can use a set or just pick with replacement
            # and check if it's better.
            
            for _ in range(3):
                edge_idx = edges[random.randint(0, len(edges) - 1)]
                
                old_viol = violations
                new_viol = flip_state(colors, counts_k4, counts_k5, violations, edge_idx)
                delta = new_viol - old_viol
                
                if delta < best_delta:
                    best_delta = delta
                    best_edge = edge_idx
                    
                # Revert
                flip_state(colors, counts_k4, counts_k5, violations, edge_idx)
                
            # Apply best edge
            if best_edge is not None:
                flip_state(colors, counts_k4, counts_k5, violations, best_edge)
                
                # If best_delta > 0, we worsened. 
                # To escape, do a global random flip
                if best_delta > 0:
                    # We are in a local minimum relative to this violation's edges.
                    # Global random flip to jump.
                    global_edge = random.randint(0, num_triples - 1)
                    flip_state(colors, counts_k4, counts_k5, violations, global_edge)
            else:
                # No edges tried? Should not happen.
                break
                
        # End local search
        if best_viol_in_run < best_violations:
            best_violations = best_viol_in_run
            best_colors = best_col_in_run.copy()
            
        if best_violations == 0:
            break

    if best_violations == 0:
        return {'n': 34, 'colors': best_colors.tolist()}
    else:
        # Fallback
        n_fallback = 9
        colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fallback), 3)]
        return {'n': n_fallback, 'colors': colors_fallback}
