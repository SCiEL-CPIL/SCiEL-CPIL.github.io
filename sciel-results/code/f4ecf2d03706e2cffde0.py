import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Target n: Try 34 down to 25
    # The published construction has 34 points.
    # We aim to find it.
    
    best_n = 0
    best_colors = None

    for n in range(34, 24, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Precompute triples
        triples = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triples.append((a, b, c))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4 sets (4 vertices, 4 triples)
        # k4_triples[i] = list of triple indices in the i-th K4
        k4_triples = []
        # Map triple index to list of K4 indices
        t_to_k4 = [[] for _ in range(num_triples)]
        
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        idxs = [
                            triple_index[(a,b,c)],
                            triple_index[(a,b,d)],
                            triple_index[(a,c,d)],
                            triple_index[(b,c,d)]
                        ]
                        k4_id = len(k4_triples)
                        k4_triples.append(idxs)
                        for idx in idxs:
                            t_to_k4[idx].append(k4_id)
                            
        num_k4 = len(k4_triples)
        
        # Precompute K5 sets (5 vertices, 10 triples)
        k5_triples = []
        t_to_k5 = [[] for _ in range(num_triples)]
        
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        for e in range(d+1, n):
                            idxs = [
                                triple_index[(a,b,c)],
                                triple_index[(a,b,d)],
                                triple_index[(a,b,e)],
                                triple_index[(a,c,d)],
                                triple_index[(a,c,e)],
                                triple_index[(a,d,e)],
                                triple_index[(b,c,d)],
                                triple_index[(b,c,e)],
                                triple_index[(b,d,e)],
                                triple_index[(c,d,e)]
                            ]
                            k5_id = len(k5_triples)
                            k5_triples.append(idxs)
                            for idx in idxs:
                                t_to_k5[idx].append(k5_id)
                                
        num_k5 = len(k5_triples)
        
        # Simulated Annealing
        # State: colors (0=red, 1=blue)
        # k4_sum[i]: number of red edges in K4 i. Violation if 4.
        # k5_sum[i]: number of blue edges in K5 i. Violation if 10.
        
        max_restarts = 20
        steps_per_restart = 5000
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize colors
            # Heuristic: Random
            colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
            
            # Initialize sums
            k4_sum = np.zeros(num_k4, dtype=np.int8)
            k5_sum = np.zeros(num_k5, dtype=np.int8)
            
            # Calculate initial sums
            # Vectorized:
            # k4_sum = sum of (1-colors) over k4_triples
            # k5_sum = sum of colors over k5_triples
            
            # Create sparse-like matrices or just loop?
            # Loop is O(num_k4 * 4 + num_k5 * 10) which is ~ 46k*4 + 278k*10 ~ 3M ops.
            # In Python, 3M ops is ~0.1-0.5s. Acceptable for init.
            
            for i, idxs in enumerate(k4_triples):
                # Red is 0. Count reds = count of 0s = 4 - count of 1s.
                # Or just sum (1 - colors[idx])
                # Since we want red count:
                # k4_sum[i] = np.sum(1 - colors[idxs])
                # This is slow in Python loop.
                # Let's use a trick:
                # Pass 1: Compute k4_sum and k5_sum using NumPy if possible?
                # Matrices are sparse.
                
                # Faster init:
                # Just iterate triples and add to relevant sets.
                pass
            
            # Reset and do it efficiently
            k4_sum = np.zeros(num_k4, dtype=np.int8)
            k5_sum = np.zeros(num_k5, dtype=np.int8)
            
            for t_idx in range(num_triples):
                c = colors[t_idx]
                # Update k4 sums
                for k4_id in t_to_k4[t_idx]:
                    if c == 0: # Red
                        k4_sum[k4_id] += 1
                # Update k5 sums
                for k5_id in t_to_k5[t_idx]:
                    if c == 1: # Blue
                        k5_sum[k5_id] += 1
                        
            num_red_viol = np.sum(k4_sum == 4)
            num_blue_viol = np.sum(k5_sum == 10)
            current_cost = num_red_viol + num_blue_viol
            
            if current_cost == 0:
                best_n = n
                best_colors = colors.tolist()
                return {'n': n, 'colors': best_colors}
                
            T = 1.0
            T_min = 0.01
            cooling = 0.995
            
            for step in range(steps_per_restart):
                if time.time() - start_time > time_limit:
                    break
                    
                # Pick a random triple to flip
                t_idx = np.random.randint(0, num_triples)
                old_c = colors[t_idx]
                new_c = 1 - old_c
                
                # Calculate delta cost
                delta_cost = 0
                
                # Update K4s
                for k4_id in t_to_k4[t_idx]:
                    old_sum = k4_sum[k4_id]
                    if old_c == 0: # Was Red, becoming Blue
                        new_sum = old_sum - 1
                        if old_sum == 4: # Was violation, now not
                            delta_cost -= 1
                        # if new_sum == 4: impossible since old was 4 and we subtracted
                    else: # Was Blue, becoming Red
                        new_sum = old_sum + 1
                        if new_sum == 4: # Was not violation, now is
                            delta_cost += 1
                        # if old_sum == 4: impossible since old was 3 (max non-violation)
                    
                    k4_sum[k4_id] = new_sum
                    
                # Update K5s
                for k5_id in t_to_k5[t_idx]:
                    old_sum = k5_sum[k5_id]
                    if old_c == 1: # Was Blue, becoming Red
                        new_sum = old_sum - 1
                        if old_sum == 10: # Was violation, now not
                            delta_cost -= 1
                    else: # Was Red, becoming Blue
                        new_sum = old_sum + 1
                        if new_sum == 10: # Was not violation, now is
                            delta_cost += 1
                    
                    k5_sum[k5_id] = new_sum
                
                # Acceptance
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    colors[t_idx] = new_c
                    current_cost += delta_cost
                else:
                    # Revert sums
                    for k4_id in t_to_k4[t_idx]:
                        if new_c == 0: # Became Red, reverting to Blue
                            k4_sum[k4_id] -= 1
                        else:
                            k4_sum[k4_id] += 1
                    for k5_id in t_to_k5[t_idx]:
                        if new_c == 1: # Became Blue, reverting to Red
                            k5_sum[k5_id] -= 1
                        else:
                            k5_sum[k5_id] += 1
                
                if current_cost == 0:
                    best_n = n
                    best_colors = colors.tolist()
                    return {'n': n, 'colors': best_colors}
                    
                T *= cooling
                if T < T_min:
                    T = T_min

    # If we didn't find 34, we might have found a lower n in the loop
    # But the loop breaks on time. 
    # If we exit the loop without returning, we need to return the best found.
    # However, the loop structure above only returns on success.
    # We should track best_n and best_colors in case we fail 34 but succeed 33.
    # The loop goes 34, 33, ...
    # If 34 fails, it tries 33.
    # If 33 succeeds, it returns.
    # If all fail, we return the best_n found (which might be 0 if none succeeded).
    
    if best_n == 0:
        # Fallback: Return a trivial valid coloring for small n
        # n=9 is always solvable?
        # Let's just return n=1
        n = 1
        triples = []
        colors = []
        best_n = n
        best_colors = colors

    return {'n': best_n, 'colors': best_colors}
