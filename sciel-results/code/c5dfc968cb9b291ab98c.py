"""Rewritten program with Structured Initialization and Efficient Local Search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    def build_precomputation(n):
        """
        Precomputes indices for fast violation counting.
        """
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4_list = list(combinations(range(n), 4))
        k5_list = list(combinations(range(n), 5))
        
        # Map each K4 to its 4 constituent triple indices
        k4_triples = []
        for k4 in k4_list:
            ts = []
            for i in range(4):
                t = tuple(sorted(k4[:i] + k4[i+1:]))
                ts.append(triple_to_idx[t])
            k4_triples.append(ts)
        k4_arr = np.array(k4_triples, dtype=np.int32)
        
        # Map each K5 to its 10 constituent triple indices
        k5_triples = []
        for k5 in k5_list:
            ts = []
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[i] for i in combo]))
                ts.append(triple_to_idx[t])
            k5_triples.append(ts)
        k5_arr = np.array(k5_triples, dtype=np.int32)
        
        # Inverse mapping: for each triple, which K4s and K5s does it belong to?
        triple_to_k4s = [[] for _ in range(num_triples)]
        for k4_idx, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4s[t_idx].append(k4_idx)
        
        triple_to_k5s = [[] for _ in range(num_triples)]
        for k5_idx, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5s[t_idx].append(k5_idx)
                
        return triples, num_triples, k4_arr, k5_arr, triple_to_k4s, triple_to_k5s

    def run_local_search(n, time_budget, seed_offset=0):
        """
        Runs Local Search with Structured Initialization.
        """
        triples, num_triples, k4_arr, k5_arr, triple_to_k4s, triple_to_k5s = build_precomputation(n)
        
        # Structured Initialization
        # Try different modular patterns to find a good starting point
        # Pattern: color = (a + b + c) % m < k
        # For R(4,5;3), red is K4, blue is K5.
        # We want to avoid all-red K4s and all-blue K5s.
        # A common heuristic is to balance the colors.
        
        # Let's try a few different structured starts
        best_v = float('inf')
        best_colors = None
        
        # We will try a few distinct initializations
        initializations = []
        
        # 1. Random
        initializations.append('random')
        
        # 2. Modular 3: (a+b+c)%3 == 0 -> Red, else Blue
        initializations.append('mod3_0')
        # 3. Modular 3: (a+b+c)%3 == 1 -> Red, else Blue
        initializations.append('mod3_1')
        # 4. Modular 5: (a+b+c)%5 < 2 -> Red, else Blue
        initializations.append('mod5_2')
        # 5. Modular 5: (a+b+c)%5 < 3 -> Red, else Blue
        initializations.append('mod5_3')
        # 6. Modular 7: (a+b+c)%7 < 3 -> Red, else Blue
        initializations.append('mod7_3')

        for init_type in initializations:
            if time.time() > start_time + time_budget:
                break
                
            # Generate initial colors
            colors = np.zeros(num_triples, dtype=np.int8)
            
            if init_type == 'random':
                colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
            else:
                parts = init_type.split('_')
                m = int(parts[0][3:])
                k = int(parts[1])
                for i, (a, b, c) in enumerate(triples):
                    if (a + b + c) % m < k:
                        colors[i] = 0  # Red
                    else:
                        colors[i] = 1  # Blue
            
            # Compute initial violations
            # Red K4 violation if sum of colors in K4 is 0
            # Blue K5 violation if sum of colors in K5 is 10
            
            # We maintain counts of red edges in each K4 and blue edges in each K5
            # k4_red_counts: number of red edges in each K4. Violation if 0.
            # k5_blue_counts: number of blue edges in each K5. Violation if 10.
            
            k4_red_counts = np.sum(1 - colors[k4_arr], axis=1) # 1 if red (0), 0 if blue (1)
            k5_blue_counts = np.sum(colors[k5_arr], axis=1)    # 1 if blue (1), 0 if red (0)
            
            num_red_violations = np.sum(k4_red_counts == 0)
            num_blue_violations = np.sum(k5_blue_counts == 10)
            current_v = num_red_violations + num_blue_violations
            
            if current_v < best_v:
                best_v = current_v
                best_colors = colors.copy()
                best_k4_red_counts = k4_red_counts.copy()
                best_k5_blue_counts = k5_blue_counts.copy()
                
                if best_v == 0:
                    return n, best_colors.tolist()
            
            # If we found a good start, run local search from it
            if current_v < best_v * 0.8 or current_v < 100: # Only search if start is promising
                # Local Search
                T = 10.0
                T_end = 0.1
                cooling_rate = 0.999
                no_improve = 0
                max_no_improve = 5000
                
                # Copy states
                cur_colors = colors.copy()
                cur_k4_red = k4_red_counts.copy()
                cur_k5_blue = k5_blue_counts.copy()
                cur_v = current_v
                
                while time.time() < start_time + time_budget:
                    # Pick a random triple
                    idx = random.randint(0, num_triples - 1)
                    old_val = cur_colors[idx]
                    new_val = 1 - old_val
                    
                    # Calculate delta in violations
                    delta_v = 0
                    
                    # Update K4s
                    for k4_idx in triple_to_k4s[idx]:
                        old_count = cur_k4_red[k4_idx]
                        # If old_val was Red (0), count was 1 for this edge. Now it's Blue (1), so count decreases by 1.
                        # If old_val was Blue (1), count was 0 for this edge. Now it's Red (0), so count increases by 1.
                        if old_val == 0: # was Red
                            new_count = old_count - 1
                        else: # was Blue
                            new_count = old_count + 1
                        
                        # Check violation change
                        was_violation = (old_count == 0)
                        is_violation = (new_count == 0)
                        
                        if was_violation and not is_violation:
                            delta_v -= 1
                        elif not was_violation and is_violation:
                            delta_v += 1
                            
                        cur_k4_red[k4_idx] = new_count
                        
                    # Update K5s
                    for k5_idx in triple_to_k5s[idx]:
                        old_count = cur_k5_blue[k5_idx]
                        # If old_val was Blue (1), count was 1 for this edge. Now it's Red (0), so count decreases by 1.
                        # If old_val was Red (0), count was 0 for this edge. Now it's Blue (1), so count increases by 1.
                        if old_val == 1: # was Blue
                            new_count = old_count - 1
                        else: # was Red
                            new_count = old_count + 1
                        
                        # Check violation change
                        was_violation = (old_count == 10)
                        is_violation = (new_count == 10)
                        
                        if was_violation and not is_violation:
                            delta_v -= 1
                        elif not was_violation and is_violation:
                            delta_v += 1
                            
                        cur_k5_blue[k5_idx] = new_count
                        
                    new_v = cur_v + delta_v
                    
                    if new_v < cur_v:
                        cur_v = new_v
                        cur_colors[idx] = new_val
                        no_improve = 0
                        
                        if cur_v < best_v:
                            best_v = cur_v
                            best_colors = cur_colors.copy()
                            best_k4_red_counts = cur_k4_red.copy()
                            best_k5_blue_counts = cur_k5_blue.copy()
                            
                            if best_v == 0:
                                return n, best_colors.tolist()
                    else:
                        # Accept with probability exp(-delta/T)
                        if delta_v > 0:
                            if T > T_end:
                                prob = np.exp(-delta_v / T)
                                if random.random() < prob:
                                    cur_v = new_v
                                    cur_colors[idx] = new_val
                                    no_improve += 1
                                else:
                                    no_improve += 1
                            else:
                                no_improve += 1
                        else:
                            # delta_v == 0
                            cur_v = new_v
                            cur_colors[idx] = new_val
                            no_improve += 1
                            
                    T *= cooling_rate
                    
                    if no_improve > max_no_improve:
                        T = 10.0
                        no_improve = 0
                        
        if best_v == 0:
            return n, best_colors.tolist()
        else:
            return None, None

    # Strategy:
    # 1. Try n=34 with multiple restarts/initializations.
    # 2. If fail, try n=33, etc.
    
    # We allocate time to n=34 primarily.
    # The local search is fast, so we can run many iterations.
    
    time_budget_34 = 120.0
    res_n, res_colors = run_local_search(34, time_budget_34)
    if res_n is not None:
        return {'n': res_n, 'colors': res_colors}
        
    # If n=34 failed, try n=33 with remaining time
    remaining = deadline - time.time()
    if remaining > 10:
        res_n, res_colors = run_local_search(33, remaining - 5)
        if res_n is not None:
            return {'n': res_n, 'colors': res_colors}
            
    # Fallback: n=9
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

# EVOLVE-BLOCK-END
