import random
import math
import time
import numpy as np

def construct():
    n = 256
    
    # Precompute 3-APs and 4-APs
    ap3s = []
    for d in range(1, n):
        for a in range(0, n - 2 * d):
            ap3s.append((a, a + d, a + 2 * d))
            
    ap4s = []
    for d in range(1, n):
        for a in range(0, n - 3 * d):
            ap4s.append((a, a + d, a + 2 * d, a + 3 * d))
            
    num_3aps = len(ap3s)
    num_4aps = len(ap4s)
    
    # For each element, precompute which 3-APs and 4-APs it belongs to
    elem_3aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap3s):
        for i in ap:
            elem_3aps[i].append(idx)
            
    elem_4aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap4s):
        for i in ap:
            elem_4aps[i].append(idx)
    
    def count_3aps(state):
        count = 0
        for ap in ap3s:
            if all(state[i] == 1 for i in ap):
                count += 1
        return count
        
    def count_4aps(state):
        count = 0
        for ap in ap4s:
            if all(state[i] == 1 for i in ap):
                count += 1
        return count
    
    def compute_delta(idx, state):
        """Compute delta_3 and delta_4 if we flip element idx"""
        old_val = state[idx]
        new_val = 1 - old_val
        
        delta_3 = 0
        for ap_idx in elem_3aps[idx]:
            ap = ap3s[ap_idx]
            others_present = sum(1 for j in ap if j != idx and state[j] == 1)
            if new_val == 1 and others_present == 2:
                delta_3 += 1
            elif new_val == 0 and others_present == 2:
                delta_3 -= 1
        
        delta_4 = 0
        for ap_idx in elem_4aps[idx]:
            ap = ap4s[ap_idx]
            others_present = sum(1 for j in ap if j != idx and state[j] == 1)
            if new_val == 1 and others_present == 3:
                delta_4 += 1
            elif new_val == 0 and others_present == 3:
                delta_4 -= 1
        
        return delta_3, delta_4
    
    def greedy_construct():
        """Greedy construction prioritizing high 3-AP potential"""
        state = [0] * n
        # Sort by number of 3-APs the element participates in
        potentials = [len(elem_3aps[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: potentials[i], reverse=True)
        
        for i in order:
            # Check if adding i creates any 4-AP
            can_add = True
            for ap_idx in elem_4aps[i]:
                ap = ap4s[ap_idx]
                others_in_set = sum(1 for j in ap if j != i and state[j] == 1)
                if others_in_set == 3:
                    can_add = False
                    break
            if can_add:
                state[i] = 1
        return state
    
    def local_search(state, time_limit):
        """Iterative improvement local search"""
        start_time = time.time()
        current_3ap = count_3aps(state)
        current_4ap = count_4aps(state)
        
        if current_4ap > 0:
            # Repair: remove elements from 4-APs
            changed = True
            while changed:
                changed = False
                for ap in ap4s:
                    if all(state[i] == 1 for i in ap):
                        # Remove element with lowest 3-AP contribution
                        min_contrib = float('inf')
                        victim = ap[0]
                        for i in ap:
                            # Count how many 3-APs would be lost
                            contrib = 0
                            for ap_idx in elem_3aps[i]:
                                other_3ap = ap3s[ap_idx]
                                if all(state[j] == 1 for j in other_3ap if j != i):
                                    contrib += 1
                            if contrib < min_contrib:
                                min_contrib = contrib
                                victim = i
                        state[victim] = 0
                        changed = True
            current_3ap = count_3aps(state)
            current_4ap = count_4aps(state)
        
        best_state = state[:]
        best_score = current_3ap
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for idx in range(n):
                delta_3, delta_4 = compute_delta(idx, state)
                
                # Only accept if no new 4-APs created and 3-AP count improves
                if delta_4 > 0:
                    continue
                if delta_3 > 0:
                    state[idx] = 1 - state[idx]
                    current_3ap += delta_3
                    current_4ap += delta_4
                    improved = True
                    if current_4ap == 0 and current_3ap > best_score:
                        best_score = current_3ap
                        best_state = state[:]
        
        return best_state, best_score
    
    # Run multiple restarts with different strategies
    start_time = time.time()
    max_time = 55.0
    
    best_overall_state = None
    best_overall_score = -1
    
    # Strategy 1: Greedy
    state = greedy_construct()
    state, score = local_search(state, time_limit=max_time - (time.time() - start_time) - 5)
    if score > best_overall_score:
        best_overall_score = score
        best_overall_state = state[:]
    
    # Strategy 2: Random initialization + local search
    remaining_time = max_time - (time.time() - start_time)
    if remaining_time > 10:
        state = [1 if random.random() < 0.4 else 0 for _ in range(n)]
        state, score = local_search(state, time_limit=remaining_time - 5)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_state = state[:]
    
    # Strategy 3: Simulated annealing from best state
    remaining_time = max_time - (time.time() - start_time)
    if remaining_time > 10 and best_overall_state is not None:
        state = best_overall_state[:]
        current_3ap = count_3aps(state)
        current_4ap = count_4aps(state)
        
        T = 30.0
        T_min = 0.1
        alpha = 0.999
        penalty = 10000
        best_state = state[:]
        best_score = current_3ap - penalty * current_4ap
        
        sa_start = time.time()
        while T > T_min and (time.time() - sa_start) < remaining_time - 2:
            idx = random.randint(0, n - 1)
            delta_3, delta_4 = compute_delta(idx, state)
            delta_score = delta_3 - penalty * delta_4
            
            if delta_score >= 0:
                accept = True
            else:
                accept = random.random() < math.exp(delta_score / T)
                
            if accept:
                state[idx] = 1 - state[idx]
                current_3ap += delta_3
                current_4ap += delta_4
                current_score = current_3ap - penalty * current_4ap
                
                if current_score > best_score:
                    best_score = current_score
                    best_state = state[:]
                    
            T *= alpha
        
        if count_4aps(best_state) == 0:
            final_score = count_3aps(best_state)
            if final_score > best_overall_score:
                best_overall_score = final_score
                best_overall_state = best_state[:]
    
    # Final fallback if nothing worked
    if best_overall_state is None:
        best_overall_state = [1] * n  # Just return first element
    
    # Final verification
    if count_4aps(best_overall_state) > 0:
        changed = True
        while changed:
            changed = False
            for ap in ap4s:
                if all(best_overall_state[i] == 1 for i in ap):
                    victim = random.choice(ap)
                    best_overall_state[victim] = 0
                    changed = True
    
    result = [i + 1 for i in range(n) if best_overall_state[i] == 1]
    return result
