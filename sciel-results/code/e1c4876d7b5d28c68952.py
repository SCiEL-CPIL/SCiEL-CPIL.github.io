import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_score = -1
    best_set = [0]
    
    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        sumset = set()
        diffset = set()
        
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Exhaustive search for small M
    # M up to 14 is 2^14 = 16384 subsets.
    # We iterate through all subsets of {1, ..., M} and add 0.
    for M in range(1, 15):
        if time.time() > deadline - 5:
            break
        for mask in range(1, 1 << M):
            if time.time() > deadline - 5:
                break
            cand = [0]
            for i in range(M):
                if mask & (1 << i):
                    cand.append(i + 1)
            try_set(cand)
            
    # Phase 2: Simulated Annealing
    current = list(best_set)
    current_score = best_score
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    alpha = 0.99
    
    # If exhaustive search didn't find anything good (unlikely), start with a random set
    if len(current) <= 1:
        current = [0, 1, 2, 3, 5]
        current_score = compute_score(current)
        try_set(current)
        current = list(best_set)
        current_score = best_score

    T = T_start
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        # Generate a neighbor
        move_type = random.random()
        
        if move_type < 0.3:
            # Add an element
            # Choose from 1 to max(current) + 10 or random larger
            max_cur = max(current)
            if random.random() < 0.5:
                new_el = random.randint(1, max_cur + 10)
            else:
                new_el = random.randint(max_cur + 1, max_cur + 50)
            
            if new_el not in current:
                neighbor = current + [new_el]
            else:
                continue
                
        elif move_type < 0.6:
            # Remove an element
            if len(current) > 1:
                idx = random.randint(1, len(current) - 1) # Don't remove 0
                neighbor = [x for i, x in enumerate(current) if i != idx]
            else:
                continue
                
        else:
            # Replace an element
            if len(current) > 1:
                idx = random.randint(1, len(current) - 1)
                old_el = current[idx]
                # New value near old value
                delta = random.randint(-10, 10)
                new_el = old_el + delta
                if new_el < 1:
                    new_el = 1
                if new_el not in current:
                    neighbor = list(current)
                    neighbor[idx] = new_el
                    neighbor = sorted(set(neighbor))
                else:
                    continue
            else:
                continue

        neighbor_score = compute_score(neighbor)
        
        if neighbor_score > current_score:
            current = neighbor
            current_score = neighbor_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_set = list(current)
        else:
            # Accept worse move with probability exp(delta / T)
            delta = current_score - neighbor_score
            if delta < 0:
                accept_prob = 1.0
            else:
                accept_prob = math.exp(-delta / T)
                
            if random.random() < accept_prob:
                current = neighbor
                current_score = neighbor_score
            no_improve += 1
        
        T *= alpha
        if T < T_end:
            T = T_end
            # Restart from best if stuck
            if no_improve > 100:
                current = list(best_set)
                current_score = best_score
                no_improve = 0
                T = T_start * 0.5

    return list(best_set)
