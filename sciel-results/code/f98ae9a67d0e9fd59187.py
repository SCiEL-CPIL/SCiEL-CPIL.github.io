import random
import math
import time

def _fast_score(S):
    S_set = set(S)
    valid_diffs = set()
    lst = list(S)
    n = len(lst)
    for i in range(n):
        x = lst[i]
        for j in range(i + 1, n):
            y = lst[j]
            d_val = y - x
            
            if (x - d_val) in S_set:
                valid_diffs.add(d_val)
            if (y + d_val) in S_set:
                valid_diffs.add(d_val)
            
            if d_val % 2 == 0:
                d_half = d_val // 2
                if (x + d_half) in S_set:
                    valid_diffs.add(d_half)
                    
    return len(valid_diffs)

def construct():
    # Use multiple restarts to avoid local minima
    # Time limit 120s. We will use ~10s per run for ~10 runs? 
    # Or just one long run with better tuning.
    # Let's try 5 runs of 2s each.
    
    best_score = 0
    best_set = set()
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    run_id = 0
    while time.time() - start_time < time_limit:
        run_id += 1
        # Different initialization strategies
        if run_id % 2 == 0:
            # Dense interval start
            current = set(range(1, 33))
        else:
            # Random start
            current = set(random.sample(range(1, 513), 32))
        
        current_score = _fast_score(current)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current)
        
        # SA parameters
        T_start = 5.0
        T_end = 0.1
        # Estimate iterations based on time
        # Each iteration ~0.1ms? 500k iters ~ 50s? 
        # Let's do 100,000 iterations per run.
        iterations = 100000
        alpha = (T_end / T_start) ** (1.0 / iterations)
        T = T_start
        
        all_vals = list(range(1, 513))
        
        # Convert to list for faster random access? 
        # We need to swap elements.
        # Let's maintain a list `S_list` and a set `S_set`.
        S_list = list(current)
        S_set = set(current)
        
        for i in range(iterations):
            # Pick random index
            idx = random.randrange(32)
            old_elem = S_list[idx]
            
            # Pick new element
            # Ensure it's not in S_set
            while True:
                new_elem = random.choice(all_vals)
                if new_elem not in S_set:
                    break
            
            # Perform swap
            S_set.remove(old_elem)
            S_set.add(new_elem)
            S_list[idx] = new_elem
            
            new_score = _fast_score(S_set)
            
            # Acceptance
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(S_set)
            else:
                delta = new_score - current_score
                if delta < 0:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_score = new_score
                # If delta == 0, accept always? 
                # Standard SA: accept if exp(delta/T) > random. 
                # If delta=0, exp(0)=1, always accept.
                else:
                    current_score = new_score
            
            T *= alpha
            
            # Check time within run? 
            # If we've been running too long for this run, break.
            # 100k iters * ~1ms = 100s. Too slow.
            # We need to reduce iterations or speed up score.
            
            # Let's break if time per run exceeds 10s
            if time.time() - start_time > time_limit - 5.0:
                break
        
        # End of run
        
    return list(best_set)
