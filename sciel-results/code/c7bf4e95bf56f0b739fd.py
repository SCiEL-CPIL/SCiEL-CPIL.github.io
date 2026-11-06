import time
import sys
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin for 60s limit
    max_val = 4096
    
    # Helper to check if adding x is valid
    def can_add(S, used_sums, x):
        k = len(S)
        # Generate new sums
        new_sums = set()
        
        # 1. x + x + x
        s1 = 3 * x
        if s1 in used_sums or s1 in new_sums:
            return False
        new_sums.add(s1)
        
        # 2. x + x + s
        for s in S:
            s2 = 2 * x + s
            if s2 in used_sums or s2 in new_sums:
                return False
            new_sums.add(s2)
            
        # 3. x + s_i + s_j
        for i in range(k):
            for j in range(i, k):
                s3 = x + S[i] + S[j]
                if s3 in used_sums or s3 in new_sums:
                    return False
                new_sums.add(s3)
                
        return True

    def add_to_state(S, used_sums, x):
        k = len(S)
        S_k = S[:-1] # Assuming x was just appended
        sums_to_add = []
        sums_to_add.append(3 * x)
        for s in S_k:
            sums_to_add.append(2 * x + s)
        for i in range(len(S_k)):
            for j in range(i, len(S_k)):
                sums_to_add.append(x + S_k[i] + S_k[j])
        for s_val in sums_to_add:
            used_sums.add(s_val)
        return sums_to_add

    def remove_from_state(S, used_sums, x, sums_to_remove):
        for s_val in sums_to_remove:
            used_sums.remove(s_val)

    # Fast Greedy first to get a good initial bound
    S = []
    used_sums = set()
    
    for x in range(1, max_val + 1):
        if can_add(S, used_sums, x):
            S.append(x)
            add_to_state(S, used_sums, x)
            
    best_set = list(S)
    best_len = len(S)
    
    # Local Search: Try to replace elements with larger ones to "free up" space or pack tighter?
    # Actually, for B_3[1], we want to maximize count. 
    # A common heuristic is to try to swap a small element for a larger one if it allows adding more elements later.
    # Or simply, try to "shift" the set.
    
    # Let's try a simple local search: 
    # 1. Take the current set.
    # 2. For each element, try to remove it and add a different one? No, that keeps size constant.
    # 3. Try to remove one element and add two? That increases size by 1.
    
    # Let's implement a "swap" based local search that tries to increase the set size.
    # Or better, a "rebuild" local search.
    
    # Given the time limit, let's focus on randomized restarts with different strategies.
    # The previous code had a loop for random restarts but it was essentially identical to greedy (no actual randomization in choice).
    # Let's implement proper randomized greedy:
    # At each step, among valid candidates, pick one randomly? 
    # Or: with probability p, skip a valid candidate to allow different future choices.
    
    # Also, let's try "backward" greedy? No, forward is standard.
    
    # Let's try multiple random restarts with "skipping" probability.
    
    for attempt in range(100):
        if time.time() - start_time > DEADLINE:
            break
            
        # Randomize the "skip" probability
        # Higher skip prob -> more exploration, potentially larger sets if lucky
        p_skip = random.uniform(0.0, 0.2)
        
        S_rand = []
        used_sums_rand = set()
        
        for x in range(1, max_val + 1):
            if can_add(S_rand, used_sums_rand, x):
                # With probability p_skip, skip this valid candidate
                if random.random() > p_skip:
                    S_rand.append(x)
                    add_to_state(S_rand, used_sums_rand, x)
                    
        if len(S_rand) > best_len:
            best_set = list(S_rand)
            best_len = len(S_rand)
            
        # If we have found a large set, we can try to polish it with local search
        if best_len >= 17:
            # Simple local search: try to swap elements to free up sums?
            # This is complex. Let's stick to restarts for now.
            pass

    return best_set
